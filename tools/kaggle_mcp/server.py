"""
MCP server that reads, runs and debugs the owner's Kaggle notebooks and runs bounded tuning.

Wraps the Kaggle CLI, which handles authentication. Pushes are only allowed to the notebook
named by KAGGLE_MCP_NOTEBOOK, because a push replaces a notebook's code.
"""
import collections
import copy
import csv
import io
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / ".kaggle_runs"
CLI_TIMEOUT = 300
MAX_TRIALS_LIMIT = 12
STALE_SECONDS = 90
RUN_TIMEOUT_SECONDS = 13 * 3600  # longer than Kaggle's session limit, so the run is certainly over
METRICS_PATTERN = r"metrics\.json$"
ACTIVE_STATES = ("pending", "pushing", "running")

mcp = MCPServer("kaggle")
LOCK = threading.RLock()  # tools run in worker threads, and every push must see the latest state


class KaggleError(Exception):
    pass


class KaggleTimeout(KaggleError):
    """The CLI gave no answer, so a push may or may not have been accepted."""


def scrub(text: str) -> str:
    return re.sub(r"KGAT_?[A-Za-z0-9_-]+", "[REDACTED]", text)


def validate_ref(ref: str) -> str:
    if not re.fullmatch(r"[\w-]+/[\w-]+", ref or "") or ref.startswith("-"):
        raise KaggleError(f"Invalid notebook reference: {ref!r}. Expected owner/notebook-name.")
    return ref


def require_dedicated(ref: str) -> None:
    allowed = os.environ.get("KAGGLE_MCP_NOTEBOOK")
    if not allowed or ref != allowed:
        raise KaggleError(f"Refusing to push to {ref!r}. Only the dedicated notebook {allowed!r} can be pushed.")


def run_cli(args: list[str]) -> str:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        result = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=CLI_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        raise KaggleTimeout(f"Kaggle CLI timed out after {CLI_TIMEOUT}s: {args[:2]}")
    if result.returncode != 0:
        raise KaggleError(scrub(result.stderr or result.stdout or f"Kaggle CLI exited {result.returncode}").strip())
    return scrub(result.stdout)


def write_metadata(folder: Path, ref: str, code_file: str, dataset_sources: list[str]) -> Path:
    path = Path(folder) / "kernel-metadata.json"
    path.write_text(json.dumps({
        "id": ref,
        "title": ref.split("/")[1],
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2), encoding="utf-8")
    return path


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def read_cells(path: Path) -> list[dict]:
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{"number": i, "type": c["cell_type"], "source": scrub(source_text(c))}
            for i, c in enumerate(notebook["cells"], 1)]


def log_text(raw: str) -> str:
    """Kaggle logs are a JSON array of {"data": ...}; anything else is treated as plain text."""
    try:
        text = "".join(entry.get("data", "") for entry in json.loads(raw))
    except (ValueError, TypeError, AttributeError):
        text = raw
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def find_error(log: str, cells: list[dict]) -> dict | None:
    text = log_text(log)
    matches = list(re.finditer(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit))(?::[ \t]*(.*))?$", text, re.M))
    if not matches:
        return None
    last = matches[-1]
    start = text.rfind("Traceback", 0, last.start())
    lines = text[max(start, 0) if start != -1 else last.start():last.end()].splitlines()
    code = [c for c in cells if c["type"] == "code"]
    found = None
    # IPython's arrow marks the executing line, and the first frame found in a cell is the user's call.
    for line in sorted(lines, key=lambda l: "->" not in l):
        stripped = re.sub(r"^\s*(?:-*>\s*)?(?:\d+\s+)?", "", line).strip()
        found = next((c["number"] for c in code if len(stripped) > 3
                      and stripped in (l.strip() for l in c["source"].splitlines())), None)
        if found:
            break
    return {"type": last.group(1), "message": (last.group(2) or "").strip(), "traceback": "\n".join(lines), "cell": found}


def push(path: Path, ref: str) -> str:
    require_dedicated(validate_ref(ref))
    path = Path(path)
    if path.suffix != ".ipynb" or not path.is_file():
        raise KaggleError(f"Not a notebook file: {path}")
    datasets = json.loads(path.read_text(encoding="utf-8")).get("metadata", {}).get("kaggle_mcp", {}).get("dataset_sources", [])
    with tempfile.TemporaryDirectory() as folder:
        shutil.copy(path, Path(folder) / "notebook.ipynb")
        write_metadata(Path(folder), ref, "notebook.ipynb", datasets)
        out = run_cli(["kernels", "push", "-p", folder])
    # The CLI can report a rejected push on stdout and still exit 0.
    if "successfully pushed" not in out:
        raise KaggleError(f"Kaggle did not accept the push: {out.strip()}")
    return out.strip()


def run_folder(ref: str, name: str) -> Path:
    folder = RUNS / ref.split("/")[1] / name
    shutil.rmtree(folder, ignore_errors=True)  # stale files must not be read as this run's results
    folder.mkdir(parents=True)
    return folder


def pulled_cells(ref: str) -> list[dict]:
    folder = run_folder(validate_ref(ref), "source")
    run_cli(["kernels", "pull", ref, "-p", str(folder)])
    notebooks = list(folder.glob("*.ipynb"))
    if not notebooks:
        raise KaggleError(f"{ref} has no notebook source to read.")
    return read_cells(notebooks[0])


@mcp.tool()
def list_notebooks() -> list[dict]:
    """List the owner's Kaggle notebooks."""
    # shortcut: first 100 notebooks only, add paging if the account outgrows that
    return list(csv.DictReader(io.StringIO(run_cli(["kernels", "list", "--mine", "--csv", "--page-size", "100"]))))


@mcp.tool()
def read_notebook(ref: str) -> list[dict]:
    """Return a notebook's numbered cells."""
    return pulled_cells(ref)


@mcp.tool()
def run_notebook(path: str) -> str:
    """Push a local .ipynb to the dedicated notebook and start a run of the whole notebook."""
    ref = os.environ.get("KAGGLE_MCP_NOTEBOOK", "")
    require_dedicated(ref)
    with LOCK:
        refuse_if_tuning_active()
        return push(Path(path), ref)


@mcp.tool()
def run_status(ref: str) -> str:
    """Return queued, running, complete, error or cancelled for a notebook's latest run."""
    out = run_cli(["kernels", "status", validate_ref(ref)]).lower()
    if "cancel" in out:
        return "cancelled"
    found = re.search(r"queued|running|complete|error", out)
    if not found:
        raise KaggleError(f"Unrecognised Kaggle status: {out.strip()}")
    return found.group(0)


@mcp.tool()
def check_errors(ref: str) -> dict:
    """Find the exception in a notebook's latest run and the cell it came from, if one matches."""
    error = find_error(run_cli(["kernels", "logs", validate_ref(ref)]), pulled_cells(ref))
    return error or {"message": "no error found"}


def download_results(ref: str, file_pattern: str | None = None) -> dict:
    folder = run_folder(validate_ref(ref), "output")
    args = ["kernels", "output", ref, "-p", str(folder), "-o"]
    run_cli(args + ["--file-pattern", file_pattern] if file_pattern else args)
    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    metrics_file = next(folder.rglob("metrics.json"), None)
    try:
        metrics = json.loads(scrub(metrics_file.read_text(encoding="utf-8"))) if metrics_file else None
    except ValueError:
        metrics = None
    return {"folder": str(folder), "files": files, "metrics": metrics}


@mcp.tool()
def get_results(ref: str) -> dict:
    """Download a notebook's output files and return their names and metrics.json."""
    return download_results(ref)


def expand_grid(grid: dict[str, list], max_trials: int) -> list[dict]:
    if not 1 <= max_trials <= MAX_TRIALS_LIMIT:
        raise KaggleError(f"max_trials must be between 1 and {MAX_TRIALS_LIMIT}.")
    if not grid or not all(isinstance(v, list) and v for v in grid.values()):
        raise KaggleError("Grid must map each parameter name to a non-empty list.")
    for name, values in grid.items():
        if not name.isidentifier() or not all(isinstance(v, (bool, int, float, str)) for v in values):
            raise KaggleError(f"Invalid grid entry: {name!r}.")
    trials = [dict(zip(grid, combo)) for combo in itertools.product(*grid.values())]
    if len(trials) > max_trials:
        raise KaggleError(f"Grid has {len(trials)} trials, more than max_trials ({max_trials}).")
    return trials


def set_parameters(notebook: dict, params: dict) -> dict:
    out = copy.deepcopy(notebook)
    for cell in out["cells"]:
        if cell["cell_type"] == "code" and "parameters" in cell.get("metadata", {}).get("tags", []):
            cell["source"] = "".join(f"{name} = {value!r}\n" for name, value in params.items())
            return out
    raise KaggleError("Notebook has no code cell tagged 'parameters'.")


def tuning_path(tuning_id: str, suffix: str = ".json") -> Path:
    if not re.fullmatch(r"[\w-]+", tuning_id or ""):
        raise KaggleError(f"Invalid tuning id: {tuning_id!r}.")
    return RUNS / "tuning" / f"{tuning_id}{suffix}"


def save_state(tuning_id: str, state: dict) -> None:
    path = tuning_path(tuning_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temp, path)


def load_state(tuning_id: str) -> dict:
    try:
        state = json.loads(tuning_path(tuning_id).read_text(encoding="utf-8"))
        state["trials"], state["ref"], state["stopped"]
    except OSError:
        raise KaggleError(f"Unknown tuning id: {tuning_id!r}.")
    except (ValueError, KeyError, TypeError):
        raise KaggleError(f"Tuning state for {tuning_id!r} is corrupt.")
    return state


def launch(tuning_id: str, state: dict, trial: dict) -> None:
    notebook = json.loads(tuning_path(tuning_id, ".ipynb").read_text(encoding="utf-8"))
    prepared = set_parameters(notebook, {"TRIAL_ID": trial["id"], **trial["params"]})
    # Saved first: if the process dies mid-push, the next call must not push this trial again.
    trial.update(state="pushing", pushed_at=time.time())
    save_state(tuning_id, state)
    try:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "notebook.ipynb"
            path.write_text(json.dumps(prepared), encoding="utf-8")
            push(path, state["ref"])
    except KaggleTimeout as error:
        stop(tuning_id, state, f"Push outcome unknown, stopped to avoid a duplicate GPU run: {error}")
        raise
    except KaggleError:
        trial["state"] = "pending"
        save_state(tuning_id, state)
        raise
    trial.update(state="running", pushed_at=time.time())
    save_state(tuning_id, state)


def stop(tuning_id: str, state: dict, reason: str) -> None:
    state.update(stopped=True, stop_reason=reason)
    save_state(tuning_id, state)


def refuse_if_tuning_active() -> None:
    for path in (RUNS / "tuning").glob("*.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            active = not state["stopped"] and any(t["state"] in ACTIVE_STATES for t in state["trials"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if active:
            raise KaggleError(f"Tuning {path.stem} is still active, and a push would replace its running trial.")


def summary(tuning_id: str, state: dict, note: str = "") -> dict:
    counts = collections.Counter(t["state"] for t in state["trials"])
    return {"tuning_id": tuning_id, "stopped": state["stopped"], "counts": dict(counts), "note": note}


def record_outcome(state: dict, trial: dict, status: str) -> bool:
    """Updates the running trial. Returns False if it should stay running."""
    age = time.time() - trial["pushed_at"]
    if age < STALE_SECONDS:
        return False  # Kaggle can still be reporting the previous version's outcome
    if age > RUN_TIMEOUT_SECONDS:
        trial.update(state="failed", reason="no result before the run time limit")
    elif status in ("queued", "running"):
        return False
    elif status != "complete":
        trial.update(state="failed", reason=f"run ended with status {status}")
    else:
        metrics = download_results(state["ref"], METRICS_PATTERN)["metrics"]
        if isinstance(metrics, dict) and metrics.get("trial_id") not in (None, trial["id"]):
            return False  # output of an earlier version
        if valid_metrics(metrics):
            trial.update(state="done", metrics=metrics)
        else:
            trial.update(state="failed", reason="metrics.json needs trial_id and a numeric val_macro_f1")
    finished = [t["state"] for t in state["trials"] if t["state"] in ("done", "failed")]
    if finished[-2:] == ["failed", "failed"]:
        state.update(stopped=True, stop_reason="Stopped after two failed trials in a row.")
    return True


def valid_metrics(metrics) -> bool:
    if not isinstance(metrics, dict) or metrics.get("trial_id") is None:
        return False
    score = metrics.get("val_macro_f1")
    return isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)


@mcp.tool()
def start_tuning(path: str, grid: dict, max_trials: int = 8) -> dict:
    """Start a grid of trials on the dedicated notebook, one run per trial, and launch the first."""
    ref = os.environ.get("KAGGLE_MCP_NOTEBOOK", "")
    require_dedicated(ref)
    params = expand_grid(grid, max_trials)
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    set_parameters(notebook, {})
    with LOCK:
        refuse_if_tuning_active()
        tuning_id = f"tune-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        tuning_path(tuning_id).parent.mkdir(parents=True, exist_ok=True)
        tuning_path(tuning_id, ".ipynb").write_text(json.dumps(notebook), encoding="utf-8")
        state = {"notebook": path, "ref": ref, "stopped": False,
                 "trials": [{"id": f"{tuning_id}-{i}", "params": p, "state": "pending", "metrics": None}
                            for i, p in enumerate(params, 1)]}
        save_state(tuning_id, state)
        try:
            launch(tuning_id, state, state["trials"][0])
        except KaggleError as error:
            return summary(tuning_id, state, f"First push failed: {error}")
        return summary(tuning_id, state)


@mcp.tool()
def tuning_status(tuning_id: str) -> dict:
    """Move a tuning forward: record a finished trial and start the next one."""
    with LOCK:
        state = load_state(tuning_id)
        if state["stopped"]:
            return summary(tuning_id, state, state.get("stop_reason", "Stopped."))
        if any(t["state"] == "pushing" for t in state["trials"]):
            stop(tuning_id, state, "A push was interrupted and may have started a run, so nothing more was pushed.")
            return summary(tuning_id, state, state["stop_reason"])
        try:
            running = next((t for t in state["trials"] if t["state"] == "running"), None)
            if running:
                if not record_outcome(state, running, run_status(state["ref"])):
                    return summary(tuning_id, state)
                save_state(tuning_id, state)
                if state["stopped"]:
                    return summary(tuning_id, state, state["stop_reason"])
            pending = next((t for t in state["trials"] if t["state"] == "pending"), None)
            if pending:
                launch(tuning_id, state, pending)
        except (KaggleError, OSError, ValueError) as error:
            return summary(tuning_id, state, f"Error, will retry on the next call unless stopped: {error}")
        return summary(tuning_id, state)


@mcp.tool()
def compare_trials(tuning_id: str) -> list[dict]:
    """Rank finished trials by validation macro F1; unfinished and failed trials come last."""
    trials = load_state(tuning_id)["trials"]
    return sorted(trials, key=lambda t: -t["metrics"]["val_macro_f1"] if t["state"] == "done" else float("inf"))


if __name__ == "__main__":
    mcp.run()
