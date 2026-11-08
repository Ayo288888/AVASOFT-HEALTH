"""
MCP server that reads, runs and debugs the owner's Kaggle notebooks and runs bounded tuning.

Wraps the Kaggle CLI, which handles authentication. Pushes are only allowed to the notebook
named by KAGGLE_MCP_NOTEBOOK, because a push replaces a notebook's code.
"""
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / ".kaggle_runs"
CLI_TIMEOUT = 300

mcp = MCPServer("kaggle")


class KaggleError(Exception):
    pass


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
        raise KaggleError(f"Kaggle CLI timed out after {CLI_TIMEOUT}s: {args[:2]}")
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
    return [{"number": i, "type": c["cell_type"], "source": source_text(c)} for i, c in enumerate(notebook["cells"], 1)]


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
    # The first traceback line found in a cell is the user's call; later frames are usually library code.
    for line in lines:
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


@mcp.tool()
def get_results(ref: str) -> dict:
    """Download a notebook's output files and return their names and metrics.json."""
    folder = run_folder(validate_ref(ref), "output")
    run_cli(["kernels", "output", ref, "-p", str(folder), "-o"])
    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    metrics_file = next(folder.rglob("metrics.json"), None)
    try:
        metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file else None
    except ValueError:
        metrics = None
    return {"folder": str(folder), "files": files, "metrics": metrics}
