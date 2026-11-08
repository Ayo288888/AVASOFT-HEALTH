"""
MCP server that reads, runs and debugs the owner's Kaggle notebooks and runs bounded tuning.

Wraps the Kaggle CLI, which handles authentication. Pushes are only allowed to the notebook
named by KAGGLE_MCP_NOTEBOOK, because a push replaces a notebook's code.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / ".kaggle_runs"
CLI_TIMEOUT = 300


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
