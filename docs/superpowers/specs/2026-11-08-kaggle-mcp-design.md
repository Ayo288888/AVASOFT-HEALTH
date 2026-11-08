# Kaggle MCP tool: design

Date: 2026-11-08. Status: awaiting review.

## Goal

Let Claude Code work with the owner's Kaggle notebooks from this project: read them, run them, find errors, and run bounded hyperparameter tuning for the AfroCare Stage 1 v2 model.

Success means Claude can push the training notebook, see which cell failed and why, fix it, rerun it, and return a ranked table of tuning trials, without any risk to the owner's existing notebooks.

## Constraints

- Kaggle's API has no way to run one cell. A run is a push of the whole notebook, executed top to bottom. The available operations are `list`, `pull`, `push`, `status`, `logs` and `output` (`python -m kaggle kernels`, CLI 2.2.5).
- A push replaces the notebook's code with a new version.
- Each run spends the owner's Kaggle GPU quota.
- Kaggle runs last hours, so no MCP call may wait for one to finish.
- The token lives in `~/.kaggle/access_token` and is read by the CLI only.

## Approach

A small Python stdio MCP server that wraps the Kaggle CLI through `subprocess` with argument lists (never a shell). The CLI already handles authentication and is verified working on this machine, so the server adds no HTTP client code.

## Files

| Path | Purpose |
|---|---|
| `tools/kaggle_mcp/server.py` | MCP server and all tools |
| `tools/kaggle_mcp/requirements.txt` | `mcp` only, kept out of the deployed app's requirements |
| `tests/test_kaggle_mcp.py` | Unit tests with the CLI mocked |
| `.mcp.json` | Registers the server for this project |
| `.gitignore` | Adds `.kaggle_runs/` |

Run state and downloaded outputs live in `.kaggle_runs/<slug>/`.

## Tools

| Tool | Input | Returns | Writes to Kaggle |
|---|---|---|---|
| `list_notebooks` | none | The owner's notebooks: ref, title, last run | No |
| `read_notebook` | `ref` | Numbered cells with type and source | No |
| `run_notebook` | `path` to a local `.ipynb` | Confirmation that a new version started | Yes, dedicated notebook only |
| `run_status` | `ref` | queued, running, complete, error or cancelled | No |
| `check_errors` | `ref` | Exception type, message, traceback, and the matching cell number when one can be found | No |
| `get_results` | `ref` | Downloaded file list and contents of `metrics.json` | No |
| `start_tuning` | `path`, `grid`, `max_trials` | Tuning id; starts trial 1 | Yes, dedicated notebook only |
| `tuning_status` | `tuning_id` | Progress; records a finished trial and starts the next | Yes, dedicated notebook only |
| `compare_trials` | `tuning_id` | Trials ranked by validation macro F1 | No |

Read tools accept any of the owner's notebooks. `check_errors` reads the run logs and matches the traceback's source lines against the notebook's cells. If no cell matches, it returns the traceback with no cell number.

## Tuning

- The notebook has one code cell tagged `parameters`. For each trial, the server rewrites only that cell.
- The notebook must write `metrics.json` containing at least `val_macro_f1`. A completed trial with no such file is recorded as failed.
- Trials run one at a time. Each is a new version of the dedicated notebook.
- No background process runs. `tuning_status` moves the tuning forward each time it is called, and state is saved to `.kaggle_runs/` so a restarted session can continue.
- If a trial errors, it is recorded as failed and the next trial starts. Two failures in a row stop the tuning, because the fault is probably in the notebook rather than the parameters.
- `max_trials` defaults to 8 and may not exceed 12. A grid larger than `max_trials` is rejected, not sampled.
- Ranking uses validation data only. The test split is not used for tuning.

## Safeguards

- Pushes are refused unless the slug equals `KAGGLE_MCP_NOTEBOOK`, set in `.mcp.json` to `wisdomilori/afrocare-stage1-v2`.
- The server writes `kernel-metadata.json` itself with `is_private: true`, so a run cannot publish the notebook.
- Tools never read or return token files. CLI output is scrubbed of `KGAT` tokens before it is returned.
- `ref` values and grid names are validated before they reach the CLI.
- No tool deletes a notebook.

## Error handling

- CLI not found, auth failure, or Kaggle rate limits return a plain error naming the cause. Nothing is retried automatically except inside `tuning_status`, which leaves the trial pending and tries again on its next call.
- A non-zero CLI exit is reported with its scrubbed message, never as success.

## Testing

Unit tests, with `subprocess` mocked:

- a push to any other slug is refused
- generated metadata is always private
- traceback matching finds the right cell, and returns none when nothing matches
- rewriting the parameters cell leaves all other cells untouched
- trial sequencing: success, a single failure, two failures stopping, and resuming from saved state
- oversized grids are rejected
- tokens are scrubbed from returned text

One live check after the unit tests: push a one-cell notebook to the dedicated slug and go through `run_status`, `check_errors` and `get_results`, then repeat with a cell that raises an exception.

## Out of scope

- Running a single cell, which Kaggle's API does not offer.
- Uploading AfroCare-Dx v2. That needs its own explicit approval from the owner.
- Writing the DeBERTa training notebook. That follows once this tool works.
- Parallel trials, and search methods other than a fixed grid.
