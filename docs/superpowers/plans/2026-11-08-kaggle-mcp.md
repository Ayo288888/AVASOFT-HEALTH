# Kaggle MCP Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A project MCP server that lets Claude read, run and debug the owner's Kaggle notebooks and run bounded hyperparameter tuning.

**Architecture:** One stdio server, `tools/kaggle_mcp/server.py`, calling `python -m kaggle` through `subprocess`. Plain functions hold the logic and are tested with the CLI mocked; thin `@mcp.tool()` functions expose them. Tuning state is JSON under `.kaggle_runs/`.

**Tech Stack:** Python, `mcp` (FastMCP), Kaggle CLI 2.2.5, `unittest` with `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-11-08-kaggle-mcp-design.md`

## Global Constraints

- Dedicated notebook: `KAGGLE_MCP_NOTEBOOK` = `wisdomilori/afrocare-stage1-v2`. Every push to another ref is refused.
- Generated `kernel-metadata.json` always has `"is_private": true`.
- `subprocess.run` with an argument list, never `shell=True`.
- Never read or return token files. Scrub `KGAT` tokens (`KGAT_?[A-Za-z0-9_-]+` → `[REDACTED]`) from all returned CLI text.
- `max_trials` default 8, hard limit 12. A grid larger than `max_trials` is rejected, not sampled.
- Two consecutive failed trials stop the tuning.
- Tuning ranks on `val_macro_f1` only.
- No tool deletes a notebook. No MCP call waits for a run to finish.
- `mcp` goes in `tools/kaggle_mcp/requirements.txt` only, not the app's requirements.
- Test command: `python -m unittest tests.test_kaggle_mcp -v`.

## Review Focus

1. A notebook with no `parameters` cell passed to `start_tuning`: refused before anything is pushed (Task 3).
2. `tuning_status` called right after a push, when Kaggle still reports the previous version as complete: must not record the old result. The parameters cell carries `TRIAL_ID`, `metrics.json` must echo it, and a mismatch leaves the trial pending (Task 3).
3. `tuning_status` called repeatedly while a trial runs: no second push (Task 3).
4. `check_errors` on a run with empty or non-JSON logs: returns "no error found", no crash (Task 2).
5. Unknown `tuning_id` or corrupt state file: plain error, no traceback, no push (Task 3).

---

### Task 1: CLI wrapper and safeguards

**Files:**
- Create: `tools/kaggle_mcp/server.py`, `tools/kaggle_mcp/requirements.txt`
- Modify: `.gitignore` (add `.kaggle_runs/`)
- Test: `tests/test_kaggle_mcp.py`

**Interfaces:**
- Produces: `class KaggleError(Exception)`; `scrub(text: str) -> str`; `validate_ref(ref: str) -> str` (accepts only `^[\w-]+/[\w-]+$`, else `KaggleError`); `require_dedicated(ref: str) -> None`; `run_cli(args: list[str]) -> str` (raises `KaggleError` with scrubbed text on non-zero exit); `write_metadata(folder: Path, ref: str, code_file: str, dataset_sources: list[str]) -> Path`.

- [ ] **Step 1:** `pip install -r tools/kaggle_mcp/requirements.txt`. Expected: `python -c "import mcp"` exits 0.
- [ ] **Step 2: Write failing tests**
  - `test_push_to_other_notebook_is_refused`: `require_dedicated("wisdomilori/healthcare-model")` raises `KaggleError`.
  - `test_dedicated_notebook_is_allowed`.
  - `test_metadata_is_always_private`: loaded JSON has `is_private is True`, `kernel_type == "notebook"`, `enable_gpu is True`.
  - `test_token_is_scrubbed`: `scrub("x KGAT_abc123 y") == "x [REDACTED] y"`.
  - `test_bad_ref_is_rejected`: `"a/b; rm -rf"` and `"--flag"` raise.
  - `test_cli_error_is_scrubbed`: mocked exit 1 with stderr containing a token raises `KaggleError` without the token.
  - `test_cli_uses_no_shell`: mocked `subprocess.run` has no `shell=True` and its first args are `[sys.executable, "-m", "kaggle"]`.
- [ ] **Step 3:** Run tests. Expected: FAIL on import.
- [ ] **Step 4:** Implement the Interfaces above. Read `KAGGLE_MCP_NOTEBOOK` from the environment at call time; if unset, every push is refused.
- [ ] **Step 5:** Run tests. Expected: PASS.
- [ ] **Step 6:** Commit `feat: add Kaggle CLI wrapper with push safeguards`.

### Task 2: Read, run and error tools

**Files:** Modify `tools/kaggle_mcp/server.py`; Test `tests/test_kaggle_mcp.py`

**Interfaces:**
- Consumes: Task 1.
- Produces: `read_cells(path: Path) -> list[dict]` (`{"number", "type", "source"}`, numbered from 1); `find_error(log_text: str, cells: list[dict]) -> dict | None` (`{"type", "message", "traceback", "cell"}`, `cell` may be `None`); `push(path: Path, ref: str) -> str`; tools `list_notebooks()`, `read_notebook(ref)`, `run_notebook(path)`, `run_status(ref)`, `check_errors(ref)`, `get_results(ref)`.

- [ ] **Step 1: Write failing tests**
  - `test_find_error_names_the_failing_cell`: log with `ValueError: bad` and a traceback line `x = load_data()`; cells with that line in cell 3 → `type == "ValueError"`, `cell == 3`.
  - `test_find_error_without_matching_cell_keeps_traceback`: `cell is None`, `traceback` non-empty.
  - `test_find_error_handles_empty_and_non_json_logs`: `""` and `"not json"` → `None`.
  - `test_run_notebook_refuses_before_calling_cli`: `KAGGLE_MCP_NOTEBOOK` differs; mocked CLI not called.
  - `test_push_stages_private_metadata_with_the_notebook`: at CLI call time the push folder has the notebook and private metadata.
  - `test_get_results_reads_metrics_json`: mocked output download writes `metrics.json`; tool returns its contents and the file list.
- [ ] **Step 2:** Run tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Kaggle logs are a JSON array of `{"data": ...}`; if parsing fails, treat the whole text as the log. Match the last `Name: message` exception line, and the traceback's source lines against cell source. Downloads go to `.kaggle_runs/<slug>/`. Commands: `kernels list --mine --csv`, `kernels pull`, `kernels push -p`, `kernels status`, `kernels logs`, `kernels output -p`. If a Task 5 live check shows different flags or log shapes on CLI 2.2.5, correct them here.
- [ ] **Step 4:** Run tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: add Kaggle notebook read, run and error tools`.

### Task 3: Tuning

**Files:** Modify `tools/kaggle_mcp/server.py`; Test `tests/test_kaggle_mcp.py`

**Interfaces:**
- Consumes: `push`, `run_cli`, `require_dedicated`, `KaggleError`.
- Produces: `expand_grid(grid: dict[str, list], max_trials: int) -> list[dict]`; `set_parameters(notebook: dict, params: dict) -> dict`; tools `start_tuning(path, grid, max_trials=8)`, `tuning_status(tuning_id)`, `compare_trials(tuning_id)`. State file `.kaggle_runs/tuning/<tuning_id>.json` with `{"notebook", "trials": [{"id", "params", "state", "metrics"}], "stopped"}`; `state` is one of `pending`, `running`, `done`, `failed`.

- [ ] **Step 1: Write failing tests**
  - `test_grid_over_limit_is_rejected`: 3×3 grid with `max_trials=8` raises; `max_trials=13` raises.
  - `test_set_parameters_changes_only_tagged_cell`: other cells identical; tagged source contains the params and `TRIAL_ID`.
  - `test_start_tuning_refuses_notebook_without_parameters_cell`: raises, push not called.
  - `test_status_records_result_and_starts_next_trial`.
  - `test_status_while_running_does_not_push_again`: two calls, push count unchanged.
  - `test_stale_result_is_ignored`: Kaggle says complete, `metrics.json` has another `trial_id`; trial stays `running`, no next push.
  - `test_completed_run_without_metrics_is_failed`.
  - `test_one_failure_continues`, `test_two_failures_in_a_row_stop`.
  - `test_kaggle_outage_leaves_trial_pending`: `run_cli` raises during status; state unchanged, retried next call.
  - `test_resumes_from_saved_state`: a fresh call using only the file continues the tuning.
  - `test_unknown_or_corrupt_tuning_id_is_plain_error`: `KaggleError`, no push.
  - `test_compare_ranks_by_val_macro_f1`: descending; failed trials listed last.
- [ ] **Step 2:** Run tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Validate `tuning_id` against `^[\w-]+$`. Write the state file through a temp file and `os.replace`. The tagged cell is the first code cell with `"parameters"` in `metadata.tags`. Each `tuning_status` call handles at most one state change.
- [ ] **Step 4:** Run tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: add bounded Kaggle hyperparameter tuning`.

### Task 4: Register the server

**Files:** Modify `tools/kaggle_mcp/server.py` (`mcp.run()` under `__main__`); Create `.mcp.json`

- [ ] **Step 1: Write failing test** `test_all_nine_tools_are_registered`: the FastMCP tool list equals the spec's nine names.
- [ ] **Step 2:** Run. Expected: FAIL if any is missing.
- [ ] **Step 3:** Add `.mcp.json` with server `kaggle`, command `python`, args `["tools/kaggle_mcp/server.py"]`, env `KAGGLE_MCP_NOTEBOOK=wisdomilori/afrocare-stage1-v2`.
- [ ] **Step 4:** Run the full suite: `python -m unittest discover -s tests -t .`. Expected: OK.
- [ ] **Step 5:** Commit `feat: register Kaggle MCP server for this project`.

### Task 5: Live check

No new files beyond fixes. Uses the owner's Kaggle account; the notebook stays private.

- [ ] **Step 1:** Ask the owner to restart the session so the tools load.
- [ ] **Step 2:** Push a one-cell notebook that writes `metrics.json` to the dedicated slug. Follow `run_status` to complete, then `get_results`. Expected: metrics returned.
- [ ] **Step 3:** Push a notebook whose second cell raises `ValueError`. Expected: `check_errors` returns `ValueError` and cell 2.
- [ ] **Step 4:** Confirm on `list_notebooks` that no other notebook's last-run time changed.
- [ ] **Step 5:** Fix any CLI differences found, with a test for each, then commit `fix: match Kaggle CLI 2.2.5 behaviour`.
