import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.kaggle_mcp import server

DEDICATED = "wisdomilori/afrocare-stage1-v2"


def completed(stdout="", stderr="", code=0):
    return mock.Mock(returncode=code, stdout=stdout, stderr=stderr)


@mock.patch.dict(os.environ, {"KAGGLE_MCP_NOTEBOOK": DEDICATED})
class TestSafeguards(unittest.TestCase):
    def test_push_to_other_notebook_is_refused(self):
        with self.assertRaises(server.KaggleError):
            server.require_dedicated("wisdomilori/healthcare-model")

    def test_dedicated_notebook_is_allowed(self):
        server.require_dedicated(DEDICATED)

    def test_metadata_is_always_private(self):
        with tempfile.TemporaryDirectory() as d:
            meta = json.loads(server.write_metadata(Path(d), DEDICATED, "nb.ipynb", []).read_text())
        self.assertIs(meta["is_private"], True)
        self.assertEqual(meta["kernel_type"], "notebook")
        self.assertIs(meta["enable_gpu"], True)

    def test_token_is_scrubbed(self):
        self.assertEqual(server.scrub("x KGAT_abc123 y"), "x [REDACTED] y")

    def test_bad_ref_is_rejected(self):
        for ref in ("a/b; rm -rf", "--flag"):
            with self.assertRaises(server.KaggleError):
                server.validate_ref(ref)

    @mock.patch("subprocess.run", return_value=completed(stderr="denied KGAT_secret9", code=1))
    def test_cli_error_is_scrubbed(self, run):
        with self.assertRaises(server.KaggleError) as e:
            server.run_cli(["kernels", "list"])
        self.assertNotIn("KGAT_secret9", str(e.exception))

    @mock.patch("subprocess.run", return_value=completed(stdout="ok"))
    def test_cli_uses_no_shell(self, run):
        self.assertEqual(server.run_cli(["kernels", "list"]), "ok")
        self.assertEqual(run.call_args.args[0][:3], [sys.executable, "-m", "kaggle"])
        self.assertFalse(run.call_args.kwargs.get("shell", False))


def notebook(*cells):
    return {"cells": [{"cell_type": t, "source": src, "metadata": dict(meta), "outputs": [], "execution_count": None}
                      for t, src, meta in cells],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def code(src, **meta):
    return ("code", src, meta)


class TestFindError(unittest.TestCase):
    cells = [{"number": 1, "type": "markdown", "source": "x = load_data()"},
             {"number": 2, "type": "code", "source": "import pandas"},
             {"number": 3, "type": "code", "source": "a = 1\nx = load_data()"}]

    def test_find_error_names_the_failing_cell(self):
        log = json.dumps([{"data": "Traceback (most recent call last):\n----> 2 x = load_data()\n"},
                          {"data": "ValueError: bad\n"}])
        err = server.find_error(log, self.cells)
        self.assertEqual((err["type"], err["message"], err["cell"]), ("ValueError", "bad", 3))

    def test_find_error_without_matching_cell_keeps_traceback(self):
        err = server.find_error("Traceback (most recent call last):\n  y = other()\nKeyError: 'z'\n", self.cells)
        self.assertIsNone(err["cell"])
        self.assertIn("y = other()", err["traceback"])

    def test_find_error_handles_empty_and_non_json_logs(self):
        self.assertIsNone(server.find_error("", self.cells))
        self.assertIsNone(server.find_error("not json", self.cells))

    def test_arrow_line_beats_context_lines_shared_with_other_cells(self):
        cells = [{"number": 1, "type": "code", "source": "model.train()"},
                 {"number": 2, "type": "code", "source": "model.train()\nboom()"}]
        log = "Traceback (most recent call last):\n      1 model.train()\n----> 2 boom()\nValueError: bad\n"
        self.assertEqual(server.find_error(log, cells)["cell"], 2)

    def test_warnings_are_not_errors(self):
        self.assertIsNone(server.find_error("FutureWarning: old api\n", self.cells))

    def test_colour_codes_do_not_hide_the_error(self):
        err = server.find_error("\x1b[31mValueError\x1b[0m: bad\n", self.cells)
        self.assertEqual(err["type"], "ValueError")


class TestNotebookTools(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.nb = self.dir / "train.ipynb"
        self.nb.write_text(json.dumps(notebook(code("print(1)"))), encoding="utf-8")
        for patcher in (mock.patch.object(server, "RUNS", self.dir / "runs"),
                        mock.patch.dict(os.environ, {"KAGGLE_MCP_NOTEBOOK": DEDICATED})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_read_cells_numbers_from_one_and_joins_source(self):
        self.nb.write_text(json.dumps(notebook(code(["a\n", "b"]))), encoding="utf-8")
        self.assertEqual(server.read_cells(self.nb), [{"number": 1, "type": "code", "source": "a\nb"}])

    def test_tokens_in_notebook_source_are_scrubbed(self):
        self.nb.write_text(json.dumps(notebook(code("key = 'KGAT_secret9'"))), encoding="utf-8")
        self.assertNotIn("KGAT_secret9", server.read_cells(self.nb)[0]["source"])

    def test_tokens_in_metrics_are_scrubbed(self):
        def cli(args):
            folder = Path(args[args.index("-p") + 1])
            (folder / "metrics.json").write_text('{"note": "KGAT_secret9"}')
            return ""

        with mock.patch.object(server, "run_cli", side_effect=cli):
            self.assertEqual(server.get_results(DEDICATED)["metrics"], {"note": "[REDACTED]"})

    @mock.patch.object(server, "run_cli", return_value="")
    def test_download_can_be_limited_to_one_file(self, cli):
        server.download_results(DEDICATED, "metrics")
        args = cli.call_args.args[0]
        self.assertEqual(args[args.index("--file-pattern") + 1], "metrics")

    @mock.patch.object(server, "run_cli")
    def test_run_notebook_refuses_before_calling_cli(self, cli):
        with mock.patch.dict(os.environ, clear=True):
            with self.assertRaises(server.KaggleError):
                server.run_notebook(str(self.nb))
        cli.assert_not_called()

    @mock.patch.object(server, "run_cli")
    def test_push_to_another_notebook_never_reaches_cli(self, cli):
        with self.assertRaises(server.KaggleError):
            server.push(self.nb, "wisdomilori/healthcare-model")
        cli.assert_not_called()

    def test_push_stages_private_metadata_with_the_notebook(self):
        seen = {}

        def cli(args):
            folder = Path(args[args.index("-p") + 1])
            seen["meta"] = json.loads((folder / "kernel-metadata.json").read_text())
            seen["has_nb"] = (folder / seen["meta"]["code_file"]).exists()
            return "Kernel version 3 successfully pushed."

        with mock.patch.object(server, "run_cli", side_effect=cli):
            server.push(self.nb, DEDICATED)
        self.assertIs(seen["meta"]["is_private"], True)
        self.assertTrue(seen["has_nb"])

    @mock.patch.object(server, "run_cli", return_value="Kernel push error: bad slug")
    def test_push_failure_reported_in_stdout_is_an_error(self, cli):
        with self.assertRaises(server.KaggleError):
            server.push(self.nb, DEDICATED)

    def fake_output(self, args):
        folder = Path(args[args.index("-p") + 1])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metrics.json").write_text('{"val_macro_f1": 0.5}')
        return ""

    def test_get_results_reads_metrics_json(self):
        with mock.patch.object(server, "run_cli", side_effect=self.fake_output):
            result = server.get_results(DEDICATED)
        self.assertEqual(result["metrics"], {"val_macro_f1": 0.5})
        self.assertIn("metrics.json", result["files"])

    def test_old_metrics_are_not_returned_when_a_run_has_none(self):
        with mock.patch.object(server, "run_cli", side_effect=self.fake_output):
            server.get_results(DEDICATED)
        with mock.patch.object(server, "run_cli", return_value=""):
            self.assertIsNone(server.get_results(DEDICATED)["metrics"])

    @mock.patch.object(server, "run_cli", return_value='x has status "KernelWorkerStatus.COMPLETE"')
    def test_status_is_normalised(self, cli):
        self.assertEqual(server.run_status(DEDICATED), "complete")


class TestRegistration(unittest.TestCase):
    def test_all_nine_tools_are_registered(self):
        names = {t.name for t in asyncio.run(server.mcp.list_tools())}
        self.assertEqual(names, {"list_notebooks", "read_notebook", "run_notebook", "run_status", "check_errors",
                                 "get_results", "start_tuning", "tuning_status", "compare_trials"})


class TestGrid(unittest.TestCase):
    def test_grid_over_limit_is_rejected(self):
        grid = {"lr": [1, 2, 3], "bs": [1, 2, 3]}
        with self.assertRaises(server.KaggleError):
            server.expand_grid(grid, 8)
        with self.assertRaises(server.KaggleError):
            server.expand_grid({"lr": [1]}, 13)

    def test_grid_expands_every_combination(self):
        self.assertEqual(server.expand_grid({"a": [1, 2], "b": [3]}, 8), [{"a": 1, "b": 3}, {"a": 2, "b": 3}])

    def test_parameter_names_must_be_identifiers(self):
        with self.assertRaises(server.KaggleError):
            server.expand_grid({"a=1;import os": [1]}, 8)

    def test_set_parameters_changes_only_tagged_cell(self):
        nb = notebook(code("first"), code("old", tags=["parameters"]), code("last"))
        out = server.set_parameters(nb, {"TRIAL_ID": "t-1", "lr": 0.01})
        self.assertEqual([out["cells"][0]["source"], out["cells"][2]["source"]], ["first", "last"])
        self.assertEqual(server.source_text(out["cells"][1]), "TRIAL_ID = 't-1'\nlr = 0.01\n")
        self.assertEqual(nb["cells"][1]["source"], "old")


@mock.patch.dict(os.environ, {"KAGGLE_MCP_NOTEBOOK": DEDICATED})
@mock.patch.object(server, "STALE_SECONDS", 0)
class TestTuning(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.nb = self.dir / "train.ipynb"
        self.nb.write_text(json.dumps(notebook(code("old", tags=["parameters"]), code("train()"))), encoding="utf-8")
        for name, kwargs in (("push", {"return_value": "ok"}), ("run_status", {"return_value": "running"}),
                             ("download_results", {})):
            patcher = mock.patch.object(server, name, **kwargs)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(server, "RUNS", self.dir / "runs")
        patcher.start()
        self.addCleanup(patcher.stop)

    def start(self, values=(1, 2, 3)):
        return server.start_tuning(str(self.nb), {"lr": list(values)})["tuning_id"]

    def state(self, tid):
        return json.loads((self.dir / "runs" / "tuning" / f"{tid}.json").read_text(encoding="utf-8"))

    def finish(self, tid, n, f1=0.5):
        self.run_status.return_value = "complete"
        self.download_results.return_value = {"metrics": {"trial_id": f"{tid}-{n}", "val_macro_f1": f1}}
        return server.tuning_status(tid)

    def fail(self):
        self.run_status.return_value = "error"

    def test_start_tuning_refuses_notebook_without_parameters_cell(self):
        self.nb.write_text(json.dumps(notebook(code("x"))), encoding="utf-8")
        with self.assertRaises(server.KaggleError):
            self.start()
        self.push.assert_not_called()

    def test_each_trial_is_pushed_with_its_own_id_and_values(self):
        seen = []

        def capture(path, ref):
            seen.append(server.read_cells(path)[0]["source"])
            return "ok"

        self.push.side_effect = capture
        tid = self.start()
        self.assertEqual(seen, [f"TRIAL_ID = '{tid}-1'\nlr = 1\n"])

    def test_status_records_result_and_starts_next_trial(self):
        tid = self.start()
        self.finish(tid, 1, 0.7)
        state = self.state(tid)
        self.assertEqual([t["state"] for t in state["trials"]], ["done", "running", "pending"])
        self.assertEqual(state["trials"][0]["metrics"]["val_macro_f1"], 0.7)
        self.assertEqual(self.push.call_count, 2)

    def test_status_while_running_does_not_push_again(self):
        tid = self.start()
        server.tuning_status(tid)
        server.tuning_status(tid)
        self.assertEqual(self.push.call_count, 1)

    def test_stale_result_is_ignored(self):
        tid = self.start()
        self.run_status.return_value = "complete"
        self.download_results.return_value = {"metrics": {"trial_id": "older-trial", "val_macro_f1": 0.9}}
        server.tuning_status(tid)
        self.assertEqual(self.state(tid)["trials"][0]["state"], "running")
        self.assertEqual(self.push.call_count, 1)

    def test_recent_error_status_may_belong_to_the_previous_run(self):
        tid = self.start()
        self.fail()
        with mock.patch.object(server, "STALE_SECONDS", 90):
            server.tuning_status(tid)
        self.assertEqual(self.state(tid)["trials"][0]["state"], "running")

    def test_completed_run_without_metrics_is_failed(self):
        tid = self.start()
        self.run_status.return_value = "complete"
        self.download_results.return_value = {"metrics": None}
        server.tuning_status(tid)
        self.assertEqual(self.state(tid)["trials"][0]["state"], "failed")

    def test_metrics_missing_the_score_are_failed(self):
        tid = self.start()
        self.run_status.return_value = "complete"
        self.download_results.return_value = {"metrics": {"trial_id": f"{tid}-1"}}
        server.tuning_status(tid)
        self.assertEqual(self.state(tid)["trials"][0]["state"], "failed")

    def test_one_failure_continues(self):
        tid = self.start()
        self.fail()
        server.tuning_status(tid)
        state = self.state(tid)
        self.assertFalse(state["stopped"])
        self.assertEqual(state["trials"][1]["state"], "running")

    def test_two_failures_in_a_row_stop(self):
        tid = self.start()
        self.fail()
        server.tuning_status(tid)
        server.tuning_status(tid)
        state = self.state(tid)
        self.assertTrue(state["stopped"])
        self.assertEqual(state["trials"][2]["state"], "pending")
        self.assertEqual(self.push.call_count, 2)
        server.tuning_status(tid)
        self.assertEqual(self.push.call_count, 2)

    def test_a_success_resets_the_failure_count(self):
        tid = self.start()
        self.fail()
        server.tuning_status(tid)
        self.finish(tid, 2)
        self.fail()
        server.tuning_status(tid)
        self.assertFalse(self.state(tid)["stopped"])

    def test_kaggle_outage_leaves_trial_pending(self):
        tid = self.start()
        before = self.state(tid)
        self.run_status.side_effect = server.KaggleError("503")
        result = server.tuning_status(tid)
        self.assertEqual(self.state(tid), before)
        self.assertIn("503", result["note"])

    def test_failed_push_is_retried_on_next_status(self):
        self.push.side_effect = server.KaggleError("503")
        tid = self.start()
        self.assertEqual(self.state(tid)["trials"][0]["state"], "pending")
        self.push.side_effect = None
        server.tuning_status(tid)
        self.assertEqual(self.state(tid)["trials"][0]["state"], "running")

    def test_resumes_from_saved_state(self):
        tid = self.start()
        self.nb.unlink()
        self.finish(tid, 1)
        self.assertEqual(self.push.call_count, 2)

    def test_unknown_or_corrupt_tuning_id_is_plain_error(self):
        tid = self.start()
        self.push.reset_mock()
        (self.dir / "runs" / "tuning" / f"{tid}.json").write_text("{broken", encoding="utf-8")
        for bad in (tid, "missing", "../../etc"):
            with self.assertRaises(server.KaggleError):
                server.tuning_status(bad)
        self.push.assert_not_called()

    def test_previous_run_without_metrics_is_not_this_trials_failure(self):
        tid = self.start()
        self.run_status.return_value = "complete"
        self.download_results.return_value = {"metrics": None}
        with mock.patch.object(server, "STALE_SECONDS", 90):
            server.tuning_status(tid)
        self.assertEqual(self.state(tid)["trials"][0]["state"], "running")
        self.assertEqual(self.push.call_count, 1)

    def test_second_tuning_is_refused_while_one_is_active(self):
        self.start()
        with self.assertRaises(server.KaggleError):
            self.start()
        self.assertEqual(self.push.call_count, 1)

    def test_run_notebook_is_refused_during_a_tuning(self):
        self.start()
        with self.assertRaises(server.KaggleError):
            server.run_notebook(str(self.nb))
        self.assertEqual(self.push.call_count, 1)

    def test_a_stopped_tuning_no_longer_blocks_pushes(self):
        tid = self.start()
        self.fail()
        server.tuning_status(tid)
        server.tuning_status(tid)
        server.run_notebook(str(self.nb))
        self.assertEqual(self.push.call_count, 3)

    def test_run_that_never_finishes_is_eventually_failed(self):
        tid = self.start()
        state = self.state(tid)
        state["trials"][0]["pushed_at"] -= server.RUN_TIMEOUT_SECONDS + 1
        server.save_state(tid, state)
        self.run_status.return_value = "complete"
        self.download_results.return_value = {"metrics": {"trial_id": "someone-else", "val_macro_f1": 0.9}}
        server.tuning_status(tid)
        self.assertEqual(self.state(tid)["trials"][0]["state"], "failed")

    def test_push_with_unknown_outcome_stops_without_pushing_again(self):
        self.push.side_effect = server.KaggleTimeout("timed out")
        tid = self.start()
        for _ in range(3):
            result = server.tuning_status(tid)
        self.assertEqual(self.push.call_count, 1)
        self.assertTrue(self.state(tid)["stopped"])
        self.assertIn("timed out", result["note"])

    def test_interrupted_push_is_not_repeated(self):
        tid = self.start()
        state = self.state(tid)
        state["trials"][0]["state"] = "pushing"
        server.save_state(tid, state)
        server.tuning_status(tid)
        self.assertEqual(self.push.call_count, 1)
        self.assertTrue(self.state(tid)["stopped"])

    def test_tuning_downloads_only_metrics(self):
        tid = self.start()
        self.finish(tid, 1)
        self.download_results.assert_called_with(DEDICATED, server.METRICS_PATTERN)

    def test_bad_metrics_values_fail_the_trial(self):
        for bad in (["not", "a", "dict"], {"val_macro_f1": None}, {"val_macro_f1": "high"},
                    {"val_macro_f1": float("nan")}, {"val_macro_f1": True}):
            with self.subTest(bad=bad):
                tid = server.start_tuning(str(self.nb), {"lr": [1]})["tuning_id"]
                if isinstance(bad, dict):
                    bad = {**bad, "trial_id": f"{tid}-1"}
                self.run_status.return_value = "complete"
                self.download_results.return_value = {"metrics": bad}
                server.tuning_status(tid)
                self.assertEqual(self.state(tid)["trials"][0]["state"], "failed")
                server.compare_trials(tid)

    def test_missing_saved_notebook_is_a_plain_note(self):
        tid = self.start()
        (self.dir / "runs" / "tuning" / f"{tid}.ipynb").unlink()
        result = self.finish(tid, 1)
        self.assertIn("error", result["note"].lower())
        self.assertEqual(self.push.call_count, 1)

    def test_state_without_stopped_flag_is_corrupt(self):
        tid = self.start()
        state = self.state(tid)
        del state["stopped"]
        server.save_state(tid, state)
        with self.assertRaises(server.KaggleError):
            server.tuning_status(tid)

    def test_compare_ranks_by_val_macro_f1(self):
        tid = self.start()
        self.finish(tid, 1, 0.4)
        self.finish(tid, 2, 0.8)
        self.fail()
        server.tuning_status(tid)
        ranked = server.compare_trials(tid)
        self.assertEqual([t["id"] for t in ranked], [f"{tid}-2", f"{tid}-1", f"{tid}-3"])


if __name__ == "__main__":
    unittest.main()
