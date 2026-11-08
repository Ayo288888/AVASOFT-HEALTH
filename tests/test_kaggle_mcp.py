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


if __name__ == "__main__":
    unittest.main()
