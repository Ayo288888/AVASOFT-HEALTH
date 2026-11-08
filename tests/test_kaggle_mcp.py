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


if __name__ == "__main__":
    unittest.main()
