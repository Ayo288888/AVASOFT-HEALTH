import json
import unittest
from pathlib import Path

import pandas as pd

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "afrocare_stage1_v2.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def sources():
    return ["".join(c["source"]) for c in cells()["cells"] if c["cell_type"] == "code"]


def score_function():
    namespace = {}
    exec(next(s for s in sources() if "def score(" in s), namespace)
    return namespace["score"]


class TestNotebook(unittest.TestCase):
    def test_every_code_cell_compiles(self):
        for source in sources():
            compile(source, "cell", "exec")

    def test_private_dataset_is_attached(self):
        self.assertEqual(cells()["metadata"]["kaggle_mcp"]["dataset_sources"], ["wisdomilori/afrocare-dx-v2"])

    def test_defaults_come_before_the_tagged_parameters_cell(self):
        code = [c for c in cells()["cells"] if c["cell_type"] == "code"]
        tagged = [i for i, c in enumerate(code) if "parameters" in c["metadata"].get("tags", [])]
        self.assertEqual(len(tagged), 1)
        self.assertIn("TRIAL_ID", "".join(code[tagged[0]]["source"]))
        self.assertTrue(any("LEARNING_RATE =" in "".join(c["source"]) for c in code[:tagged[0]]))

    def test_test_split_is_only_used_when_final_eval_is_on(self):
        text = "\n".join(sources())
        self.assertIn("FINAL_EVAL = False", text)
        self.assertIn("if FINAL_EVAL:", text)
        self.assertEqual(text.count('"test"'), 1)

    def test_model_is_forced_to_32_bit_for_mixed_precision(self):
        self.assertIn(".float().to(device)", "\n".join(sources()))

    def test_metrics_file_carries_trial_id_and_score(self):
        text = "\n".join(sources())
        self.assertIn('"trial_id": TRIAL_ID', text)
        self.assertIn('"metrics.json"', text)


class TestScore(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "label": ["Malaria", "Malaria", "Cholera", "Tetanus"],
            "source": ["kaggle773", "ddxplus", "kaggle773", "synthetic"],
        })
        self.train_counts = {"Malaria": 5000, "Cholera": 10, "Tetanus": 8}

    def test_generated_cases_never_count(self):
        result = score_function()(self.df, ["Malaria", "Malaria", "Cholera", "WRONG"], self.train_counts)
        self.assertEqual((result["rows"], result["accuracy"]), (3, 1.0))
        self.assertEqual(result["macro_f1"], 1.0)

    def test_thin_conditions_are_reported_separately(self):
        result = score_function()(self.df, ["Malaria", "Malaria", "Malaria", "Tetanus"], self.train_counts)
        self.assertEqual(result["thin_macro_f1"], 0.0)

    def test_scores_are_reported_by_source(self):
        result = score_function()(self.df, ["Malaria", "Cholera", "Cholera", "Tetanus"], self.train_counts)
        self.assertEqual(result["by_source"], {"ddxplus": 0.0, "kaggle773": 1.0})


if __name__ == "__main__":
    unittest.main()
