import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_baseline_v2 import evaluate


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "label": ["Malaria", "Malaria", "Cholera", "Tetanus"],
            "source": ["kaggle773", "ddxplus", "kaggle773", "synthetic"],
        })

    def test_generated_cases_never_count_toward_scores(self):
        result = evaluate(self.df, ["Malaria", "Malaria", "Cholera", "WRONG"])
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertNotIn("Tetanus", result["per_label"].index)

    def test_reports_accuracy_for_each_source(self):
        result = evaluate(self.df, ["Malaria", "Cholera", "Cholera", "Tetanus"])
        self.assertEqual(result["by_source"].to_dict(), {"ddxplus": 0.0, "kaggle773": 1.0})

    def test_macro_f1_covers_only_labels_that_can_be_scored(self):
        result = evaluate(self.df, ["Malaria", "Malaria", "Tetanus", "Tetanus"])
        self.assertAlmostEqual(result["macro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
