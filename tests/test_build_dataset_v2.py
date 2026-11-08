import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_dataset_v2 import cap_per_label, drop_duplicates_and_conflicts, split


def frame(rows):
    df = pd.DataFrame(rows, columns=["id", "text", "label", "source"])
    return df.assign(original_label=df["label"])


class TestDuplicates(unittest.TestCase):
    def test_keeps_one_copy_and_drops_conflicting_labels(self):
        df = frame([
            ("a", "I have fever and chills", "Malaria (uncomplicated)", "ddxplus"),
            ("b", "i have  FEVER and chills", "Malaria (uncomplicated)", "kaggle773"),  # same text, same label
            ("c", "My stomach is running", "Cholera", "zenodo_infectious"),
            ("d", "my stomach is running", "Acute gastroenteritis / watery diarrhoea", "ddxplus"),  # conflict
        ])
        out = drop_duplicates_and_conflicts(df)
        self.assertEqual(list(out["id"]), ["a"])

    def test_same_words_in_another_order_or_age_is_a_duplicate(self):
        df = frame([
            ("a", "Patient is a 30-year-old male presenting with: cough; fever.", "Pneumonia", "ddxplus"),
            ("b", "Patient is a 54-year-old male presenting with: fever; cough.", "Pneumonia", "ddxplus"),
        ])
        self.assertEqual(len(drop_duplicates_and_conflicts(df)), 1)


class TestCap(unittest.TestCase):
    def test_caps_and_prefers_real_patient_sources(self):
        df = frame([(f"d{i}", f"ddx {i}", "Pneumonia", "ddxplus") for i in range(5)]
                   + [("s0", "s2d text", "Pneumonia", "symptom2disease")])
        out = cap_per_label(df, {"Pneumonia": 2}, seed=1)
        self.assertEqual(len(out), 2)
        self.assertIn("s0", set(out["id"]))  # natural text is kept before templated rows
        self.assertEqual(list(out.columns), ["id", "text", "label", "source", "original_label"])  # split() needs label

    def test_no_single_disease_dominates_other(self):
        df = frame([(f"r{i}", f"rash {i}", "OTHER", "kaggle773") for i in range(10)]
                   + [(f"p{i}", f"pain {i}", "OTHER", "kaggle773") for i in range(3)])
        df["original_label"] = ["Roseola"] * 10 + ["acute pancreatitis"] * 3
        out = cap_per_label(df, {"OTHER": 100}, seed=1, cap_per_other_disease=4)
        self.assertEqual(out["original_label"].value_counts().to_dict(), {"Roseola": 4, "acute pancreatitis": 3})


class TestSplit(unittest.TestCase):
    def test_every_label_appears_in_every_split(self):
        rows = [(f"{lab}{i}", f"{lab} text {i}", lab, "x") for lab in ("A", "B") for i in range(12)]
        out = split(frame(rows), seed=1)
        self.assertEqual(set(out["split"]), {"train", "val", "test"})
        for _, g in out.groupby("split"):
            self.assertEqual(set(g["label"]), {"A", "B"})
        self.assertEqual(len(out), 24)


if __name__ == "__main__":
    unittest.main()
