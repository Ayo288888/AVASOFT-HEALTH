import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.afrimedqa_adapter import extract_afrimedqa_seeds
from scripts.build_label_map import target_for


def prompt(who, symptoms, condition):
    return f"Your {who} complains of {symptoms} and thinks {('he' if 'male' == who.split()[0] else 'she')} has {condition} and is going to visit the nearest doctor. Question: "


class TestAfrimedqaSeeds(unittest.TestCase):
    def test_extracts_unique_symptom_condition_pairs(self):
        rows = [
            ("consumer_queries", prompt("female friend", "Fever, chills, headache", "Malaria")),
            ("consumer_queries", prompt("male coworker", "fever, chills, headache", "malaria")),  # duplicate pair
            ("consumer_queries", prompt("female colleague", "seizures, headaches", "3. Epilepsy/Seizures")),
            ("consumer_queries", "Free-text question with no template"),
            ("mcq", prompt("female friend", "rash, fever", "Measles")),  # exam rows are ignored
        ]
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "afr.csv", Path(d) / "seeds.jsonl"
            pd.DataFrame(rows, columns=["question_type", "prompt"]).to_csv(src, index=False)
            self.assertEqual(extract_afrimedqa_seeds(str(src), str(out)), 2)
            recs = [json.loads(line) for line in open(out, encoding="utf-8")]

        self.assertEqual(recs[0]["text"], "I have fever, chills, headache.")
        self.assertEqual(recs[0]["labels"], ["malaria"])
        self.assertEqual(recs[0]["source"], "afrimedqa_seed")
        self.assertEqual(recs[1]["labels"], ["epilepsy/seizures"])  # leading list number stripped


class TestAfrimedqaSeedMapping(unittest.TestCase):
    def test_seed_labels_map_onto_conditions(self):
        self.assertEqual(target_for("afrimedqa_seed", "cerebral malaria"), "Severe malaria (cerebral or severe anaemia)")
        self.assertEqual(target_for("afrimedqa_seed", "kidney infection (pyelonephritis)"), "Urinary tract infection")
        self.assertEqual(target_for("afrimedqa_seed", "infections"), "DROP")
        self.assertEqual(target_for("afrimedqa_seed", "ebola"), "OTHER")


if __name__ == "__main__":
    unittest.main()
