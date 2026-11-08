import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.symptom2disease_adapter import process_symptom2disease_csv
from src.adapters.zenodo_infectious_adapter import process_zenodo_infectious_csv
from scripts.build_label_map import target_for


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


class TestSymptom2DiseaseAdapter(unittest.TestCase):
    def test_keeps_patient_text_and_label(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "s2d.csv", Path(d) / "s2d.jsonl"
            src.write_text(
                ",label,text\n"
                '0,Malaria," I have high fever, chills and sweating at night. "\n'
                "1,Typhoid,\n",  # empty text is skipped
                encoding="utf-8",
            )
            self.assertEqual(process_symptom2disease_csv(str(src), str(out)), 1)
            rec = read_jsonl(out)[0]
            self.assertEqual(rec["text"], "I have high fever, chills and sweating at night.")
            self.assertEqual(rec["labels"], ["Malaria"])
            self.assertEqual(rec["source"], "symptom2disease")
            self.assertEqual(rec["id"], "S2D_00000")


class TestZenodoInfectiousAdapter(unittest.TestCase):
    def test_ragged_rows_and_repeated_symptoms(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "z.csv", Path(d) / "z.jsonl"
            src.write_text(
                "Disease,Symptom_1,Symptom_2,Symptom_3\n"
                "Shigellosis (Bacillary Dysentery),Bloody stools,Fever,fever,,\n"  # extra column, repeat
                "Measles,High fever,Rash\n"
                "Mumps,,,\n",  # no symptoms is skipped
                encoding="utf-8",
            )
            self.assertEqual(process_zenodo_infectious_csv(str(src), str(out)), 2)
            recs = read_jsonl(out)
            self.assertEqual(recs[0]["text"], "I have been experiencing bloody stools, fever.")
            self.assertEqual(recs[0]["labels"], ["Shigellosis (Bacillary Dysentery)"])
            self.assertEqual(recs[0]["source"], "zenodo_infectious")
            self.assertEqual(recs[1]["text"], "I have been experiencing high fever, rash.")


class TestNewLabelMappings(unittest.TestCase):
    def test_new_sources_map_to_conditions(self):
        self.assertEqual(target_for("symptom2disease", "Malaria"), "Malaria (uncomplicated)")
        self.assertEqual(target_for("symptom2disease", "Arthritis"), "OTHER")
        self.assertEqual(target_for("zenodo_infectious", "Shigellosis (Bacillary Dysentery)"),
                         "Dysentery (bloody diarrhoea)")
        self.assertEqual(target_for("zenodo_infectious", "Roseola"), "OTHER")


if __name__ == "__main__":
    unittest.main()
