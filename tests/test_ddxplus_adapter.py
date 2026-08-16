import unittest
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.base_adapter import compute_md5_hash, validate_canonical_record
from src.adapters.ddxplus_adapter import process_ddxplus_csv

class TestDDXPlusAdapter(unittest.TestCase):
    def test_compute_md5_hash(self):
        h1 = compute_md5_hash("High fever and headache")
        h2 = compute_md5_hash("  high  fever  and   headache ")
        self.assertEqual(h1, h2)

    def test_validate_canonical_record(self):
        valid = {
            "id": "DDX_001",
            "text": "High fever and cough",
            "labels": ["Pneumonia"],
            "language_variant": "Standard_Clinical",
            "severity_level": "Urgent",
            "icd10_codes": ["J18.9"],
            "source": "ddxplus"
        }
        self.assertTrue(validate_canonical_record(valid))

    def test_ddxplus_csv_processing(self):
        tmp_csv = Path("tests/tmp_ddx.csv")
        tmp_jsonl = Path("tests/tmp_ddx.jsonl")
        
        df = pd.DataFrame({
            "AGE": [34],
            "SEX": ["M"],
            "PATHOLOGY": ["Pneumonia"],
            "EVIDENCES": ["['E_91']"]
        })
        df.to_csv(tmp_csv, index=False)
        
        count = process_ddxplus_csv(str(tmp_csv), str(tmp_jsonl))
        self.assertEqual(count, 1)
        
        if tmp_csv.exists(): tmp_csv.unlink()
        if tmp_jsonl.exists(): tmp_jsonl.unlink()

if __name__ == "__main__":
    unittest.main()
