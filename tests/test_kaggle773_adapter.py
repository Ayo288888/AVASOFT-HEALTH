import unittest
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.kaggle773_adapter import process_kaggle773_csv
from src.adapters.symcat_adapter import process_symcat_csv

class TestKaggleAndSymcatAdapters(unittest.TestCase):
    def test_kaggle773_adapter(self):
        tmp_csv = Path("tests/tmp_kaggle.csv")
        tmp_jsonl = Path("tests/tmp_kaggle.jsonl")
        
        df = pd.DataFrame({
            "diseases": ["Malaria_Plasmodium_falciparum"],
            "fever": [1],
            "chills": [1]
        })
        df.to_csv(tmp_csv, index=False)
        
        count = process_kaggle773_csv(str(tmp_csv), str(tmp_jsonl))
        self.assertEqual(count, 1)
        
        if tmp_csv.exists(): tmp_csv.unlink()
        if tmp_jsonl.exists(): tmp_jsonl.unlink()

    def test_symcat_adapter(self):
        tmp_csv = Path("tests/tmp_symcat.csv")
        tmp_jsonl = Path("tests/tmp_symcat.jsonl")
        
        df = pd.DataFrame({
            "disease_name": ["Tuberculosis_Pulmonary"]
        })
        df.to_csv(tmp_csv, index=False)
        
        count = process_symcat_csv(str(tmp_csv), str(tmp_jsonl))
        self.assertEqual(count, 1)
        
        if tmp_csv.exists(): tmp_csv.unlink()
        if tmp_jsonl.exists(): tmp_jsonl.unlink()

if __name__ == "__main__":
    unittest.main()
