import unittest
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.afrimedqa_adapter import process_afrimedqa_csv

class TestAfriMedQAAdapter(unittest.TestCase):
    def test_afrimedqa_adapter(self):
        tmp_csv = Path("tests/tmp_afri.csv")
        tmp_jsonl = Path("tests/tmp_afri.jsonl")
        
        df = pd.DataFrame({
            "prompt": ["A 25yo woman in Nigeria presents with severe fever and joint pain."],
            "specialty": ["Infectious_Diseases"],
            "country": ["Nigeria"]
        })
        df.to_csv(tmp_csv, index=False)
        
        count = process_afrimedqa_csv(str(tmp_csv), str(tmp_jsonl))
        self.assertEqual(count, 1)
        
        if tmp_csv.exists(): tmp_csv.unlink()
        if tmp_jsonl.exists(): tmp_jsonl.unlink()

if __name__ == "__main__":
    unittest.main()
