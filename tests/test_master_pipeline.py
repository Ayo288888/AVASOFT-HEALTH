import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.base_adapter import compute_md5_hash, validate_canonical_record

class TestMasterPipeline(unittest.TestCase):
    def test_deduplication_logic(self):
        h1 = compute_md5_hash("Patient has high fever and cough.")
        h2 = compute_md5_hash("patient HAS high fever AND cough.")
        self.assertEqual(h1, h2)

    def test_canonical_record_schema(self):
        rec = {
            "id": "TEST_001",
            "text": "Fever and chills",
            "labels": ["Malaria"],
            "language_variant": "Standard_Clinical",
            "severity_level": "Urgent",
            "icd10_codes": ["B50.9"],
            "source": "test"
        }
        self.assertTrue(validate_canonical_record(rec))

if __name__ == "__main__":
    unittest.main()
