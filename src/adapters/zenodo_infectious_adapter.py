"""
Zenodo Infectious Disease Data Adapter: Normalizes the Zenodo infectious disease dataset
(DOI 10.5281/zenodo.13338116, CC BY 4.0) into canonical schema.
Each row is a disease followed by a variable number of symptom columns (ragged CSV).
"""

import csv
import json
from pathlib import Path

from src.adapters.base_adapter import validate_canonical_record


def process_zenodo_infectious_csv(csv_path: str, out_jsonl_path: str) -> int:
    """Reads the ragged Disease,Symptom_1..N CSV and exports canonical JSONL."""
    if not Path(csv_path).exists():
        print(f"Warning: Raw file {csv_path} not found.")
        return 0

    Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(csv_path, encoding="utf-8", newline="") as src, open(out_jsonl_path, "w", encoding="utf-8") as f:
        reader = csv.reader(src)
        next(reader, None)  # header
        for idx, row in enumerate(reader):
            if not row or not row[0].strip():
                continue
            disease = row[0].strip()

            # Rows have more fields than the header and repeat symptoms in different case.
            symptoms = list(dict.fromkeys(s.strip().lower() for s in row[1:] if s.strip()))
            if not symptoms:
                continue

            record = {
                "id": f"ZEN_{idx:05d}",
                "text": f"I have been experiencing {', '.join(symptoms)}.",
                "labels": [disease],
                "language_variant": "Standard_English",
                "severity_level": "Routine",
                "icd10_codes": ["R69"],
                "source": "zenodo_infectious",
            }
            if validate_canonical_record(record):
                f.write(json.dumps(record) + "\n")
                count += 1

    return count
