"""
Symptom2Disease Data Adapter: Normalizes the Kaggle Symptom2Disease CSV (niyarrbarman/symptom2disease)
into canonical schema. Rows are first-person patient descriptions, 50 per disease across 24 diseases.
"""

import json
from pathlib import Path

import pandas as pd

from src.adapters.base_adapter import validate_canonical_record


def process_symptom2disease_csv(csv_path: str, out_jsonl_path: str) -> int:
    """Reads Symptom2Disease CSV (columns: label, text) and exports canonical JSONL."""
    if not Path(csv_path).exists():
        print(f"Warning: Raw file {csv_path} not found.")
        return 0

    df = pd.read_csv(csv_path, dtype=str)
    Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            text = str(row.get("text") or "").strip()
            label = str(row.get("label") or "").strip()
            if not text or text == "nan" or not label or label == "nan":
                continue

            record = {
                "id": f"S2D_{idx:05d}",
                "text": text,
                "labels": [label],
                "language_variant": "Patient_Narrative",
                "severity_level": "Routine",
                "icd10_codes": ["R69"],
                "source": "symptom2disease",
            }
            if validate_canonical_record(record):
                f.write(json.dumps(record) + "\n")
                count += 1

    return count
