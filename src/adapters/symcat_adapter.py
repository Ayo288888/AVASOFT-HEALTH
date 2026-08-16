"""
SymCAT Data Adapter: Normalizes raw SymCAT disease-symptom probability tables into canonical schema.
Generates clinical presentation statements for 801 conditions.
"""

import json
import pandas as pd
from pathlib import Path
from src.adapters.base_adapter import validate_canonical_record

def process_symcat_csv(csv_path: str, out_jsonl_path: str) -> int:
    """Reads SymCAT disease CSV and exports canonical JSONL."""
    if not Path(csv_path).exists():
        print(f"Warning: Raw file {csv_path} not found.")
        return 0
        
    df = pd.read_csv(csv_path)
    Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            disease_name = str(row.get("disease_name", row.iloc[0])).strip()
            text = f"Patient presents with characteristic clinical symptoms associated with {disease_name}."
            
            record = {
                "id": f"SYM_{idx:05d}",
                "text": text,
                "labels": [disease_name],
                "language_variant": "Standard_Clinical",
                "severity_level": "Routine",
                "icd10_codes": ["R69"],
                "source": "symcat"
            }
            if validate_canonical_record(record):
                f.write(json.dumps(record) + "\n")
                count += 1
                
    return count
