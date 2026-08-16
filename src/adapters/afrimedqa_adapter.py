"""
AfriMed-QA v2 Data Adapter: Normalizes raw Pan-African clinical Q&A CSV into canonical schema.
Extracts clinical prompts, specialty classifications, and country metadata across 16 African nations.
"""

import json
import pandas as pd
from pathlib import Path
from src.adapters.base_adapter import validate_canonical_record

def process_afrimedqa_csv(csv_path: str, out_jsonl_path: str) -> int:
    """Reads raw AfriMed-QA CSV, extracts clinical complaints, and exports canonical JSONL."""
    if not Path(csv_path).exists():
        print(f"Warning: Raw file {csv_path} not found.")
        return 0
        
    df = pd.read_csv(csv_path)
    Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            prompt = str(row.get("prompt", row.get("question", ""))).strip()
            specialty = str(row.get("specialty", "General_Medicine")).strip()
            country = str(row.get("country", "Africa")).strip()
            
            if not prompt or len(prompt) < 10 or prompt.lower() == "nan":
                continue
                
            text = f"[{country} Clinical Complaint] {prompt}"
            
            record = {
                "id": f"AFR_{idx:06d}",
                "text": text,
                "labels": [specialty],
                "language_variant": "Local_Clinical_Idioms",
                "severity_level": "Urgent",
                "icd10_codes": ["R69"],
                "source": "afrimedqa"
            }
            if validate_canonical_record(record):
                f.write(json.dumps(record) + "\n")
                count += 1
                
    return count
