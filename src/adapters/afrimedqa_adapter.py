"""
AfriMed-QA v2 Data Adapter: Normalizes raw Pan-African clinical Q&A CSV into canonical schema.
Extracts clinical prompts, clean question text, specialty classifications, and country metadata across 18 African nations.
"""

import json
import pandas as pd
from pathlib import Path
from src.adapters.base_adapter import validate_canonical_record

def process_afrimedqa_csv(csv_path: str, out_jsonl_path: str) -> int:
    """Reads raw AfriMed-QA CSV, handles consumer prompt concatenation, and exports canonical JSONL."""
    csv_p = Path(csv_path)
    if not csv_p.exists():
        # Fallback check for manual data/Afrimed folder
        alt_path = Path("data/Afrimed/afri_med_qa_15k_v2.5_phase_2_15275.csv")
        if alt_path.exists():
            csv_p = alt_path
        else:
            print(f"Warning: Raw file {csv_path} not found.")
            return 0
            
    df = pd.read_csv(csv_p)
    Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            prompt = str(row.get("prompt", "")).strip()
            if prompt == "nan":
                prompt = ""
                
            q_clean = str(row.get("question_clean", row.get("question", ""))).strip()
            if q_clean == "nan":
                q_clean = ""
                
            specialty = str(row.get("specialty", "General_Medicine")).strip()
            if specialty == "nan" or not specialty:
                specialty = "General_Medicine"
                
            country = str(row.get("country", "Africa")).strip()
            if country == "nan" or not country:
                country = "Africa"
                
            q_type = str(row.get("question_type", "consumer_queries")).strip()
            
            # Concatenate prompt + clean question for consumer queries as specified in README
            if q_type == "consumer_queries" and prompt:
                full_text = f"[{country} Clinical Complaint] {prompt} {q_clean}"
            else:
                full_text = f"[{country} Clinical Complaint] {q_clean}"
                
            if not full_text or len(full_text) < 15:
                continue
                
            sample_id = str(row.get("sample_id", f"AFR_{idx:06d}")).strip()
            
            record = {
                "id": sample_id if sample_id != "nan" else f"AFR_{idx:06d}",
                "text": full_text,
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
