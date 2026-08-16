"""
Kaggle 773 Data Adapter: Normalizes raw Kaggle 773-disease symptom vector CSV into canonical schema.
Converts active binary symptom indicators into natural chief complaints.
"""

import json
import pandas as pd
from pathlib import Path
from src.adapters.base_adapter import validate_canonical_record

def process_kaggle773_csv(csv_path: str, out_jsonl_path: str, limit: int = 50000) -> int:
    """Reads Kaggle 773 CSV, converts symptom matrices to text, and exports canonical JSONL."""
    csv_p = Path(csv_path)
    if not csv_p.exists():
        # Fallback check for root data/ directory
        alt_path = Path("data/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
        if alt_path.exists():
            csv_p = alt_path
        else:
            print(f"Warning: Raw file {csv_path} not found.")
            return 0
        
    df = pd.read_csv(csv_p)
    disease_col = df.columns[0]
    symptom_cols = df.columns[1:]
    
    Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            if idx >= limit:
                break
            disease = str(row[disease_col]).strip()
            active_symptoms = [col.replace("_", " ").strip() for col in symptom_cols if row[col] == 1]
            
            if not active_symptoms:
                symptoms_str = "general body weakness and discomfort"
            else:
                symptoms_str = ", ".join(active_symptoms)
            
            text = f"I have been experiencing {symptoms_str}."
            
            record = {
                "id": f"KAG_{idx:06d}",
                "text": text,
                "labels": [disease],
                "language_variant": "Standard_English",
                "severity_level": "Routine",
                "icd10_codes": ["R69"],
                "source": "kaggle773"
            }
            if validate_canonical_record(record):
                f.write(json.dumps(record) + "\n")
                count += 1
                
    return count
