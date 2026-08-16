"""
DDXPlus Data Adapter: Normalizes raw DDXPlus CSV files and JSON evidence definitions into canonical schema format.
Decodes coded evidence strings (e.g., E_91, E_55_@_V_89) into human-readable clinical symptom text.
"""

import json
import pandas as pd
from pathlib import Path
from src.adapters.base_adapter import validate_canonical_record

def load_evidence_map(evidences_json_path: Path) -> dict:
    """Loads English evidence definitions from release_evidences.json."""
    if not evidences_json_path.exists():
        return {}
    with open(evidences_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_evidence_phrase(ev_code_val: str, evidence_map: dict) -> str:
    """Decodes a single evidence code into natural English text."""
    if "_@_" in ev_code_val:
        parts = ev_code_val.split("_@_")
        ev_code, val_code = parts[0], parts[1]
        
        if ev_code in evidence_map:
            ev_info = evidence_map[ev_code]
            question = ev_info.get("question_en", ev_code).strip()
            val_meanings = ev_info.get("value_meaning", {})
            
            if val_code in val_meanings:
                val_text = val_meanings[val_code].get("en", val_code)
                return f"{question.rstrip('?')} ({val_text})"
            else:
                return f"{question.rstrip('?')} ({val_code})"
        return ev_code_val
    else:
        if ev_code_val in evidence_map:
            ev_info = evidence_map[ev_code_val]
            return ev_info.get("question_en", ev_code_val).rstrip("?")
        return ev_code_val

def process_ddxplus_csv(csv_path: str, out_jsonl_path: str, limit: int = 50000) -> int:
    """Reads raw DDXPlus CSV, decodes evidence codes into clinical sentences, and exports canonical JSONL."""
    csv_p = Path(csv_path)
    if not csv_p.exists():
        # Fallback check for manual data/DDXPlus path
        alt_path = Path("data/DDXPlus/release_train_patients/release_train_patients")
        if alt_path.exists():
            csv_p = alt_path
        else:
            print(f"Warning: Raw file {csv_path} not found.")
            return 0
            
    # Locate release_evidences.json
    ev_json_path = csv_p.parent.parent / "release_evidences.json"
    if not ev_json_path.exists():
        ev_json_path = Path("data/DDXPlus/release_evidences.json")
    if not ev_json_path.exists():
        ev_json_path = Path("data/raw/ddxplus/release_evidences.json")
        
    evidence_map = load_evidence_map(ev_json_path)
    
    df = pd.read_csv(csv_p)
    Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            if idx >= limit:
                break
            age = row.get("AGE", "adult")
            sex = "male" if str(row.get("SEX", "M")).upper() == "M" else "female"
            pathology = str(row.get("PATHOLOGY", "Unknown_Condition")).strip()
            raw_evidences = row.get("EVIDENCES", "[]")
            
            if isinstance(raw_evidences, str) and raw_evidences.startswith("["):
                try:
                    ev_list = eval(raw_evidences)
                except Exception:
                    ev_list = [raw_evidences]
            elif isinstance(raw_evidences, list):
                ev_list = raw_evidences
            else:
                ev_list = [str(raw_evidences)]
                
            decoded_symptoms = [clean_evidence_phrase(ev, evidence_map) for ev in ev_list if ev]
            
            if not decoded_symptoms:
                symptoms_str = "general malaise and fatigue"
            else:
                symptoms_str = "; ".join(decoded_symptoms)
                
            text = f"Patient is a {age}-year-old {sex} presenting with: {symptoms_str}."
            
            record = {
                "id": f"DDX_{idx:07d}",
                "text": text,
                "labels": [pathology],
                "language_variant": "Standard_Clinical",
                "severity_level": "Urgent",
                "icd10_codes": ["R50.9"],
                "source": "ddxplus"
            }
            if validate_canonical_record(record):
                f.write(json.dumps(record) + "\n")
                count += 1
                
    return count
