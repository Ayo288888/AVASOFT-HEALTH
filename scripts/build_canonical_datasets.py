"""
Master Canonical Builder Script:
Processes raw datasets from data/ (DDXPlus, Kaggle 773, AfriMed-QA v2, SymCAT)
using source adapters and outputs canonical JSONL files to data/canonical/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.ddxplus_adapter import process_ddxplus_csv
from src.adapters.kaggle773_adapter import process_kaggle773_csv
from src.adapters.afrimedqa_adapter import process_afrimedqa_csv
from src.adapters.symcat_adapter import process_symcat_csv

def build_all_canonical():
    print("--- Building Canonical JSONL Datasets from Raw Sources ---")
    
    # 1. DDXPlus
    ddx_path = Path("data/DDXPlus/release_train_patients/release_train_patients")
    if not ddx_path.exists():
        ddx_path = Path("data/raw/ddxplus/test.csv")
    c1 = process_ddxplus_csv(str(ddx_path), "data/canonical/ddxplus.jsonl", limit=50000)
    print(f"DDXPlus Canonical Records: {c1:,}")
    
    # 2. Kaggle 773
    kag_path = Path("data/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
    if not kag_path.exists():
        kag_path = Path("data/raw/kaggle773/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
    c2 = process_kaggle773_csv(str(kag_path), "data/canonical/kaggle773.jsonl", limit=50000)
    print(f"Kaggle 773 Canonical Records: {c2:,}")
    
    # 3. AfriMed-QA v2
    afri_path = Path("data/Afrimed/afri_med_qa_15k_v2.5_phase_2_15275.csv")
    if not afri_path.exists():
        afri_path = Path("data/raw/afrimedqa/afrimedqa_train.csv")
    c3 = process_afrimedqa_csv(str(afri_path), "data/canonical/afrimedqa.jsonl")
    print(f"AfriMed-QA Canonical Records: {c3:,}")
    
    # 4. SymCAT
    sym_path = Path("data/raw/symcat_pipeline/symcat/symcat-801-diseases.csv")
    c4 = process_symcat_csv(str(sym_path), "data/canonical/symcat.jsonl")
    print(f"SymCAT Canonical Records: {c4:,}")

if __name__ == "__main__":
    build_all_canonical()
