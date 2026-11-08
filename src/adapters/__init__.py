"""
Source adapters package for canonical schema normalization.
"""

from src.adapters.base_adapter import compute_md5_hash, validate_canonical_record
from src.adapters.ddxplus_adapter import process_ddxplus_csv
from src.adapters.kaggle773_adapter import process_kaggle773_csv
from src.adapters.symcat_adapter import process_symcat_csv
from src.adapters.afrimedqa_adapter import process_afrimedqa_csv
from src.adapters.symptom2disease_adapter import process_symptom2disease_csv
from src.adapters.zenodo_infectious_adapter import process_zenodo_infectious_csv

__all__ = [
    "process_symptom2disease_csv",
    "process_zenodo_infectious_csv",
    "compute_md5_hash",
    "validate_canonical_record",
    "process_ddxplus_csv",
    "process_kaggle773_csv",
    "process_symcat_csv",
    "process_afrimedqa_csv"
]
