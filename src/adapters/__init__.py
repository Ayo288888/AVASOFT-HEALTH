"""
Source adapters package for canonical schema normalization.
"""

from src.adapters.base_adapter import compute_md5_hash, validate_canonical_record
from src.adapters.ddxplus_adapter import process_ddxplus_csv
from src.adapters.kaggle773_adapter import process_kaggle773_csv
from src.adapters.symcat_adapter import process_symcat_csv
from src.adapters.afrimedqa_adapter import process_afrimedqa_csv

__all__ = [
    "compute_md5_hash",
    "validate_canonical_record",
    "process_ddxplus_csv",
    "process_kaggle773_csv",
    "process_symcat_csv",
    "process_afrimedqa_csv"
]
