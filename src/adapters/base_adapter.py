"""
Base Adapter Framework & Helper Functions for Canonical Schema Normalization.
Ensures uniform keys, text cleaning, validation, and MD5 text hashing.
"""

import hashlib
import json
from pathlib import Path

CANONICAL_KEYS = {"id", "text", "labels", "language_variant", "severity_level", "icd10_codes", "source"}

def compute_md5_hash(text: str) -> str:
    """Computes MD5 hash on normalized lowercase whitespace-collapsed text string."""
    normalized = " ".join(text.strip().lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

def validate_canonical_record(record: dict) -> bool:
    """Validates that a canonical record meets schema constraints."""
    if not isinstance(record, dict):
        return False
    if not CANONICAL_KEYS.issubset(record.keys()):
        return False
    if not record["text"] or not isinstance(record["text"], str):
        return False
    if not isinstance(record["labels"], list) or len(record["labels"]) == 0:
        return False
    if not record["source"] or not isinstance(record["source"], str):
        return False
    return True
