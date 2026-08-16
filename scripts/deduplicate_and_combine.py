"""
Deduplication & Master Combination Script:
Ingests all canonical JSONL files from data/canonical/, performs MD5 text hash deduplication,
and exports the master dataset files data/combined_dataset.csv and data/combined_dataset.jsonl.
"""

import json
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.adapters.base_adapter import compute_md5_hash

CANONICAL_DIR = Path("data/canonical")
OUT_CSV = Path("data/combined_dataset.csv")
OUT_JSONL = Path("data/combined_dataset.jsonl")

def merge_and_deduplicate():
    seen_hashes = set()
    unique_records = []
    total_raw_read = 0
    
    print("--- Running MD5 Text Hash Deduplication & Combination ---")
    for jsonl_file in CANONICAL_DIR.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                total_raw_read += 1
                rec = json.loads(line.strip())
                h = compute_md5_hash(rec["text"])
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_records.append(rec)
    
    duplicates_removed = total_raw_read - len(unique_records)
    dedup_rate = (duplicates_removed / total_raw_read * 100) if total_raw_read > 0 else 0
    
    print(f"Total Raw Input Records: {total_raw_read:,}")
    print(f"Duplicates Removed: {duplicates_removed:,} ({dedup_rate:.2f}% dedup rate)")
    print(f"Unique Deduplicated Records: {len(unique_records):,}")
    
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    
    # Save Master JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in unique_records:
            f.write(json.dumps(rec) + "\n")
            
    # Save Master CSV
    csv_rows = []
    for rec in unique_records:
        csv_rows.append({
            "id": rec["id"],
            "text": rec["text"],
            "labels": "|".join(rec["labels"]),
            "source": rec["source"]
        })
    df = pd.DataFrame(csv_rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    
    print(f"Saved Master Dataset -> CSV: {OUT_CSV} ({len(df):,} rows)")
    print(f"Saved Master Dataset -> JSONL: {OUT_JSONL} ({len(unique_records):,} records)")

if __name__ == "__main__":
    merge_and_deduplicate()
