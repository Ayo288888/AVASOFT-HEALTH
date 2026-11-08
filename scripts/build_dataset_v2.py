"""
Builds AfroCare-Dx v2: one training file over the 60 conditions in data/conditions_v2.csv plus OTHER.

Sources: combined_dataset.csv (DDXPlus, Kaggle 773; SymCAT and specialty-labelled AfriMed-QA are
dropped by the label map), the Symptom2Disease, Zenodo and AfriMed-QA seed canonical files, and
generated cases (source "synthetic", reviewed=False until a clinician signs them off).

Usage:
  python scripts/build_dataset_v2.py --data path/to/data --generated data/generated/cases_batch1.csv
Writes <data>/afrocare_dx_v2.csv and prints per-label and per-split counts.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_label_map import load_conditions, target_for

CAP_PER_CONDITION = 5_000  # stops DDXPlus respiratory classes (131k common-cold rows) swamping the rest
CAP_OTHER = 20_000
CAP_PER_OTHER_DISEASE = 500  # Roseola alone was 17% of OTHER, teaching "fever + rash = OTHER" next to thin measles
# Natural patient text first, templated symptom lists last.
SOURCE_PRIORITY = ["symptom2disease", "synthetic", "afrimedqa_seed", "zenodo_infectious", "kaggle773", "ddxplus"]


def word_set(text: str) -> str:
    """Order- and number-insensitive signature: templated rows that only reorder symptoms or change the age match."""
    return " ".join(sorted(set(re.findall(r"[a-z]+", text.lower()))))


def drop_duplicates_and_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps one row per word set; drops word sets that appear under more than one label."""
    h = df["text"].map(word_set)
    conflicting = df.groupby(h)["label"].transform("nunique") > 1
    return df[~conflicting].loc[~h[~conflicting].duplicated()]


def cap_per_label(df: pd.DataFrame, caps: dict, seed: int,
                  cap_per_other_disease: int = CAP_PER_OTHER_DISEASE) -> pd.DataFrame:
    rank = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    shuffled = df.sample(frac=1, random_state=seed)
    shuffled = shuffled.assign(_p=shuffled["source"].map(rank).fillna(len(rank)))
    ordered = shuffled.sort_values("_p", kind="stable").drop(columns="_p")
    disease_rank = ordered.groupby(["label", ordered["original_label"].str.lower()]).cumcount()
    ordered = ordered[(ordered["label"] != "OTHER") | (disease_rank < cap_per_other_disease)]
    limit = ordered["label"].map(caps).fillna(len(ordered))
    return ordered[ordered.groupby("label").cumcount() < limit]


def split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """80/10/10 per label, so even a 10-row condition has at least one val and one test row."""
    parts = []
    for _, g in df.groupby("label"):
        g = g.sample(frac=1, random_state=seed)
        n = len(g)
        n_test = max(1, round(n * 0.1)) if n >= 3 else 0
        n_val = max(1, round(n * 0.1)) if n >= 3 else 0
        labels = ["test"] * n_test + ["val"] * n_val + ["train"] * (n - n_test - n_val)
        parts.append(g.assign(split=labels))
    return pd.concat(parts).sort_values(["split", "label"], kind="stable")


def load_sources(data: Path, generated: Path) -> pd.DataFrame:
    conditions = load_conditions()
    keep = set(conditions) | {"OTHER"}
    frames = []

    for chunk in pd.read_csv(data / "combined_dataset.csv", usecols=["id", "text", "labels", "source"],
                             chunksize=200_000, dtype=str):
        chunk = chunk.dropna(subset=["text", "labels"])
        chunk["original_label"] = chunk["labels"].str.split("|").str[0].str.strip()
        chunk["label"] = [target_for(s, l) for s, l in zip(chunk["source"], chunk["original_label"])]
        frames.append(chunk[chunk["label"].isin(keep)][["id", "text", "label", "source", "original_label"]])

    for name in ("symptom2disease", "zenodo_infectious", "afrimedqa_seed"):
        recs = [json.loads(line) for line in open(data / "canonical" / f"{name}.jsonl", encoding="utf-8")]
        df = pd.DataFrame([{"id": r["id"], "text": r["text"], "source": r["source"], "original_label": r["labels"][0],
                            "label": target_for(r["source"], r["labels"][0])} for r in recs])
        frames.append(df[df["label"].isin(keep)])

    gen = pd.read_csv(generated, dtype=str).fillna("")
    frames.append(pd.DataFrame({
        "id": [f"GEN_{i:05d}" for i in range(len(gen))],
        "text": gen["text"], "label": gen["condition"], "source": "synthetic", "original_label": gen["condition"],
    }))

    df = pd.concat(frames, ignore_index=True)
    df["reviewed"] = df["source"] != "synthetic"  # generated cases stay unreviewed until a clinician signs off
    df["label_id"] = df["label"].map(conditions).fillna("OTHER")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--generated", default=str(ROOT / "data" / "generated" / "cases_batch1.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    data = Path(args.data)

    df = load_sources(data, Path(args.generated))
    print(f"Mapped rows: {len(df):,}")
    df = drop_duplicates_and_conflicts(df)
    print(f"After removing duplicates and conflicting labels: {len(df):,}")
    caps = {label: CAP_PER_CONDITION for label in load_conditions()} | {"OTHER": CAP_OTHER}
    df = split(cap_per_label(df, caps, args.seed), args.seed)

    out = data / "afrocare_dx_v2.csv"
    df[["id", "text", "label_id", "label", "original_label", "source", "reviewed", "split"]].to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(df):,} rows to {out}")
    print("\nRows per split:", df["split"].value_counts().to_dict())
    print("Rows per source:", df["source"].value_counts().to_dict())
    counts = df.groupby(["label_id", "label"]).size().reset_index(name="rows").sort_values("label_id")
    print("\n" + counts.to_string(index=False))


if __name__ == "__main__":
    main()
