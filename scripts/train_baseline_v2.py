"""
TF-IDF + linear SVM baseline on AfroCare-Dx v2: the score a transformer has to beat.

Trains on the train split, scores on the test split, and never counts generated cases toward a score.

Usage:
  python scripts/train_baseline_v2.py --data path/to/data
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent


def evaluate(df: pd.DataFrame, preds) -> dict:
    scored = df.assign(pred=list(preds))
    scored = scored[scored["source"] != "synthetic"]
    labels = sorted(scored["label"].unique())
    report = classification_report(scored["label"], scored["pred"], labels=labels, output_dict=True, zero_division=0)
    per_label = pd.DataFrame({l: report[l] for l in labels}).T.rename(columns={"support": "test_rows"})
    return {
        "rows": len(scored),
        "accuracy": float((scored["label"] == scored["pred"]).mean()),
        "macro_f1": float(per_label["f1-score"].mean()),
        "by_source": (scored["label"] == scored["pred"]).groupby(scored["source"]).mean(),
        "per_label": per_label,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()

    df = pd.read_csv(Path(args.data) / "afrocare_dx_v2.csv", dtype=str)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        LinearSVC(class_weight="balanced", random_state=42),
    )
    model.fit(train["text"], train["label"])

    result = evaluate(test, model.predict(test["text"]))
    print(f"Scored test rows: {result['rows']:,}")
    print(f"Accuracy: {result['accuracy']:.3f}   Macro F1: {result['macro_f1']:.3f}")
    print("\nAccuracy by source:\n" + result["by_source"].round(3).to_string())
    per_label = result["per_label"].sort_values("f1-score").round(3)
    print("\nEvery label, worst first:\n" + per_label[["precision", "recall", "f1-score", "test_rows"]].to_string())


if __name__ == "__main__":
    main()
