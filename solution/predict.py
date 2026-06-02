"""Task 1.2: inference on holdout predict split (stub)."""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PREDICTIONS_PATH = ROOT / "artifacts" / "task02" / "predictions.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict labels for Task 2.")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    parser.parse_args()

    predict_dir = DATA_DIR / "predict"
    if not predict_dir.is_dir():
        raise FileNotFoundError(f"Expected predict split at {predict_dir}")

    frames = []
    for path in sorted(predict_dir.glob("*.parquet")):
        frames.append(pd.read_parquet(path, columns=["row_id"]))
    if not frames:
        raise FileNotFoundError(f"No parquet files in {predict_dir}")

    rows = pd.concat(frames, ignore_index=True).sort_values("row_id", ignore_index=True)
    rows["predicted_label"] = 0

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows[["row_id", "predicted_label"]].astype({"predicted_label": int}).to_csv(
        PREDICTIONS_PATH, index=False
    )
    print(f"predict.py: wrote {len(rows)} rows to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
