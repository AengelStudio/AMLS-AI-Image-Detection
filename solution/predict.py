"""Task 1.2 — inference on the holdout predict split.

Loads the Task 2 checkpoint + calibrated threshold, decodes the predict images
on demand (the split is deliberately not prepared ahead of time), and writes
artifacts/task02/predictions.csv with columns row_id,predicted_label.
"""

import argparse

import numpy as np
import pandas as pd

from amls_common import (
    DATA_DIR,
    IMG_SIZE,
    TASK02_DIR,
    _to_bytes,
    decode_square,
    seed_everything,
    setup_threads,
    split_files,
)
from amls_model import load_checkpoint, predict_scores

CKPT = TASK02_DIR / "model.pt"
PREDICTIONS_PATH = TASK02_DIR / "predictions.csv"


def load_predict():
    files = split_files("predict")
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR / 'predict'}")
    row_ids, imgs = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["row_id", "image"])
        row_ids.extend(int(r) for r in df["row_id"])
        imgs.extend(decode_square(_to_bytes(v), IMG_SIZE) for v in df["image"])
    return np.asarray(row_ids, dtype=np.int64), np.stack(imgs).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict labels for Task 2.")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    parser.parse_args()
    seed_everything()
    setup_threads()

    model, mean, std, thr, meta = load_checkpoint(CKPT)
    row_ids, x = load_predict()
    scores = predict_scores(model, x, mean, std)
    pred = (scores >= thr).astype(int)

    out = pd.DataFrame({"row_id": row_ids, "predicted_label": pred})
    out = out.sort_values("row_id", ignore_index=True).astype({"row_id": int, "predicted_label": int})
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS_PATH, index=False)
    print(f"predict.py: wrote {len(out)} rows to {PREDICTIONS_PATH} "
          f"(threshold={thr:.3f}, predicted ai={int(pred.sum())}/{len(pred)})")


if __name__ == "__main__":
    main()
