"""Task 1.2 — train the binary real/AI detector under a strict CPU budget.

Trains SmallCNN from scratch on the cleaned, squared training images, calibrates
the decision threshold on the calibration split for FPR <= 20%, selects the best
epoch on the held-out validation split, and writes the checkpoint + operating
threshold to artifacts/task02/. The best checkpoint is flushed every time the
validation objective improves, so an early timeout still leaves a usable model.
"""

import argparse
import json

import numpy as np

from amls_common import (
    FPR_LIMIT,
    TARGET_FPR,
    TASK02_DIR,
    Budget,
    fpr_recall,
    load_prepared,
    seed_everything,
    setup_threads,
    summarize,
)
from amls_model import load_checkpoint, predict_scores, train_model

CKPT = TASK02_DIR / "model.pt"


def evaluate(split: str, model, mean, std, thr: float) -> dict:
    try:
        x, y = load_prepared(split)
    except FileNotFoundError:
        return {"split": split, "missing": True}
    scores = predict_scores(model, x, mean, std)
    out = summarize(scores, y, thr)
    out["split"] = split
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 2 model.")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=12)
    args = parser.parse_args()

    seed_everything()
    setup_threads()
    budget = Budget(args.timeout_seconds)

    train_x, train_y = load_prepared("train")
    cal_x, cal_y = load_prepared("calibration")
    val_x, val_y = load_prepared("validation")
    print(f"train.py: train={train_x.shape} cal={cal_x.shape} val={val_x.shape}")

    best = train_model(
        train_x, train_y, cal_x, cal_y, val_x, val_y,
        budget=budget, ckpt_path=CKPT,
        k=args.k, max_epochs=args.max_epochs, augment=False, target_fpr=TARGET_FPR,
    )

    # final report from the saved best checkpoint
    model, mean, std, thr, meta = load_checkpoint(CKPT)
    report = {"best_epoch": best.get("epoch"), "threshold": thr, "fpr_limit": FPR_LIMIT,
              "train_seconds": round(budget.elapsed(), 1)}
    for split in ("validation", "validation_augmented", "calibration"):
        report[split] = evaluate(split, model, mean, std, thr)
    (TASK02_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    print("\ntrain.py: final operating point (threshold calibrated on calibration):")
    for split in ("validation", "validation_augmented"):
        r = report[split]
        if not r.get("missing"):
            ok = "OK" if r["fpr_real"] <= FPR_LIMIT else "VIOLATION"
            print(f"  {split:22s} fpr_real={r['fpr_real']:.3f} recall_ai={r['recall_ai']:.3f} [{ok}]")
    print(f"  wrote {CKPT} and metrics.json")


if __name__ == "__main__":
    main()
