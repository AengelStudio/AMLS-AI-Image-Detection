"""Task 1.3 — robust training via data augmentation.

Continues from the Task 2 checkpoint and fine-tunes SmallCNN with on-the-fly
augmentation (downscale-upscale, JPEG recompression, blur, photometric jitter,
noise) so the detector stops relying on fragile high-frequency / compression
cues. The threshold is recalibrated on the combined clean+augmented calibration
real images so the <=20% FPR constraint holds under perturbation as well, and
the result is compared against Task 2 on both validation and validation_augmented.
"""

import argparse
import json

import numpy as np

from amls_common import (
    FPR_LIMIT,
    TARGET_FPR,
    TASK02_DIR,
    TASK03_DIR,
    Budget,
    calibrate_threshold,
    load_prepared,
    seed_everything,
    setup_threads,
    summarize,
)
from amls_model import load_checkpoint, predict_scores, save_checkpoint, train_model

TASK02_CKPT = TASK02_DIR / "model.pt"
CKPT = TASK03_DIR / "model.pt"
# Conservative operating target for Task 3: below Task 2's 0.15 because the few
# real calibration images make the FPR noisy and the augmented distribution
# shifts scores, so extra margin keeps validation FPR <= 0.20 on clean AND
# augmented data.
TASK3_TARGET_FPR = 0.14


def evaluate(split, model, mean, std, thr):
    try:
        x, y = load_prepared(split)
    except FileNotFoundError:
        return {"split": split, "missing": True}
    scores = predict_scores(model, x, mean, std)
    out = summarize(scores, y, thr)
    out["split"] = split
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 3 augmented model.")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--aug_strength", type=float, default=1.0)
    args = parser.parse_args()

    seed_everything()
    setup_threads()
    budget = Budget(args.timeout_seconds)

    train_x, train_y = load_prepared("train")
    # robust calibration set: clean + augmented calibration real images
    cal_x, cal_y = load_prepared("calibration")
    # combined validation for epoch selection: balances clean AND augmented
    # performance, which is the explicit Task 3 goal (don't pick the least-robust
    # early epoch just because it scores best on clean data).
    val_x, val_y = load_prepared("validation")
    try:
        cax, cay = load_prepared("calibration_augmented")
        cal_x = np.concatenate([cal_x, cax], 0)
        cal_y = np.concatenate([cal_y, cay], 0)
        vax, vay = load_prepared("validation_augmented")
        sel_val_x = np.concatenate([val_x, vax], 0)
        sel_val_y = np.concatenate([val_y, vay], 0)
    except FileNotFoundError:
        sel_val_x, sel_val_y = val_x, val_y
    print(f"train_augmented.py: train={train_x.shape} cal={cal_x.shape} "
          f"selection_val={sel_val_x.shape}")

    # warm-start from the Task 2 checkpoint when available
    init_state = None
    if TASK02_CKPT.exists():
        m0, *_ = load_checkpoint(TASK02_CKPT)
        init_state = m0.state_dict()

    best = train_model(
        train_x, train_y, cal_x, cal_y, sel_val_x, sel_val_y,
        budget=budget, ckpt_path=CKPT,
        k=args.k, max_epochs=args.max_epochs, augment=True, aug_strength=args.aug_strength,
        init_state=init_state, target_fpr=TARGET_FPR, lr=5e-4,
    )

    model, mean, std, thr, meta = load_checkpoint(CKPT)
    # recalibrate the operating threshold conservatively on the combined
    # clean+augmented calibration real images, then persist it on the checkpoint
    cal_scores = predict_scores(model, cal_x, mean, std)
    thr = calibrate_threshold(cal_scores[cal_y == 0], target_fpr=TASK3_TARGET_FPR)
    save_checkpoint(CKPT, model, mean, std, thr, meta)
    report = {"best_epoch": best.get("epoch"), "threshold": thr, "fpr_limit": FPR_LIMIT,
              "calibration_target": TASK3_TARGET_FPR,
              "train_seconds": round(budget.elapsed(), 1)}
    for split in ("validation", "validation_augmented", "calibration", "calibration_augmented"):
        report[split] = evaluate(split, model, mean, std, thr)
    (TASK03_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    print("\ntrain_augmented.py: robust operating point:")
    for split in ("validation", "validation_augmented"):
        r = report[split]
        if not r.get("missing"):
            ok = "OK" if r["fpr_real"] <= FPR_LIMIT else "VIOLATION"
            print(f"  {split:22s} fpr_real={r['fpr_real']:.3f} recall_ai={r['recall_ai']:.3f} [{ok}]")
    print(f"  wrote {CKPT} and metrics.json")


if __name__ == "__main__":
    main()
