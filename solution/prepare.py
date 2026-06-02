"""Task 1.2 — prepare cleaned data for modelling.

Materialises the model-ready tensors once so that train.py spends its whole time
budget on optimisation rather than JPEG decoding. Every image is decoded and
resized to a fixed SQUARE resolution (IMG_SIZE), which is also where the
train-only aspect-ratio shortcut identified in clean.py is removed.

Caches under artifacts/prepared/: train, calibration, validation,
calibration_augmented and validation_augmented. The predict split is
intentionally NOT prepared here — per the exercise it may change after training
and is decoded on demand inside predict.py / predict_augmented.py.
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from amls_common import (
    ARTIFACTS_DIR,
    DATA_DIR,
    IMG_SIZE,
    Budget,
    _to_bytes,
    decode_square,
    load_split,
    save_prepared,
    split_files,
    to_binary_label,
)

CLEAN_INDEX = ARTIFACTS_DIR / "clean" / "clean_index.parquet"
EVAL_SPLITS = ("calibration", "validation", "calibration_augmented", "validation_augmented")


def _decode_worker(b: bytes) -> np.ndarray:
    return decode_square(b, IMG_SIZE)


def prepare_train(pool) -> None:
    """Decode the cleaned training rows into a cached uint8 array."""
    if CLEAN_INDEX.exists():
        idx = pd.read_parquet(CLEAN_INDEX)
        print(f"prepare.py: using cleaned index with {len(idx)} rows")
        x = np.empty((len(idx), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        y = idx["label"].to_numpy(dtype=np.int64)
        pos = 0
        for fname, grp in idx.groupby("file", sort=True):
            df = pd.read_parquet(DATA_DIR / "train" / fname, columns=["image"])
            rows = grp["row"].to_numpy()
            raws = [_to_bytes(df["image"].iloc[int(r)]) for r in rows]
            dec = pool.map(_decode_worker, raws, chunksize=64)
            x[pos:pos + len(dec)] = np.stack(dec)
            pos += len(dec)
        save_prepared("train", x, y)
    else:  # fallback: no clean index, decode the raw split
        print("prepare.py: clean_index missing, decoding raw train split")
        x, y = load_split("train")
        save_prepared("train", x, y)
    print(f"prepare.py: cached train -> {x.shape}, real={int((y==0).sum())} ai={int((y==1).sum())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare features from cleaned training data.")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    args = parser.parse_args()
    budget = Budget(args.timeout_seconds)

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Expected read-only data at {DATA_DIR}")

    pool = mp.Pool(min(8, mp.cpu_count() or 1))
    try:
        prepare_train(pool)
        for split in EVAL_SPLITS:
            if not (DATA_DIR / split).is_dir():
                print(f"prepare.py: split {split} absent, skipping")
                continue
            x, y = load_split(split)
            save_prepared(split, x, y)
            print(f"prepare.py: cached {split} -> {x.shape}")
            if budget.expired():
                print("prepare.py: budget reached, stopping after current split")
                break
    finally:
        pool.close()
        pool.join()

    print("prepare.py: done (predict/ intentionally not prepared).")


if __name__ == "__main__":
    main()
