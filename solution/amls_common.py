"""Shared utilities for the AMLS AI-image-detection pipeline.

Imported by clean.py, prepare.py, train.py, predict.py, train_augmented.py and
predict_augmented.py. Keeping the data handling, model definition, calibration
and metrics in one place guarantees that training and inference use byte-for-byte
identical preprocessing — the single most common source of train/serve skew.

Design decisions (justified in the report):
  * All images are decoded and resized to a fixed SQUARE resolution. The training
    split leaks the label through aspect ratio (real/COCO images are non-square,
    AI images are square); the calibration/validation/predict splits are uniformly
    320x320. Squaring every image removes this train-only shortcut so the model is
    forced to learn transferable content/frequency cues.
  * CPU-only, fixed seeds, capped threads -> reproducible within a strict budget.
  * The operating threshold is calibrated automatically on a held-out calibration
    split to satisfy the <=20% false-positive-rate constraint, never hard-coded.
"""

from __future__ import annotations

import io
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 0
IMG_SIZE = 64             # model input resolution (square); CPU-budget driven
TARGET_FPR = 0.15         # calibrate below the 0.20 limit to keep holdout margin
FPR_LIMIT = 0.20          # hard constraint from the exercise

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
PREPARED_DIR = ARTIFACTS_DIR / "prepared"
TASK02_DIR = ARTIFACTS_DIR / "task02"
TASK03_DIR = ARTIFACTS_DIR / "task03"

LABELED_SPLITS = (
    "train",
    "calibration",
    "validation",
    "calibration_augmented",
    "validation_augmented",
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def setup_threads() -> None:
    """Cap CPU threads exactly like the Appendix C reference (max 8)."""
    n = min(8, os.cpu_count() or 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    try:
        import torch

        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Image decoding / dataset loading
# ---------------------------------------------------------------------------
def _to_bytes(v) -> bytes:
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, dict) and "bytes" in v:        # HuggingFace {bytes, path}
        return bytes(v["bytes"])
    if isinstance(v, np.ndarray):
        return v.tobytes()
    return bytes(v)


def decode_square(b: bytes, size: int = IMG_SIZE) -> np.ndarray:
    """Decode JPEG/PNG bytes to an (size, size, 3) uint8 RGB array.

    Resizing to a square deliberately discards aspect ratio: the evaluation
    splits are already square, so this matches train and test distributions.
    """
    im = Image.open(io.BytesIO(b)).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def to_binary_label(source_class: int) -> int:
    """0 -> real, {1..5} -> ai_generated."""
    return 0 if int(source_class) == 0 else 1


def split_files(split: str) -> list[Path]:
    return sorted((DATA_DIR / split).glob("*.parquet"))


def _decode_worker(b_bytes: bytes) -> np.ndarray:
    return decode_square(b_bytes, IMG_SIZE)


def load_split(split: str, size: int = IMG_SIZE, with_labels: bool = True,
               workers: int | None = None):
    """Load and decode a whole split.

    Returns (images_uint8[N,size,size,3], labels_int64[N]) for labeled splits or
    (images_uint8, row_ids_int32) when with_labels=False (the predict split).
    Decoding is parallelised across processes to stay within the time budget.
    """
    files = split_files(split)
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR / split}")

    imgs: list[np.ndarray] = []
    meta: list[int] = []
    raw: list[bytes] = []
    for f in files:
        if with_labels:
            df = pd.read_parquet(f, columns=["image", "source_class"])
            meta.extend(int(to_binary_label(c)) for c in df["source_class"])
        else:
            df = pd.read_parquet(f, columns=["row_id", "image"])
            meta.extend(int(r) for r in df["row_id"])
        raw.extend(_to_bytes(v) for v in df["image"])

    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    if workers > 1:
        import multiprocessing as mp

        with mp.Pool(workers) as pool:
            imgs = pool.map(_decode_worker, raw, chunksize=64)
    else:
        imgs = [_decode_worker(b) for b in raw]

    x = np.stack(imgs).astype(np.uint8)
    m = np.asarray(meta, dtype=np.int64)
    return x, m


# ---------------------------------------------------------------------------
# Prepared-cache I/O (written by prepare.py, read by train.py)
# ---------------------------------------------------------------------------
def cache_paths(split: str):
    return (PREPARED_DIR / f"{split}_x.npy", PREPARED_DIR / f"{split}_y.npy")


def save_prepared(split: str, x: np.ndarray, y: np.ndarray) -> None:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    px, py = cache_paths(split)
    np.save(px, x)
    np.save(py, y)


def load_prepared(split: str):
    px, py = cache_paths(split)
    if not px.exists():
        raise FileNotFoundError(f"Prepared cache missing: {px}. Run prepare.py first.")
    return np.load(px), np.load(py)


# ---------------------------------------------------------------------------
# Metrics & calibration
# ---------------------------------------------------------------------------
def fpr_recall(scores: np.ndarray, y: np.ndarray, thr: float):
    """scores = P(ai). Returns (false_positive_rate_on_real, recall_on_ai)."""
    pred = (scores >= thr).astype(int)
    real = y == 0
    ai = y == 1
    fpr = float(pred[real].mean()) if real.any() else float("nan")
    recall = float(pred[ai].mean()) if ai.any() else float("nan")
    return fpr, recall


def calibrate_threshold(scores_real: np.ndarray, target_fpr: float = TARGET_FPR) -> float:
    """Smallest threshold whose FPR on the calibration real images <= target_fpr.

    Implemented as the (1 - target_fpr) empirical quantile of real-image scores,
    nudged up to the next distinct score so the realised FPR does not exceed the
    target due to ties.
    """
    s = np.sort(np.asarray(scores_real, dtype=np.float64))
    thr = float(np.quantile(s, 1.0 - target_fpr, method="higher"))
    # ensure strictly: fraction of real with score >= thr is <= target
    frac = float((s >= thr).mean())
    if frac > target_fpr:
        higher = s[s > thr]
        if higher.size:
            thr = float(higher.min())
    return thr


def summarize(scores: np.ndarray, y: np.ndarray, thr: float) -> dict:
    fpr, recall = fpr_recall(scores, y, thr)
    pred = (scores >= thr).astype(int)
    acc = float((pred == y).mean())
    return {"threshold": float(thr), "fpr_real": fpr, "recall_ai": recall, "accuracy": acc,
            "n": int(len(y)), "n_real": int((y == 0).sum()), "n_ai": int((y == 1).sum())}


# ---------------------------------------------------------------------------
# Simple time budget helper
# ---------------------------------------------------------------------------
class Budget:
    """Tracks wall-clock against the per-script --timeout_seconds and leaves a
    safety reserve so the best checkpoint is always flushed before SIGTERM."""

    def __init__(self, timeout_seconds: float, reserve: float = 45.0):
        self.start = time.perf_counter()
        self.deadline = self.start + max(1.0, timeout_seconds - reserve)

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def expired(self) -> bool:
        return time.perf_counter() >= self.deadline

    def remaining(self) -> float:
        return self.deadline - time.perf_counter()
