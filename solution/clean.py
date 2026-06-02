"""Task 1.1 — dataset exploration and deterministic cleaning.

Scans the read-only training split, summarises the characteristics that could
leak the label (class balance, image size / aspect ratio, format, byte size),
and builds a *deterministic* cleaned index of the rows to keep for modelling.

Key finding driving the cleaning (see report): in `train/`, real (COCO) images
are non-square while AI images are square, so aspect ratio is an almost perfect
label predictor — yet every calibration/validation/predict image is 320x320.
Aspect ratio is therefore a train-only shortcut. The cleaning pipeline keeps a
de-duplicated, decodable set of rows and records that downstream preparation
must square every image (done in prepare.py) so the model cannot use it.

Outputs (under artifacts/clean/):
  * clean_index.parquet — rows to keep: [file, row, label, width, height, nbytes]
  * clean_stats.json     — exploration + cleaning summary for the report
"""

import argparse
import hashlib
import io
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from amls_common import ARTIFACTS_DIR, DATA_DIR, split_files, to_binary_label, _to_bytes

CLEAN_DIR = ARTIFACTS_DIR / "clean"


def _probe(b: bytes):
    """Return (ok, width, height, fmt) without loading full pixels when possible."""
    try:
        im = Image.open(io.BytesIO(b))
        w, h = im.size
        fmt = im.format or "?"
        im.load()  # force decode so truncated/corrupt streams are caught
        return True, int(w), int(h), str(fmt)
    except Exception:
        return False, -1, -1, "BAD"


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore and clean training data.")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    args = parser.parse_args()

    train_dir = DATA_DIR / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected training split at {train_dir}")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    files = split_files("train")
    records = []          # kept rows
    seen_hashes = set()
    n_total = n_dup = n_bad = 0
    class_counts = Counter()
    sq_by_label = defaultdict(Counter)          # label -> Counter(square/non-square)
    size_by_label = defaultdict(Counter)        # label -> Counter("WxH")
    fmt_counts = Counter()
    bytes_by_label = defaultdict(list)

    pool = mp.Pool(min(8, mp.cpu_count() or 1))
    try:
        for f in files:
            df = pd.read_parquet(f, columns=["image", "source_class"])
            raws = [_to_bytes(v) for v in df["image"]]
            labels = [int(to_binary_label(c)) for c in df["source_class"]]
            probes = pool.map(_probe, raws, chunksize=64)
            for row, (b, lab, (ok, w, h, fmt)) in enumerate(zip(raws, labels, probes)):
                n_total += 1
                class_counts[lab] += 1
                fmt_counts[fmt] += 1
                if not ok:
                    n_bad += 1
                    continue
                digest = hashlib.md5(b).hexdigest()
                if digest in seen_hashes:
                    n_dup += 1
                    continue
                seen_hashes.add(digest)
                sq_by_label[lab]["square" if w == h else "non_square"] += 1
                size_by_label[lab][f"{w}x{h}"] += 1
                bytes_by_label[lab].append(len(b))
                records.append((f.name, row, lab, w, h, len(b)))
    finally:
        pool.close()
        pool.join()

    kept = pd.DataFrame(records, columns=["file", "row", "label", "width", "height", "nbytes"])
    kept.to_parquet(CLEAN_DIR / "clean_index.parquet", index=False)

    def bstats(vals):
        a = np.asarray(vals, dtype=np.float64)
        return {"n": int(a.size), "mean": float(a.mean()) if a.size else None,
                "median": float(np.median(a)) if a.size else None}

    stats = {
        "n_total_rows": n_total,
        "n_kept": int(len(kept)),
        "n_dropped_bad": n_bad,
        "n_dropped_duplicate": n_dup,
        "class_counts_binary": {"real_0": class_counts.get(0, 0),
                                 "ai_1to5": class_counts.get(1, 0)},
        "format_counts": dict(fmt_counts),
        "square_by_label": {("real" if k == 0 else "ai"): dict(v)
                             for k, v in sq_by_label.items()},
        "top_sizes_real": size_by_label[0].most_common(8),
        "top_sizes_ai": size_by_label[1].most_common(8),
        "bytes_real": bstats(bytes_by_label[0]),
        "bytes_ai": bstats(bytes_by_label[1]),
        "cleaning_actions": [
            "drop undecodable images",
            "drop exact-duplicate images (md5 of encoded bytes, keep first)",
            "downstream: resize every image to a fixed square to remove the "
            "train-only aspect-ratio shortcut and match the 320x320 eval splits",
        ],
    }
    (CLEAN_DIR / "clean_stats.json").write_text(json.dumps(stats, indent=2))

    print("clean.py summary")
    print(f"  total train rows      : {n_total}")
    print(f"  dropped (undecodable) : {n_bad}")
    print(f"  dropped (duplicates)  : {n_dup}")
    print(f"  kept                  : {len(kept)}")
    print(f"  class balance (kept)  : real={int((kept.label==0).sum())} "
          f"ai={int((kept.label==1).sum())}")
    print(f"  real square/non-square: {dict(sq_by_label[0])}")
    print(f"  ai   square/non-square: {dict(sq_by_label[1])}")
    print(f"  wrote {CLEAN_DIR/'clean_index.parquet'} and clean_stats.json")


if __name__ == "__main__":
    main()
