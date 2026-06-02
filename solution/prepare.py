"""Task 1.2: prepare cleaned data for modeling (stub; skips predict/)."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare features from cleaned training data.")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    parser.parse_args()

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Expected read-only data at {DATA_DIR}")

    for split in ("train", "calibration", "validation"):
        split_dir = DATA_DIR / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected split at {split_dir}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("prepare.py: ready (predict/ intentionally not prepared).")


if __name__ == "__main__":
    main()
