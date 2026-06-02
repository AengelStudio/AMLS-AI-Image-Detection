"""Task 1.1: dataset exploration and cleaning (stub)."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore and clean training data.")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    parser.parse_args()

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Expected read-only data at {DATA_DIR}")

    train_dir = DATA_DIR / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected training split at {train_dir}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"clean.py: found {len(list(train_dir.glob('*.parquet')))} train parquet file(s).")


if __name__ == "__main__":
    main()
