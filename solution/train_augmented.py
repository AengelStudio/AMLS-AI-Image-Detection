"""Task 1.3: train robust / augmented model (stub)."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TASK03_DIR = ROOT / "artifacts" / "task03"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 3 augmented model.")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    parser.parse_args()

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Expected read-only data at {DATA_DIR}")

    for split in ("calibration_augmented", "validation_augmented"):
        split_dir = DATA_DIR / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected split at {split_dir}")

    TASK03_DIR.mkdir(parents=True, exist_ok=True)
    print("train_augmented.py: placeholder — implement robust training under artifacts/task03/.")


if __name__ == "__main__":
    main()
