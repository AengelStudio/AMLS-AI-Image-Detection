"""Task 1.2: train binary classifier (stub)."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
TASK02_DIR = ARTIFACTS_DIR / "task02"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 2 model.")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    parser.parse_args()

    TASK02_DIR.mkdir(parents=True, exist_ok=True)
    print("train.py: placeholder — implement training and checkpointing under artifacts/task02/.")


if __name__ == "__main__":
    main()
