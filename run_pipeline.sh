#!/usr/bin/env bash
# Local helper — not part of the exercise submission zip.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${AMLS_IMAGE:-amls-ai-image-detection:latest}"

docker build -t "$IMAGE" "$ROOT/solution"

run_step() {
  echo ">>> $*"
  docker run --rm \
    --cpus=8 \
    -v "$ROOT/data/data:/workspace/solution/data:ro" \
    -v "$ROOT/solution/artifacts:/workspace/solution/artifacts" \
    -w /workspace/solution \
    "$IMAGE" \
    "$@"
}

run_step python clean.py --timeout_seconds 600
run_step python prepare.py --timeout_seconds 600
run_step python train.py --timeout_seconds 1800
run_step python predict.py --timeout_seconds 600
run_step python train_augmented.py --timeout_seconds 1800
run_step python predict_augmented.py --timeout_seconds 600

echo "Done. Predictions:"
echo "  $ROOT/solution/artifacts/task02/predictions.csv"
echo "  $ROOT/solution/artifacts/task03/predictions.csv"
