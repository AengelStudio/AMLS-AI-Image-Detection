#!/usr/bin/env bash
# Assemble the submission zip:  AMLS Exercise <student ID>.zip
#   report.pdf
#   explain.py            (Task 1.4 explainability code, runnable at zip root)
#   solution/             (code + Dockerfile + requirements; NO data/, NO artifacts payload)
# Usage: ./make_submission.sh <student_id>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SID="${1:-STUDENTID}"
STAGE="$(mktemp -d)"
OUT="$ROOT/AMLS Exercise ${SID}.zip"

mkdir -p "$STAGE/solution/artifacts/task02" "$STAGE/solution/artifacts/task03"

# --- solution code only ---
for f in Dockerfile requirements.txt amls_common.py amls_model.py \
         clean.py prepare.py train.py predict.py train_augmented.py predict_augmented.py \
         .dockerignore; do
  [ -e "$ROOT/solution/$f" ] && cp "$ROOT/solution/$f" "$STAGE/solution/$f"
done
# keep the artifacts/ folders present but empty (runtime write targets)
touch "$STAGE/solution/artifacts/task02/.gitkeep" "$STAGE/solution/artifacts/task03/.gitkeep"

# --- report + task 1.4 code at zip root ---
[ -e "$ROOT/report/report.pdf" ] && cp "$ROOT/report/report.pdf" "$STAGE/report.pdf"
cp "$ROOT/analysis/explain.py" "$STAGE/explain.py"

# --- zip ---
rm -f "$OUT"
( cd "$STAGE" && zip -r -q "$OUT" . -x '*__pycache__*' -x '*.DS_Store' )
rm -rf "$STAGE"

echo "wrote: $OUT"
du -h "$OUT" | cut -f1 | xargs echo "size:"
echo "contents:"; unzip -l "$OUT"
