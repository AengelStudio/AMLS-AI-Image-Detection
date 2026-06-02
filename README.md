# AMLS SoSe 2026 — AI Image Detection

## What goes where

| Location | Purpose |
|----------|---------|
| **`solution/`** | Submission: six scripts + `Dockerfile` + `requirements.txt` only |
| **`run_pipeline.sh`** | Local helper (repo root) — not submitted |
| **`report.pdf`** | Zip root at submission time |
| **`README.md`**, **`AMLS_2026_Exercise.pdf`** | Local reference — not submitted |
| **`data/data/`** | Local dataset (gitignored) |

## `solution/` (matches exercise PDF)

```
solution/
├── Dockerfile
├── requirements.txt
├── clean.py
├── prepare.py
├── train.py
├── predict.py
├── train_augmented.py
└── predict_augmented.py
```

Runtime only (do not submit): `solution/data/`, `solution/artifacts/`.

## Run full pipeline locally

Requires Docker Desktop running. Dataset under `data/data/` (e.g. `data/data/train/`).

**Windows (double-click or cmd):**

```bat
run_pipeline.bat
```

**Git Bash / WSL:**

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Or run a single step:

```bash
docker build -t amls-ai-image-detection:latest solution
docker run --rm --cpus=8 \
  -v "$(pwd)/data/data:/workspace/solution/data:ro" \
  -v "$(pwd)/solution/artifacts:/workspace/solution/artifacts" \
  -w /workspace/solution \
  amls-ai-image-detection:latest \
  python clean.py --timeout_seconds 600
```

## Submission zip

`AMLS Exercise <student ID>.zip` (max 20 MB):

```
report.pdf
solution/          # no data/, no artifacts/*, no Docker image
<optional task 1.4 files at zip root>
```
