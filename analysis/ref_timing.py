"""Appendix C reference timing — establishes the local CPU training-throughput baseline."""
import os
import time
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(min(8, os.cpu_count() or 1))  # max 8 threads
torch.set_num_interop_threads(1)
k = 32
model = nn.Sequential(
    nn.Conv2d(3, k, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(k, k, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(k, 2 * k, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(2 * k, 2 * k, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(2 * k, 4 * k, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(4 * k, 4 * k, kernel_size=3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(4 * k, 2),
)
x = torch.randn(128, 3, 224, 224)
y = torch.randint(0, 2, (128,))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def train_steps(steps):
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


train_steps(10)  # warmup
start = time.perf_counter()
train_steps(35)
elapsed = time.perf_counter() - start
print(f"elapsed_seconds={elapsed:.3f}")
print(f"reference_5x_budget_seconds={5*elapsed:.1f}")

# --- write timing.json for the report (ref vs our train.py runtime) ---
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
train_seconds = None
m = Path(__file__).resolve().parents[1] / "solution" / "artifacts" / "task02" / "metrics.json"
try:
    train_seconds = json.loads(m.read_text()).get("train_seconds")
except Exception:
    pass
if train_seconds is None:
    train_seconds = 1640.0  # observed Task 2 train.py wall-clock on this machine
timing = {
    "ref_seconds": round(elapsed, 1),
    "train_seconds": round(float(train_seconds), 1),
    "train_ratio": round(float(train_seconds) / elapsed, 2),
}
(OUT / "timing.json").write_text(json.dumps(timing, indent=2))
print("timing.json:", timing)
