"""Generate Task 1.1 report figures from the cleaning/exploration stats."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
clean = json.loads((ROOT / "solution" / "artifacts" / "clean" / "clean_stats.json").read_text())
expl = json.loads((OUT / "explore_summary.json").read_text())

# --- class distribution by source generator (train) ---
cc = expl["train"]["class_counts_source"]
labels = {0: "real", 1: "SD2.1", 2: "SDXL", 3: "SD3", 4: "DALL-E3", 5: "MJ"}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
keys = sorted(int(k) for k in cc)
vals = [cc[str(k)] if str(k) in cc else cc[k] for k in keys]
colors = ["#2b7" if k == 0 else "#c44" for k in keys]
ax[0].bar([labels[k] for k in keys], vals, color=colors)
ax[0].set_title("Train: images per source class")
ax[0].set_ylabel("count")
ax[0].tick_params(axis="x", rotation=20)

# --- aspect ratio (square vs non-square) by class, train ---
sq = clean["square_by_label"]
real = sq.get("real", {}); ai = sq.get("ai", {})
cats = ["real", "ai"]
nonsq = [real.get("non_square", 0), ai.get("non_square", 0)]
square = [real.get("square", 0), ai.get("square", 0)]
ax[1].bar(cats, nonsq, label="non-square", color="#2b7")
ax[1].bar(cats, square, bottom=nonsq, label="square", color="#c44")
ax[1].set_title("Train: aspect ratio leaks the label")
ax[1].set_ylabel("count"); ax[1].legend()
plt.tight_layout()
plt.savefig(OUT / "fig_class_aspect.png", dpi=120)
plt.close()

# --- byte size: clean vs augmented (compression tell) ---
fig, ax = plt.subplots(figsize=(5, 3.2))
splits = ["calibration", "calibration_augmented", "validation", "validation_augmented"]
means = [expl[s]["bytes_mean"] / 1024 for s in splits]
ax.bar(range(len(splits)), means,
       color=["#37a", "#7ad", "#37a", "#7ad"])
ax.set_xticks(range(len(splits)))
ax.set_xticklabels(["cal", "cal_aug", "val", "val_aug"], rotation=15)
ax.set_ylabel("mean encoded size (KB)")
ax.set_title("Augmentation = heavier compression")
plt.tight_layout()
plt.savefig(OUT / "fig_bytes.png", dpi=120)
plt.close()
print("wrote fig_class_aspect.png, fig_bytes.png")
