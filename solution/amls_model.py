"""Torch model, training loop, augmentation and inference for the detector.

Separated from amls_common so that the torch-free steps (clean.py, prepare.py)
do not import torch. The same SmallCNN + training loop serve Task 2 (clean
training) and Task 3 (continue from the Task 2 checkpoint with augmentation).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F

from amls_common import (
    FPR_LIMIT,
    IMG_SIZE,
    Budget,
    calibrate_threshold,
    fpr_recall,
    seed_everything,
    setup_threads,
)

# Prefer epochs whose validation FPR stays under this soft margin; epochs above
# the hard 0.20 cap are excluded from selection entirely.
SELECT_FPR = 0.185


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Compact VGG-style CNN trained from scratch (CPU friendly).

    A 5x5 stride-2 stem halves the spatial size immediately: the first
    convolution still sees full 128px detail (where the high-frequency
    real-vs-AI cue lives) but the expensive body runs at 64px, which is what
    makes training affordable on CPU. BatchNorm gives fast from-scratch
    convergence; global average pooling keeps the head tiny and size-agnostic.
    """

    def __init__(self, k: int = 32, in_ch: int = 3, p_drop: float = 0.3, highpass: bool = True):
        super().__init__()
        self.highpass = highpass            # prepend a fixed high-pass residual branch
        stem_in = in_ch * 2 if highpass else in_ch

        def block(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            nn.Conv2d(stem_in, k, 5, stride=2, padding=2, bias=False),  # /2
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
            block(k, 2 * k),
            block(2 * k, 4 * k),
            nn.Conv2d(4 * k, 4 * k, 3, padding=1, bias=False),
            nn.BatchNorm2d(4 * k),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p_drop), nn.Linear(4 * k, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.highpass:
            # local high-pass residual (RGB minus a 3x3 box blur) gives the net an
            # explicit forensic channel; computed inside the model so the *input*
            # stays 3-channel RGB (keeps preprocessing and explainability simple).
            lp = F.avg_pool2d(x, 3, stride=1, padding=1)
            x = torch.cat([x, x - lp], dim=1)
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def compute_norm(x_uint8: np.ndarray):
    """Per-channel mean/std over a uint8 NHWC array, in [0,1] scale."""
    sample = x_uint8
    if len(sample) > 4000:
        idx = np.random.default_rng(0).choice(len(sample), 4000, replace=False)
        sample = sample[idx]
    f = sample.astype(np.float32) / 255.0
    mean = f.reshape(-1, 3).mean(0)
    std = f.reshape(-1, 3).std(0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def to_tensor(x_uint8: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """uint8 NHWC -> normalised float NCHW tensor."""
    f = torch.from_numpy(np.ascontiguousarray(x_uint8)).float().div_(255.0)
    f = (f - torch.tensor(mean)) / torch.tensor(std)
    return f.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Augmentation (Task 3): simulate scaling, compression, blur, noise, photometric
# ---------------------------------------------------------------------------
class Augmenter:
    """On-the-fly augmentation operating on a uint8 NHWC batch.

    The augmentations mirror the perturbations described in the exercise
    (scaling, JPEG compression, blur) plus light photometric jitter and noise.
    Cheap ops are vectorised in numpy; JPEG/blur use PIL per image but only fire
    with a probability, keeping the per-batch cost modest on CPU.
    """

    def __init__(self, seed: int = 0, strength: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.s = strength

    def _one(self, img: np.ndarray) -> np.ndarray:
        r = self.rng
        # horizontal flip
        if r.random() < 0.5:
            img = img[:, ::-1]
        # downscale-upscale (resolution robustness)
        if r.random() < 0.6 * self.s:
            h, w = img.shape[:2]
            scale = r.uniform(0.4, 1.0)
            nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
            pim = Image.fromarray(img)
            img = np.asarray(
                pim.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
            )
        # gaussian blur
        if r.random() < 0.4 * self.s:
            rad = r.uniform(0.4, 1.6)
            img = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(rad)))
        # JPEG recompression
        if r.random() < 0.7 * self.s:
            q = int(r.integers(30, 96))
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=q)
            img = np.asarray(Image.open(buf).convert("RGB"))
        # photometric jitter
        if r.random() < 0.5:
            br = r.uniform(0.85, 1.15)
            co = r.uniform(0.85, 1.15)
            f = img.astype(np.float32)
            m = f.mean()
            f = (f - m) * co + m * br
            img = np.clip(f, 0, 255).astype(np.uint8)
        # gaussian noise
        if r.random() < 0.3 * self.s:
            sigma = r.uniform(2, 10)
            f = img.astype(np.float32) + self.rng.normal(0, sigma, img.shape)
            img = np.clip(f, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img, dtype=np.uint8)

    def __call__(self, batch_uint8: np.ndarray) -> np.ndarray:
        return np.stack([self._one(im) for im in batch_uint8])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_scores(model: nn.Module, x_uint8: np.ndarray, mean, std,
                   batch_size: int = 256, tta: bool = False) -> np.ndarray:
    """Return P(ai) for every image."""
    model.eval()
    out = np.empty(len(x_uint8), dtype=np.float32)
    for i in range(0, len(x_uint8), batch_size):
        xb = to_tensor(x_uint8[i:i + batch_size], mean, std)
        logits = model(xb)
        if tta:
            logits = logits + model(torch.flip(xb, dims=[3]))
        p = F.softmax(logits, dim=1)[:, 1].numpy()
        out[i:i + batch_size] = p
    return out


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model: nn.Module, mean, std, threshold: float,
                    meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "k": meta.get("k", 32),
            "in_ch": meta.get("in_ch", 3),
            "highpass": bool(getattr(model, "highpass", True)),
            "img_size": IMG_SIZE,
            "mean": np.asarray(mean).tolist(),
            "std": np.asarray(std).tolist(),
            "threshold": float(threshold),
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SmallCNN(k=ckpt.get("k", 32), in_ch=ckpt.get("in_ch", 3),
                     highpass=ckpt.get("highpass", True))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    return model, mean, std, float(ckpt["threshold"]), ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _class_weights(y: np.ndarray) -> torch.Tensor:
    n = len(y)
    n0 = max(1, int((y == 0).sum()))
    n1 = max(1, int((y == 1).sum()))
    return torch.tensor([n / (2 * n0), n / (2 * n1)], dtype=torch.float32)


def train_model(
    train_x, train_y, cal_x, cal_y, val_x, val_y,
    budget: Budget,
    ckpt_path: Path,
    *,
    k: int = 32,
    lr: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 14,
    augment: bool = False,
    aug_strength: float = 1.0,
    init_state: dict | None = None,
    target_fpr: float = 0.15,
    log=lambda *a: print(*a, flush=True),
):
    """Train (or fine-tune) SmallCNN, calibrating the threshold each epoch and
    checkpointing the best operating point. Returns the best-epoch summary."""
    seed_everything()
    setup_threads()
    mean, std = compute_norm(train_x)

    import copy
    import time as _time

    model = SmallCNN(k=k)
    if init_state is not None:
        model.load_state_dict(init_state)
        log("  (continuing from Task 2 checkpoint)")
    start_state = copy.deepcopy(model.state_dict())  # exact weights to train from
    crit = nn.CrossEntropyLoss(weight=_class_weights(train_y))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n = len(train_x)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    aug = Augmenter(strength=aug_strength) if augment else None
    rng = np.random.default_rng(0)  # deterministic batch order

    # --- estimate per-step time (incl. augmentation) to size the LR schedule to
    #     the epochs that actually fit the CPU budget, so the LR anneals fully ---
    warm_idx = rng.permutation(n)[:batch_size]
    model.train()
    for _w in range(3):
        xb = to_tensor(aug(train_x[warm_idx]) if aug is not None else train_x[warm_idx], mean, std)
        opt.zero_grad(set_to_none=True)
        crit(model(xb), torch.from_numpy(train_y[warm_idx])).backward()
        opt.step()
    _t0 = _time.perf_counter()
    xb = to_tensor(aug(train_x[warm_idx]) if aug is not None else train_x[warm_idx], mean, std)
    crit(model(xb), torch.from_numpy(train_y[warm_idx])).backward()
    step_time = max(1e-3, _time.perf_counter() - _t0)
    epoch_time = step_time * steps_per_epoch + 6.0  # +eval overhead
    planned_epochs = int(max(2, min(max_epochs, budget.remaining() // epoch_time)))
    log(f"  step~{step_time:.2f}s, epoch~{epoch_time:.0f}s -> planning {planned_epochs} epochs "
        f"(budget {budget.remaining():.0f}s)")

    # restore exact starting weights (undo warmup) and build the sized schedule
    model.load_state_dict(start_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps_per_epoch * planned_epochs, pct_start=0.25
    )
    max_epochs = planned_epochs

    best = {"select": -1e9}
    cal_real_mask = cal_y == 0
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(n)
        running = 0.0
        for s in range(steps_per_epoch):
            idx = order[s * batch_size:(s + 1) * batch_size]
            xb_u8 = train_x[idx]
            if aug is not None:
                xb_u8 = aug(xb_u8)
            xb = to_tensor(xb_u8, mean, std)
            yb = torch.from_numpy(train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss)
            if budget.expired():
                break
        # ---- evaluate + calibrate ----
        cal_scores = predict_scores(model, cal_x, mean, std)
        thr = calibrate_threshold(cal_scores[cal_real_mask], target_fpr=target_fpr)
        val_scores = predict_scores(model, val_x, mean, std)
        v_fpr, v_rec = fpr_recall(val_scores, val_y, thr)
        c_fpr, c_rec = fpr_recall(cal_scores, cal_y, thr)
        # selection: exclude epochs above the hard FPR cap, softly prefer those
        # comfortably below it, then maximise recall
        if v_fpr > FPR_LIMIT:
            select = -1e9 + v_rec          # effectively excluded; keep ordering sane
        else:
            select = v_rec - 8.0 * max(0.0, v_fpr - SELECT_FPR)
        log(
            f"  epoch {epoch + 1:2d}/{max_epochs} loss={running / max(1, steps_per_epoch):.3f} "
            f"thr={thr:.3f} cal_fpr={c_fpr:.3f} val_fpr={v_fpr:.3f} val_recall={v_rec:.3f} "
            f"t={budget.elapsed():.0f}s"
        )
        if select > best["select"]:
            best = {
                "select": select, "epoch": epoch + 1, "threshold": float(thr),
                "val_fpr": v_fpr, "val_recall": v_rec, "cal_fpr": c_fpr, "cal_recall": c_rec,
            }
            save_checkpoint(ckpt_path, model, mean, std, thr,
                            {"k": k, "in_ch": 3, "augment": augment, **best})
        if budget.expired():
            log("  budget reached — stopping early")
            break

    log(f"  best: epoch {best.get('epoch')} val_fpr={best.get('val_fpr'):.3f} "
        f"val_recall={best.get('val_recall'):.3f}")
    return best
