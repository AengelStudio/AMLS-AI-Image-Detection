"""Task 1.4 — explainability for the trained detector.

Produces, for the final SmallCNN model:
  * vanilla gradient saliency maps,
  * Grad-CAM heatmaps from the last convolutional block,
  * occlusion-sensitivity maps,
for representative true-positive (AI), true-negative (real), false-positive and
false-negative validation images, plus a quantitative occlusion comparison of
where the model "looks" for real vs AI images.

Run after training. Default model: artifacts/task02/model.pt (override with --ckpt).
This file is also suitable to submit as the Task 1.4 code at the zip root.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# locate the solution package robustly (works from analysis/ or the zip root)
HERE = Path(__file__).resolve().parent
for _cand in (HERE / "solution", HERE.parent / "solution", HERE):
    if (_cand / "amls_common.py").exists():
        SOLUTION = _cand
        break
else:
    SOLUTION = HERE.parent / "solution"
sys.path.insert(0, str(SOLUTION))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amls_common import load_prepared, seed_everything, setup_threads
from amls_model import load_checkpoint, to_tensor, predict_scores

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def saliency_map(model, x_t):
    x_t = x_t.clone().requires_grad_(True)
    p_ai = F.softmax(model(x_t), dim=1)[:, 1].sum()
    grad = torch.autograd.grad(p_ai, x_t)[0]
    s = grad.abs().amax(1)[0].numpy()          # max over channels
    return (s - s.min()) / (np.ptp(s) + 1e-8)


def grad_cam(model, x_t):
    # hook the last conv activation (out-of-place) and differentiate the AI logit
    # w.r.t. it via autograd.grad — avoids backward hooks, which clash with the
    # network's in-place ReLUs.
    acts = {}
    target = model.features[-4]                 # final Conv2d before BN+ReLU+GAP
    h = target.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    logit_ai = model(x_t)[0, 1]
    grad = torch.autograd.grad(logit_ai, acts["a"])[0]
    h.remove()
    a = acts["a"][0].detach()                   # C,h,w
    g = grad[0].mean((1, 2), keepdim=True)      # C,1,1 channel weights
    cam = F.relu((g * a).sum(0)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return np.asarray(Image_resize(cam, x_t.shape[2]))


def Image_resize(arr, size):
    from PIL import Image
    return Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)


def occlusion_map(model, x_t, mean, std, patch=16, stride=8):
    base = float(F.softmax(model(x_t), dim=1)[0, 1])
    H = x_t.shape[2]
    heat = np.zeros((H, H), np.float32)
    cnt = np.zeros((H, H), np.float32)
    grayval = ((0.5 - mean) / std).astype(np.float32)  # neutral gray in normalised space
    for y in range(0, H - patch + 1, stride):
        for x in range(0, H - patch + 1, stride):
            xx = x_t.clone()
            for c in range(3):
                xx[0, c, y:y + patch, x:x + patch] = float(grayval[c])
            with torch.no_grad():
                p = float(F.softmax(model(xx), dim=1)[0, 1])
            heat[y:y + patch, x:x + patch] += (base - p)  # drop in P(ai) => importance
            cnt[y:y + patch, x:x + patch] += 1
    heat /= np.maximum(cnt, 1)
    return base, heat


def denorm(x_t, mean, std):
    img = x_t[0].permute(1, 2, 0).numpy() * std + mean
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(SOLUTION / "artifacts" / "task02" / "model.pt"))
    ap.add_argument("--n_examples", type=int, default=3)
    args = ap.parse_args()
    seed_everything(); setup_threads()

    model, mean, std, thr, meta = load_checkpoint(Path(args.ckpt))
    vx, vy = load_prepared("validation")
    scores = predict_scores(model, vx, mean, std)
    pred = (scores >= thr).astype(int)

    tp = np.where((pred == 1) & (vy == 1))[0]   # AI correctly flagged
    tn = np.where((pred == 0) & (vy == 0))[0]   # real correctly passed
    fp = np.where((pred == 1) & (vy == 0))[0]   # real wrongly flagged
    fn = np.where((pred == 0) & (vy == 1))[0]   # AI missed
    cats = [("TP_ai", tp), ("TN_real", tn), ("FP_real", fp), ("FN_ai", fn)]
    print("counts:", {n: len(ix) for n, ix in cats})

    rng = np.random.default_rng(0)
    rows = []
    for name, ix in cats:
        if len(ix):
            rows.append((name, int(ix[np.argsort(np.abs(scores[ix] - thr))[0]])))  # most borderline

    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(11, 2.7 * n))
    if n == 1:
        ax = ax[None, :]
    for r, (name, i) in enumerate(rows):
        x_t = to_tensor(vx[i:i + 1], mean, std)
        img = denorm(x_t, mean, std)
        sal = saliency_map(model, x_t)
        cam = grad_cam(model, x_t.clone())
        base, occ = occlusion_map(model, x_t, mean, std)
        ax[r, 0].imshow(img); ax[r, 0].set_ylabel(f"{name}\nP(ai)={scores[i]:.2f}", fontsize=9)
        ax[r, 1].imshow(img); ax[r, 1].imshow(sal, cmap="hot", alpha=0.55)
        ax[r, 2].imshow(img); ax[r, 2].imshow(cam, cmap="jet", alpha=0.5)
        ax[r, 3].imshow(occ, cmap="bwr", vmin=-abs(occ).max(), vmax=abs(occ).max())
        for c, t in enumerate(["input", "saliency", "Grad-CAM", "occlusion ΔP(ai)"]):
            ax[r, c].set_title(t if r == 0 else "", fontsize=10)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT / "explain_examples.png", dpi=120)
    plt.close()
    print(f"wrote {OUT/'explain_examples.png'}")

    # ---- quantitative: average occlusion importance, real vs AI ----
    def avg_occ(indices, m=24):
        sel = indices[rng.choice(len(indices), min(m, len(indices)), replace=False)]
        acc = None
        for i in sel:
            _, occ = occlusion_map(model, to_tensor(vx[i:i + 1], mean, std), mean, std,
                                   patch=24, stride=24)
            acc = occ if acc is None else acc + occ
        return acc / len(sel)

    occ_real = avg_occ(np.where(vy == 0)[0])
    occ_ai = avg_occ(np.where(vy == 1)[0])
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.4))
    mx = max(abs(occ_real).max(), abs(occ_ai).max())
    for a, occ, t in [(ax[0], occ_real, "real images"), (ax[1], occ_ai, "AI images")]:
        im = a.imshow(occ, cmap="bwr", vmin=-mx, vmax=mx); a.set_title(t); a.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="mean ΔP(ai) when occluded")
    plt.savefig(OUT / "explain_occlusion_avg.png", dpi=120, bbox_inches="tight")
    print(f"wrote {OUT/'explain_occlusion_avg.png'}")

    # centre vs border importance ratio (shortcut probe)
    def center_border_ratio(occ):
        H = occ.shape[0]; b = H // 4
        center = occ[b:H - b, b:H - b].mean()
        border = (occ.sum() - occ[b:H - b, b:H - b].sum()) / (occ.size - (H - 2 * b) ** 2)
        return float(center), float(border)
    import json
    summary = {
        "n": {n: int(len(ix)) for n, ix in cats},
        "occlusion_center_border_real": center_border_ratio(occ_real),
        "occlusion_center_border_ai": center_border_ratio(occ_ai),
    }
    (OUT / "explain_summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", summary)


if __name__ == "__main__":
    main()
