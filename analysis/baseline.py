"""Task 1.2 comparison — classical model family on engineered features.

The second required model family: hand-crafted forensic features (radial FFT
power spectrum, colour statistics, high-pass residual statistics) fed to a
gradient-boosted tree. Trains in seconds on CPU and exploits the high-frequency
real-vs-AI cue directly. Same protocol as the CNN: threshold calibrated on the
calibration split, evaluated on validation and validation_augmented.

Not submitted — produces report numbers + a feature-importance figure.
"""

import io
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "data"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
FEAT_RES = 224
N_BINS = 32
RNG = np.random.default_rng(0)
TARGET_FPR = 0.15


def radial_spectrum(gray: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray))
    power = np.log1p(np.abs(f) ** 2)
    h, w = power.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    r /= r.max()
    idx = np.clip((r.ravel() * n_bins).astype(int), 0, n_bins - 1)
    out = np.bincount(idx, weights=power.ravel(), minlength=n_bins)
    cnt = np.bincount(idx, minlength=n_bins)
    return out / np.maximum(cnt, 1)


def features_from_bytes(b: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(b)).convert("RGB").resize((FEAT_RES, FEAT_RES), Image.BILINEAR)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    spec = radial_spectrum(gray)                       # 32
    hf = spec[N_BINS // 2:].mean() - spec[:N_BINS // 2].mean()  # 1 high/low contrast

    mean_rgb = rgb.reshape(-1, 3).mean(0)              # 3
    std_rgb = rgb.reshape(-1, 3).std(0)               # 3
    mx = rgb.max(2); mn = rgb.min(2)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)
    sat_stats = np.array([sat.mean(), sat.std()])      # 2

    blur = ndimage.gaussian_filter(gray, 1.0)
    resid = gray - blur
    res_stats = np.array([resid.std(), np.abs(resid).mean(),
                          float(stats.kurtosis(resid.ravel()))])  # 3
    chan_res = np.array([(rgb[..., c] - ndimage.gaussian_filter(rgb[..., c], 1.0)).std()
                         for c in range(3)])           # 3
    lap_var = float(ndimage.laplace(gray).var())       # 1

    return np.concatenate([spec, [hf], mean_rgb, std_rgb, sat_stats,
                           res_stats, chan_res, [lap_var]]).astype(np.float32)


FEATURE_NAMES = (
    [f"spec_{i}" for i in range(N_BINS)] + ["hf_minus_lf"]
    + ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B", "sat_mean", "sat_std"]
    + ["resid_std", "resid_absmean", "resid_kurt", "cres_R", "cres_G", "cres_B", "lap_var"]
)


def load_split(split: str, cap=None):
    files = sorted((DATA / split).glob("*.parquet"))
    raws, labels = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["image", "source_class"])
        raws.extend(bytes(v) for v in df["image"])
        labels.extend(0 if int(c) == 0 else 1 for c in df["source_class"])
    labels = np.array(labels)
    if cap and len(raws) > cap:
        idx = RNG.choice(len(raws), cap, replace=False)
        raws = [raws[i] for i in idx]
        labels = labels[idx]
    with mp.Pool(min(8, mp.cpu_count() or 1)) as pool:
        feats = pool.map(features_from_bytes, raws, chunksize=32)
    return np.stack(feats), labels


def calib_threshold(scores_real, target=TARGET_FPR):
    return float(np.quantile(scores_real, 1 - target, method="higher"))


def fpr_recall(scores, y, thr):
    pred = (scores >= thr).astype(int)
    return float(pred[y == 0].mean()), float(pred[y == 1].mean())


def main():
    print("baseline: extracting features ...", flush=True)
    Xtr, ytr = load_split("train", cap=12000)
    Xcal, ycal = load_split("calibration")
    Xval, yval = load_split("validation")
    Xva, yva = load_split("validation_augmented")
    print(f"  train={Xtr.shape} cal={Xcal.shape} val={Xval.shape} val_aug={Xva.shape}")

    sw = np.where(ytr == 0, (ytr == 1).sum() / max(1, (ytr == 0).sum()), 1.0)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         random_state=0)
    clf.fit(Xtr, ytr, sample_weight=sw)

    def scores(X):
        return clf.predict_proba(X)[:, 1]

    thr = calib_threshold(scores(Xcal)[ycal == 0])
    res = {}
    for name, X, y in [("validation", Xval, yval), ("validation_augmented", Xva, yva),
                       ("calibration", Xcal, ycal)]:
        fpr, rec = fpr_recall(scores(X), y, thr)
        res[name] = {"fpr_real": fpr, "recall_ai": rec}
        print(f"  {name:22s} fpr_real={fpr:.3f} recall_ai={rec:.3f}")
    res["threshold"] = thr

    # permutation-free importance proxy: use sklearn permutation importance on val
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(clf, Xval, yval, n_repeats=5, random_state=0, n_jobs=4)
    order = np.argsort(pi.importances_mean)[::-1][:12]
    res["top_features"] = [(FEATURE_NAMES[i], float(pi.importances_mean[i])) for i in order]
    print("  top features:", res["top_features"][:6])

    (OUT / "baseline_metrics.json").write_text(json.dumps(res, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [FEATURE_NAMES[i] for i in order][::-1]
    vals = [pi.importances_mean[i] for i in order][::-1]
    plt.figure(figsize=(6, 4))
    plt.barh(names, vals)
    plt.xlabel("permutation importance (val accuracy drop)")
    plt.title("Classical baseline — top features")
    plt.tight_layout()
    plt.savefig(OUT / "baseline_importance.png", dpi=110)
    print(f"  wrote {OUT/'baseline_metrics.json'} and baseline_importance.png")


if __name__ == "__main__":
    main()
