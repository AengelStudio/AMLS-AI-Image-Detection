"""Task 1.1 — dataset exploration (report figures + stats).

Not part of the submission. Produces JSON stats and PNG figures under analysis/out/
that feed the report. Reads the read-only dataset under data/data/.

Examines, per split:
  * class distribution (source_class histogram + merged binary real/ai),
  * image format (from the encoded byte header), size, aspect ratio, mode,
  * encoded bytes per image,
  * colour statistics and the azimuthally-averaged FFT power spectrum,
all of which are candidate "tells" that could leak the label.
"""

import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "data"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

LABELED = ["train", "calibration", "calibration_augmented", "validation", "validation_augmented"]
SAMPLE_DECODE = 4000     # per split: decode header (format/size/mode/bytes)
SAMPLE_PIXEL = 1200      # per split: full pixel + spectral stats
RNG = np.random.default_rng(0)


def magic_format(b: bytes) -> str:
    if b[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "WEBP"
    if b[:2] in (b"BM",):
        return "BMP"
    if b[:4] in (b"GIF8",):
        return "GIF"
    return "OTHER"


def radial_power_spectrum(gray: np.ndarray, n_bins: int = 48) -> np.ndarray:
    """Azimuthally-averaged log power spectrum of a square grayscale image."""
    f = np.fft.fftshift(np.fft.fft2(gray))
    power = np.log1p(np.abs(f) ** 2)
    h, w = power.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / r.max()
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(r_norm.ravel(), bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    out = np.zeros(n_bins)
    flat = power.ravel()
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out[b] = flat[m].mean()
    return out


def col_image(df: pd.DataFrame) -> pd.Series:
    return df["image"]


def get_bytes(v) -> bytes:
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, dict) and "bytes" in v:      # HF-style {bytes,path}
        return bytes(v["bytes"])
    if isinstance(v, np.ndarray):
        return v.tobytes()
    return bytes(v)


def explore_split(name: str) -> dict:
    files = sorted((DATA / name).glob("*.parquet"))
    cols_probe = pd.read_parquet(files[0]).head(1)
    has_label = "source_class" in cols_probe.columns
    label_col = "source_class" if has_label else None

    # ---- full class distribution (cheap: label column only) ----
    class_counts = Counter()
    n_rows = 0
    for f in files:
        if has_label:
            s = pd.read_parquet(f, columns=[label_col])[label_col]
            class_counts.update(s.tolist())
            n_rows += len(s)
        else:
            n_rows += len(pd.read_parquet(f, columns=["row_id"]))

    # ---- sampled per-image stats ----
    rec_fmt, rec_w, rec_h, rec_mode, rec_bytes, rec_label = [], [], [], [], [], []
    spec_real, spec_ai = [], []
    col_real, col_ai = [], []   # (mean_rgb, sat_mean) rows
    decoded = 0
    pixel_done = 0
    for f in files:
        cols = ["image"] + ([label_col] if has_label else [])
        df = pd.read_parquet(f, columns=cols)
        take = min(SAMPLE_DECODE // len(files) + 1, len(df))
        sel = RNG.choice(len(df), size=take, replace=False)
        for i in sel:
            b = get_bytes(df["image"].iloc[int(i)])
            lab = int(df[label_col].iloc[int(i)]) if has_label else -1
            ybin = 0 if lab == 0 else 1
            try:
                im = Image.open(io.BytesIO(b))
                w, h = im.size
                mode = im.mode
                fmt = im.format or magic_format(b)
            except Exception:
                fmt = magic_format(b)
                w = h = -1
                mode = "?"
            rec_fmt.append(fmt); rec_w.append(w); rec_h.append(h)
            rec_mode.append(mode); rec_bytes.append(len(b)); rec_label.append(ybin)
            decoded += 1
            # pixel + spectral on a smaller subset
            if pixel_done < SAMPLE_PIXEL and w > 0:
                try:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                    mean_rgb = rgb.reshape(-1, 3).mean(0)
                    mx = rgb.max(2); mn = rgb.min(2)
                    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0).mean()
                    if has_label:
                        (col_real if ybin == 0 else col_ai).append(
                            [*mean_rgb.tolist(), float(sat)])
                    g = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
                    rps = radial_power_spectrum(g)
                    if has_label:
                        (spec_real if ybin == 0 else spec_ai).append(rps)
                    pixel_done += 1
                except Exception:
                    pass

    fmt_by_label = {}
    rec_label_arr = np.array(rec_label)
    rec_fmt_arr = np.array(rec_fmt)
    for yb in (0, 1):
        if has_label:
            m = rec_label_arr == yb
            if m.any():
                fmt_by_label[yb] = dict(Counter(rec_fmt_arr[m].tolist()))
    sizes = [(w, h) for w, h in zip(rec_w, rec_h) if w > 0]

    summary = {
        "split": name,
        "n_rows": int(n_rows),
        "has_label": has_label,
        "class_counts_source": {int(k): int(v) for k, v in sorted(class_counts.items())},
        "binary_counts": {
            "real_0": int(class_counts.get(0, 0)),
            "ai_1to5": int(sum(v for k, v in class_counts.items() if k != 0)),
        } if has_label else None,
        "decoded_sample": int(decoded),
        "format_overall": dict(Counter(rec_fmt)),
        "format_by_label": fmt_by_label,
        "mode_overall": dict(Counter(rec_mode)),
        "bytes_mean": float(np.mean(rec_bytes)) if rec_bytes else None,
        "bytes_median": float(np.median(rec_bytes)) if rec_bytes else None,
        "width_unique_top": dict(Counter(rec_w).most_common(8)),
        "height_unique_top": dict(Counter(rec_h).most_common(8)),
        "size_examples": sizes[:10],
    }
    if has_label and (col_real or col_ai):
        def agg(rows):
            a = np.array(rows) if rows else np.zeros((0, 4))
            return {
                "n": int(len(a)),
                "mean_R": float(a[:, 0].mean()) if len(a) else None,
                "mean_G": float(a[:, 1].mean()) if len(a) else None,
                "mean_B": float(a[:, 2].mean()) if len(a) else None,
                "saturation": float(a[:, 3].mean()) if len(a) else None,
            }
        summary["color_real"] = agg(col_real)
        summary["color_ai"] = agg(col_ai)

    # ---- figures ----
    if has_label and (spec_real or spec_ai):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        if spec_real:
            sr = np.array(spec_real).mean(0)
            plt.plot(np.linspace(0, 1, len(sr)), sr, label=f"real (n={len(spec_real)})", lw=2)
        if spec_ai:
            sa = np.array(spec_ai).mean(0)
            plt.plot(np.linspace(0, 1, len(sa)), sa, label=f"ai (n={len(spec_ai)})", lw=2)
        plt.xlabel("normalised spatial frequency"); plt.ylabel("log power")
        plt.title(f"Radial power spectrum — {name}"); plt.legend(); plt.tight_layout()
        plt.savefig(OUT / f"spectrum_{name}.png", dpi=110); plt.close()

    return summary


def main() -> None:
    all_summ = {}
    for name in LABELED + ["predict"]:
        if not (DATA / name).is_dir():
            continue
        print(f"\n=== exploring {name} ===", flush=True)
        s = explore_split(name)
        all_summ[name] = s
        print(json.dumps(s, indent=2))
    (OUT / "explore_summary.json").write_text(json.dumps(all_summ, indent=2))
    print(f"\nwrote {OUT/'explore_summary.json'}")


if __name__ == "__main__":
    main()
