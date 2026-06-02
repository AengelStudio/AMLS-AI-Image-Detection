"""Assemble report/numbers.tex from the pipeline's metric JSON files.

Reproducible reporting: every number in the report is generated from the actual
run artefacts rather than typed by hand. Missing artefacts emit a visible '??'.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
SOL = ROOT / "solution" / "artifacts"
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def f3(x):
    return "??" if x is None else f"{x:.3f}"


def f2(x):
    return "??" if x is None else f"{x:.2f}"


clean = load(SOL / "clean" / "clean_stats.json") or {}
expl = load(OUT / "explore_summary.json") or {}
base = load(OUT / "baseline_metrics.json") or {}
t2 = load(SOL / "task02" / "metrics.json") or {}
t3 = load(SOL / "task03" / "metrics.json") or {}
expln = load(OUT / "explain_summary.json") or {}
timing = load(OUT / "timing.json") or {}

sq = clean.get("square_by_label", {})
real_sq = sq.get("real", {})
ai_sq = sq.get("ai", {})


def split_metric(d, split, key):
    s = (d or {}).get(split)
    if isinstance(s, dict) and not s.get("missing"):
        return s.get(key)
    return None


def n_rows(split):
    return (expl.get(split) or {}).get("n_rows")


cmds = {
    "TEAMMEMBERS": r"Marco Reczuch \and David Birgel \and Jerome Engelbrecht",
    "Ntrain": str(n_rows("train") or "??"),
    "Ncal": str(n_rows("calibration") or "??"),
    "Nval": str(n_rows("validation") or "??"),
    "Nkept": str(clean.get("n_kept", "??")),
    "Ndup": str(clean.get("n_dropped_duplicate", "??")),
    "Nbad": str(clean.get("n_dropped_bad", "??")),
    "RealNonSquare": str(real_sq.get("non_square", "??")),
    "RealSquare": str(real_sq.get("square", "??")),
    "AiSquare": str(ai_sq.get("square", "??")),
    "Targetfpr": "0.15",
    "ImgSize": "64",
    "CnnParams": "430",
    # classical baseline
    "BaseValFpr": f3((base.get("validation") or {}).get("fpr_real")),
    "BaseValRecall": f3((base.get("validation") or {}).get("recall_ai")),
    "BaseAugFpr": f3((base.get("validation_augmented") or {}).get("fpr_real")),
    "BaseAugRecall": f3((base.get("validation_augmented") or {}).get("recall_ai")),
    # CNN task 2
    "CnnValFpr": f3(split_metric(t2, "validation", "fpr_real")),
    "CnnValRecall": f3(split_metric(t2, "validation", "recall_ai")),
    "CnnAugFpr": f3(split_metric(t2, "validation_augmented", "fpr_real")),
    "CnnAugRecall": f3(split_metric(t2, "validation_augmented", "recall_ai")),
    # CNN task 3
    "AugValFpr": f3(split_metric(t3, "validation", "fpr_real")),
    "AugValRecall": f3(split_metric(t3, "validation", "recall_ai")),
    "AugAugFpr": f3(split_metric(t3, "validation_augmented", "fpr_real")),
    "AugAugRecall": f3(split_metric(t3, "validation_augmented", "recall_ai")),
    # explainability
    "OccCenterReal": f3((expln.get("occlusion_center_border_real") or [None, None])[0]),
    "OccBorderReal": f3((expln.get("occlusion_center_border_real") or [None, None])[1]),
    "OccCenterAi": f3((expln.get("occlusion_center_border_ai") or [None, None])[0]),
    "OccBorderAi": f3((expln.get("occlusion_center_border_ai") or [None, None])[1]),
    # timing
    "RefSeconds": "??" if timing.get("ref_seconds") is None else str(round(timing["ref_seconds"])),
    "TrainSeconds": "??" if timing.get("train_seconds") is None else str(round(timing["train_seconds"])),
    "TrainRatio": f2(timing.get("train_ratio")),
}

lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in cmds.items()]
(REPORT / "numbers.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {REPORT/'numbers.tex'} with {len(cmds)} macros")
for k, v in cmds.items():
    print(f"  \\{k} = {v}")
