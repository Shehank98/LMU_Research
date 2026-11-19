"""Worked example of the EAA-IoU score for the report (Section 3.10.9).

Three test images are explained with Grad-CAM by the CNN, Xception and InceptionV3 (improved models): one with high
agreement, one moderate and one low (a misclassified image). For each image the script shows the heatmaps, the binary
masks at 0.5, the overlap of every pair and the resulting IoU values, and writes the numbers to
paper_assets/xai_results_improved/eaa_worked_example.json. It also counts images and errors per EAA-IoU band for
both configurations (eaa_bands.json).

Run with MODEL_SET=improved.
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(__file__))
from common import *
from xai_lib import CamModel, gradcam, load, TARGET
import style

style.apply()
assert VERSION == "improved", "run with MODEL_SET=improved"

EXAMPLES = [(149, "High agreement"), (1209, "Moderate agreement"), (235, "Low agreement (flagged)")]
NETS = ["CNN", "Xception", "InceptionV3"]
PAIRS = [("CNN", "Xception"), ("CNN", "InceptionV3"), ("Xception", "InceptionV3")]
T = 0.5


def band(v):
    return (
        "strong (0.20 or above)"
        if v >= 0.20
        else ("moderate (0.10 to 0.20)" if v >= 0.10 else "alert (below 0.10)")
    )


pred = pd.read_csv(PRED_CSV)
eaa = pd.read_csv(os.path.join(RES, "eaa_iou_per_image.csv"))
X, y, files = load_numpy("Testing", IMG)
idx = [i for i, _ in EXAMPLES]
Xs = X[idx]
del X
cls = pred.pred_confidence_weighted.values[idx]
maps = {}
for n in NETS:
    maps[n] = gradcam(CamModel(load(n), TARGET[n]), Xs, cls)

out, rows = [], []
for k, (i, title) in enumerate(EXAMPLES):
    m = {n: maps[n][k] >= T for n in NETS}
    pairs = []
    for a, b in PAIRS:
        inter = int(np.logical_and(m[a], m[b]).sum())
        union = int(np.logical_or(m[a], m[b]).sum())
        pairs.append(
            dict(
                pair=f"{a} / {b}",
                intersection_px=inter,
                union_px=union,
                iou=round(inter / union, 4) if union else 0.0,
            )
        )
    score = round(float(np.mean([p["iou"] for p in pairs])), 4)
    logged = eaa.loc[eaa.idx == i].iloc[0]
    out.append(
        dict(
            idx=i,
            title=title,
            file=files[i],
            true=LABELS[int(y[i])],
            pred=LABELS[int(cls[k])],
            correct=bool(y[i] == cls[k]),
            ensemble_confidence_pct=round(100 * float(pred.ensemble_confidence[i]), 2),
            mask_area_pct={n: round(100 * float(m[n].mean()), 2) for n in NETS},
            pairs=pairs,
            eaa_iou=score,
            eaa_iou_logged=float(logged.eaa_iou),
            band=band(score),
        )
    )

# ---------- figure: for each example, heatmaps (row 1) and pair overlaps (row 2)
OVL = ListedColormap(["#000000", "#3B6FB6", "#E07B39", "#F2F2F2"])  # neither, first only, second only, both
fig, axes = plt.subplots(6, 4, figsize=(7.2, 12.0), gridspec_kw=dict(hspace=0.55, wspace=0.12))
for k, ex in enumerate(out):
    img = Xs[k]
    r1, r2 = axes[2 * k], axes[2 * k + 1]
    r1[0].imshow(img)
    ok = "correct" if ex["correct"] else "wrong"
    r1[0].set_title(f"Input MRI\ntrue: {ex['true']}\nensemble: {ex['pred']} ({ok})", fontsize=7.5)
    for j, n in enumerate(NETS):
        r1[j + 1].imshow(img)
        r1[j + 1].imshow(maps[n][k], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        r1[j + 1].contour(maps[n][k] >= T, levels=[0.5], colors="white", linewidths=0.8)
        r1[j + 1].set_title(
            f"{n} Grad-CAM\nmask (≥ 0.5) = {ex['mask_area_pct'][n]:.1f}%\nof the image", fontsize=7.5
        )
    r2[0].axis("off")
    txt = "\n".join(f"{p['pair']}: {p['iou']:.3f}" for p in ex["pairs"])
    r2[0].text(
        0.0,
        0.5,
        f"{chr(65 + k)}. {ex['title']}\n\nPairwise IoU\n{txt}\n\nEAA-IoU = mean = {ex['eaa_iou']:.3f}\n{ex['band']}",
        fontsize=7.2,
        va="center",
        family="DejaVu Sans",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F4F6FA", ec="#9AA5B8"),
    )
    for j, ((a, b), p) in enumerate(zip(PAIRS, ex["pairs"])):
        code = (maps[a][k] >= T).astype(int) + 2 * (maps[b][k] >= T).astype(int)
        r2[j + 1].imshow(code, cmap=OVL, vmin=0, vmax=3, interpolation="nearest")
        r2[j + 1].set_title(
            f"{a} vs {b}\n{p['intersection_px']:,} / {p['union_px']:,} px\nIoU = {p['iou']:.3f}", fontsize=7.5
        )
    for a in list(r1) + list(r2):
        a.set_xticks([])
        a.set_yticks([])
from matplotlib.patches import Patch

fig.legend(
    handles=[
        Patch(color="#3B6FB6", label="first model only"),
        Patch(color="#E07B39", label="second model only"),
        Patch(facecolor="#F2F2F2", edgecolor="#777777", label="both (intersection)"),
        Patch(color="#000000", label="neither"),
    ],
    loc="lower center",
    ncol=4,
    fontsize=8,
    bbox_to_anchor=(0.5, 0.07),
)
fig.savefig(os.path.join(FIG, "fig_eaa_worked_example.png"))
plt.close(fig)

# ---------- images and errors per EAA-IoU band, both configurations
bands = {}
for name, path in [
    ("baseline", os.path.join(ROOT, "metrics", "eaa_iou_per_image.csv")),
    ("improved", os.path.join(RES, "eaa_iou_per_image.csv")),
]:
    d = pd.read_csv(path).drop_duplicates("idx")
    n_err = int((d.correct == 0).sum())
    res = []
    for lo, hi, lab in [
        (0, 0.10, "below 0.10 (alert)"),
        (0.10, 0.20, "0.10 to 0.20 (moderate)"),
        (0.20, 1.01, "0.20 or above (strong)"),
    ]:
        s = d[(d.eaa_iou >= lo) & (d.eaa_iou < hi)]
        e = int((s.correct == 0).sum())
        res.append(
            dict(
                band=lab,
                images=int(len(s)),
                images_pct=round(100 * len(s) / len(d), 2),
                errors=e,
                error_rate_pct=round(100 * e / len(s), 2) if len(s) else 0.0,
                share_of_errors_pct=round(100 * e / n_err, 2) if n_err else 0.0,
            )
        )
    bands[name] = dict(
        n_images=int(len(d)), n_errors=n_err, overall_error_rate_pct=round(100 * n_err / len(d), 2), bands=res
    )

json.dump(dict(threshold=T, examples=out), open(os.path.join(RES, "eaa_worked_example.json"), "w"), indent=2)
json.dump(bands, open(os.path.join(RES, "eaa_bands.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
print(json.dumps(bands, indent=1))
