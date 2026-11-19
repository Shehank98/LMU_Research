"""Section 5.1 and 5.2.1: Grad-CAM examples and the EAA-IoU worked example.

Figures 15 and 16: Grad-CAM of the CNN, InceptionV3 and Xception for the most confident correct image of each
class and for a confident example of each class's most frequent error.
Figure 17 / Table 19: EAA-IoU computed step by step for three test images; the values are checked against the
EAA-IoU logged by the training notebook (../2_Results/metrics/eaa_iou_per_image.csv).
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from settings import *
from gradcam import CamModel, gradcam, iou, load, TARGET

plot_style()
NETS = ["CNN", "InceptionV3", "Xception"]
pred = pd.read_csv(PRED_CSV)
cm = pd.read_csv(os.path.join(CONFUSION, "FINAL_majority_vote_confusion_matrix.csv"), index_col=0).values
X, y, files = load_numpy("Testing", 256)
cams = {n: CamModel(load(n), TARGET[n]) for n in NETS}


def save(f, name):
    f.savefig(os.path.join(FIG, name))
    plt.close(f)


# ---- choose the example images
examples = []
for c in range(4):  # most confident correct prediction per class
    ok = pred[(pred.true == c) & (pred.pred_majority == c)].sort_values("ensemble_confidence", ascending=False)
    examples.append(dict(kind="correct", idx=int(ok.idx.iloc[0]), true=c, pred=c))
for c in range(4):  # most frequent error of each class (or, if none, the most frequent miss into it)
    row = cm[c].copy()
    row[c] = 0
    if row.sum() > 0:
        t, w = c, int(row.argmax())
    else:
        col = cm[:, c].copy()
        col[c] = 0
        t, w = int(col.argmax()), c
    bad = pred[(pred.true == t) & (pred.pred_majority == w)].sort_values("ensemble_confidence", ascending=False)
    examples.append(dict(kind="misclassified", idx=int(bad.idx.iloc[0]), true=t, pred=w, n_such_errors=int(cm[t, w])))
for s in examples:
    s["file"] = files[s["idx"]]
    s["ensemble_confidence"] = round(float(pred.ensemble_confidence[s["idx"]]), 4)

idx = np.array([s["idx"] for s in examples])
cls = np.array([s["pred"] for s in examples])
maps = {n: gradcam(n, cams[n], X[idx], cls) for n in NETS}


def panel(kind, name):
    rows = [(k, s) for k, s in enumerate(examples) if s["kind"] == kind]
    f, axes = plt.subplots(len(rows), 4, figsize=(6.4, 1.75 * len(rows)), gridspec_kw=dict(hspace=0.35, wspace=0.05))
    for r, (k, s) in enumerate(rows):
        img = X[s["idx"]]
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(
            f"true: {LABELS[s['true']]}\nensemble: {LABELS[s['pred']]} ({100 * s['ensemble_confidence']:.1f}%)",
            fontsize=7,
        )
        for j, n in enumerate(NETS):
            axes[r, j + 1].imshow(img)
            axes[r, j + 1].imshow(maps[n][k], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            if r == 0:
                axes[r, j + 1].set_title(f"{n}\n", fontsize=8)
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
    save(f, name)


panel("correct", "fig_gradcam_correct.png")
panel("misclassified", "fig_gradcam_errors.png")

# ---- EAA-IoU worked example (high, moderate and low agreement)
e = pd.read_csv(os.path.join(METRICS, "eaa_iou_per_image.csv")).drop_duplicates("idx")
moderate = int(e[(e.correct == 1) & (e.true == 3) & e.eaa_iou.between(0.14, 0.16)].idx.iloc[0])
EX = [
    (int(e.sort_values("eaa_iou").idx.iloc[-1]), "High agreement"),
    (moderate, "Moderate agreement"),
    (0, "Low agreement (flagged)"),
]
ids = [i for i, _ in EX]
cls = pred.pred_confidence_weighted.values[ids]  # the maps explain the ensemble decision
wm = {n: gradcam(n, cams[n], X[ids], cls) for n in NETS}
PAIRS = [("CNN", "Xception"), ("CNN", "InceptionV3"), ("Xception", "InceptionV3")]
out = []
for k, (i, title) in enumerate(EX):
    m = {n: wm[n][k] >= 0.5 for n in NETS}
    prs = [
        dict(
            pair=f"{a} / {b}",
            intersection_px=int((m[a] & m[b]).sum()),
            union_px=int((m[a] | m[b]).sum()),
            iou=round(iou(wm[a][k], wm[b][k]), 4),
        )
        for a, b in PAIRS
    ]
    score = round(float(np.mean([p["iou"] for p in prs])), 4)
    out.append(
        dict(
            idx=i,
            title=title,
            file=files[i],
            true=LABELS[int(y[i])],
            pred=LABELS[int(cls[k])],
            ensemble_confidence_pct=round(100 * float(pred.ensemble_confidence[i]), 2),
            mask_area_pct={n: round(100 * float(m[n].mean()), 2) for n in NETS},
            pairs=prs,
            eaa_iou=score,
            eaa_iou_logged=float(e.loc[e.idx == i, "eaa_iou"].iloc[0]),
        )
    )

OVL = ListedColormap(["#000000", "#3B6FB6", "#E07B39", "#F2F2F2"])  # neither, first only, second only, both
f, axes = plt.subplots(6, 4, figsize=(7.2, 12.0), gridspec_kw=dict(hspace=0.55, wspace=0.12))
for k, ex in enumerate(out):
    r1, r2 = axes[2 * k], axes[2 * k + 1]
    img = X[ex["idx"]]
    r1[0].imshow(img)
    r1[0].set_title(
        f"Input MRI\ntrue: {ex['true']}\nensemble: {ex['pred']} ({'correct' if ex['true'] == ex['pred'] else 'wrong'})",
        fontsize=7.5,
    )
    for j, n in enumerate(NETS):
        r1[j + 1].imshow(img)
        r1[j + 1].imshow(wm[n][k], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        r1[j + 1].contour(wm[n][k] >= 0.5, levels=[0.5], colors="white", linewidths=0.8)
        r1[j + 1].set_title(f"{n} Grad-CAM\nmask (≥ 0.5) = {ex['mask_area_pct'][n]:.1f}%\nof the image", fontsize=7.5)
    r2[0].axis("off")
    band = (
        "strong (0.20 or above)"
        if ex["eaa_iou"] >= 0.2
        else "moderate (0.10 to 0.20)" if ex["eaa_iou"] >= 0.1 else "alert (below 0.10)"
    )
    txt = "\n".join(f"{p['pair']}: {p['iou']:.3f}" for p in ex["pairs"])
    r2[0].text(
        0,
        0.5,
        f"{chr(65 + k)}. {ex['title']}\n\nPairwise IoU\n{txt}\n\nEAA-IoU = mean = {ex['eaa_iou']:.3f}\n{band}",
        fontsize=7.2,
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F4F6FA", ec="#9AA5B8"),
    )
    for j, ((a, b), p) in enumerate(zip(PAIRS, ex["pairs"])):
        code = (wm[a][k] >= 0.5).astype(int) + 2 * (wm[b][k] >= 0.5).astype(int)
        r2[j + 1].imshow(code, cmap=OVL, vmin=0, vmax=3, interpolation="nearest")
        r2[j + 1].set_title(
            f"{a} vs {b}\n{p['intersection_px']:,} / {p['union_px']:,} px\nIoU = {p['iou']:.3f}", fontsize=7.5
        )
    for a in list(r1) + list(r2):
        a.set_xticks([])
        a.set_yticks([])
f.legend(
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
save(f, "fig_eaa_example_baseline.png")
json.dump(
    dict(examples=examples, eaa_worked_example=out), open(os.path.join(OUT, "gradcam_examples.json"), "w"), indent=2
)
for o in out:
    print(o["idx"], o["true"], "->", o["pred"], "EAA", o["eaa_iou"], "logged", o["eaa_iou_logged"])
