"""Figures for the final report (baseline configuration only).

1. fig_framework_final.png      conceptual framework (models, ensemble, Grad-CAM, EAA-IoU, Streamlit app)
2. fig_accuracy_bars.png        test accuracy of every configuration
3. fig_gradcam_correct.png      Grad-CAM of CNN, InceptionV3 and Xception for the most confident correct image per class
4. fig_gradcam_errors.png       the same for the most frequent error of each class
5. fig_eaa_example_baseline.png EAA-IoU worked example on three test images (+ eaa_example_baseline.json)

The CNN map uses the element-wise form (gradient x activation) that the training notebook used for the
EAA-IoU stage; for InceptionV3 and Xception this is identical to Grad-CAM.
Run with MODEL_SET=baseline (the default).
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, os.path.dirname(__file__))
from common import *
import style

style.apply()
assert VERSION == "baseline"
OUT = os.path.join(ASSETS, "figures_final")
os.makedirs(OUT, exist_ok=True)
NETS = ["CNN", "InceptionV3", "Xception"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# ---------- 1. conceptual framework
def box(ax, x, y, w, h, text, fc, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#555555", lw=0.8))
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        weight="bold" if bold else "normal",
    )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))


fig, ax = plt.subplots(figsize=(8.0, 3.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")
box(ax, 0.1, 2.0, 1.4, 1.0, "MRI dataset\n7,200 images\n4 classes", "#EEEEEE", True)
box(ax, 1.9, 2.0, 1.5, 1.0, "Pre-processing\nresize 256 / 128\nscale to [0, 1]", "#E6EEF7")
for i, (t, y) in enumerate(
    [("Custom CNN", 4.0), ("InceptionV3\n(frozen backbone)", 2.95), ("Xception\n(frozen backbone)", 1.9)]
):
    box(ax, 3.9, y, 1.6, 0.8, t, "#DCE6F2")
    arrow(ax, 3.4, 2.5, 3.9, y + 0.4)
box(ax, 3.9, 0.55, 1.6, 0.95, "Deep features\n(Dense 256)\nRF + DT + SVM", "#FFF2CC")
arrow(ax, 3.4, 2.5, 3.9, 1.0)
arrow(ax, 1.5, 2.5, 1.9, 2.5)
box(ax, 6.0, 1.9, 1.5, 1.4, "Six-predictor\nensemble\nmajority and\nconfidence-\nweighted vote", "#F4DDEA")
for y in (4.4, 3.35, 2.3, 1.0):
    arrow(ax, 5.5, y, 6.0, 2.6)
box(ax, 8.0, 3.2, 1.9, 1.2, "Evaluation\naccuracy, P, R, F1\nconfusion, ROC/PR", "#E2F0D9")
box(ax, 8.0, 1.6, 1.9, 1.2, "Grad-CAM on the\n3 deep networks\nEAA-IoU agreement", "#E2F0D9")
box(ax, 8.0, 0.1, 1.9, 1.1, "Streamlit clinical\ndecision-support app", "#FCE4D6")
arrow(ax, 7.5, 2.6, 8.0, 3.8)
arrow(ax, 7.5, 2.6, 8.0, 2.2)
arrow(ax, 8.95, 1.6, 8.95, 1.2)
save(fig, "fig_framework_final.png")

# ---------- 2. accuracy of every configuration
rm = pd.read_csv(os.path.join(ROOT, "metrics", "results_master.csv"))
NAMES = {
    "CNN_standalone": ("Custom CNN", "Deep"),
    "InceptionV3_standalone": ("InceptionV3", "Deep"),
    "Xception_standalone": ("Xception", "Deep"),
    "raw_rf": ("Raw pixels + RF", "Raw"),
    "raw_svm": ("Raw pixels + SVM", "Raw"),
    "raw_lr": ("Raw pixels + LR", "Raw"),
    "raw_dt": ("Raw pixels + DT", "Raw"),
    "raw_softvote_ensemble": ("Raw pixels soft vote", "Raw"),
    "cnn_feat_rf": ("CNN feat. + RF", "Feat"),
    "cnn_feat_svm": ("CNN feat. + SVM", "Feat"),
    "cnn_feat_dt": ("CNN feat. + DT", "Feat"),
    "cnn_feat_softvote_ensemble": ("CNN feat. soft vote", "Feat"),
    "inceptionv3_feat_rf": ("InceptionV3 feat. + RF", "Feat"),
    "inceptionv3_feat_svm": ("InceptionV3 feat. + SVM", "Feat"),
    "inceptionv3_feat_dt": ("InceptionV3 feat. + DT", "Feat"),
    "inceptionv3_feat_softvote_ensemble": ("InceptionV3 feat. soft vote", "Feat"),
    "xception_feat_rf": ("Xception feat. + RF", "Feat"),
    "xception_feat_svm": ("Xception feat. + SVM", "Feat"),
    "xception_feat_dt": ("Xception feat. + DT", "Feat"),
    "xception_feat_softvote_ensemble": ("Xception feat. soft vote", "Feat"),
    "FINAL_majority_vote": ("Six-model majority vote", "Ens"),
    "FINAL_confidence_weighted": ("Six-model confidence-weighted", "Ens"),
}
COL = {"Deep": "#3B6FB6", "Raw": "#9AA5B8", "Feat": "#E0A458", "Ens": "#B5487F"}
d = rm[rm.model.isin(NAMES)].copy()
d["label"] = d.model.map(lambda m: NAMES[m][0])
d["grp"] = d.model.map(lambda m: NAMES[m][1])
d = d.sort_values("accuracy")
fig, ax = plt.subplots(figsize=(6.6, 6.4))
ax.barh(d.label, 100 * d.accuracy, color=[COL[g] for g in d.grp])
for y, v in enumerate(100 * d.accuracy):
    ax.text(v + 0.3, y, f"{v:.2f}", va="center", fontsize=7.5)
ax.set_xlim(70, 96)
ax.set_xlabel("Test accuracy (%), 1,600 images")
ax.legend(
    handles=[
        Patch(color=COL[k], label=l)
        for k, l in [
            ("Deep", "Standalone deep model"),
            ("Raw", "Classical on raw pixels"),
            ("Feat", "Classical on deep features"),
            ("Ens", "Six-predictor ensemble"),
        ]
    ],
    loc="lower right",
    fontsize=7.5,
)
save(fig, "fig_accuracy_bars.png")

# ---------- 3./4. Grad-CAM example panels (maps computed by 02_xai_examples.py)
z = np.load(os.path.join(RES, "example_maps.npz"))
ex = json.load(open(os.path.join(RES, "examples.json")))["examples"]
MAPKEY = {"CNN": "hirescam_CNN", "InceptionV3": "gradcam_InceptionV3", "Xception": "gradcam_Xception"}


def panel(kind, name):
    rows = [(k, e) for k, e in enumerate(ex) if e["kind"] == kind]
    fig, axes = plt.subplots(
        len(rows), 4, figsize=(6.4, 1.75 * len(rows)), gridspec_kw=dict(hspace=0.35, wspace=0.05)
    )
    for r, (k, e) in enumerate(rows):
        img = z["X"][k]
        axes[r, 0].imshow(img)
        t = f"true: {LABELS[e['true']]}\nensemble: {LABELS[e['pred']]} ({100 * e['ensemble_confidence']:.1f}%)"
        axes[r, 0].set_title(t, fontsize=7)
        for j, n in enumerate(NETS):
            axes[r, j + 1].imshow(img)
            axes[r, j + 1].imshow(z[MAPKEY[n]][k], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            if r == 0:
                axes[r, j + 1].set_title(f"{n}\n", fontsize=8)
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
    save(fig, name)


panel("correct", "fig_gradcam_correct.png")
panel("misclassified", "fig_gradcam_errors.png")

# ---------- 5. EAA-IoU worked example (recomputed; must match metrics/eaa_iou_per_image.csv)
from xai_lib import CamModel, gradcam, hirescam, load, TARGET

e = pd.read_csv(os.path.join(ROOT, "metrics", "eaa_iou_per_image.csv")).drop_duplicates("idx")
pred = pd.read_csv(PRED_CSV)
moderate = int(e[(e.correct == 1) & (e.true == 3) & e.eaa_iou.between(0.14, 0.16)].idx.iloc[0])
EX = [(163, "High agreement"), (moderate, "Moderate agreement"), (0, "Low agreement (flagged)")]
X, y, files = load_numpy("Testing", IMG)
idx = [i for i, _ in EX]
Xs = X[idx]
del X
cls = pred.pred_confidence_weighted.values[idx]
maps = {}
for n in NETS:
    cm = CamModel(load(n), TARGET[n])
    maps[n] = (hirescam if n == "CNN" else gradcam)(cm, Xs, cls)
PAIRS = [("CNN", "Xception"), ("CNN", "InceptionV3"), ("Xception", "InceptionV3")]
out = []
for k, (i, title) in enumerate(EX):
    m = {n: maps[n][k] >= 0.5 for n in NETS}
    prs = []
    for a, b in PAIRS:
        inter, union = int((m[a] & m[b]).sum()), int((m[a] | m[b]).sum())
        prs.append(
            dict(
                pair=f"{a} / {b}",
                intersection_px=inter,
                union_px=union,
                iou=round(inter / union, 4) if union else 0.0,
            )
        )
    v = round(float(np.mean([p["iou"] for p in prs])), 4)
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
            pairs=prs,
            eaa_iou=v,
            eaa_iou_logged=float(e.loc[e.idx == i, "eaa_iou"].iloc[0]),
        )
    )
json.dump(out, open(os.path.join(RES, "eaa_example_baseline.json"), "w"), indent=2)

OVL = ListedColormap(["#000000", "#3B6FB6", "#E07B39", "#F2F2F2"])
fig, axes = plt.subplots(6, 4, figsize=(7.2, 12.0), gridspec_kw=dict(hspace=0.55, wspace=0.12))
for k, ex_ in enumerate(out):
    r1, r2 = axes[2 * k], axes[2 * k + 1]
    ok = "correct" if ex_["correct"] else "wrong"
    r1[0].imshow(Xs[k])
    r1[0].set_title(f"Input MRI\ntrue: {ex_['true']}\nensemble: {ex_['pred']} ({ok})", fontsize=7.5)
    for j, n in enumerate(NETS):
        r1[j + 1].imshow(Xs[k])
        r1[j + 1].imshow(maps[n][k], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        r1[j + 1].contour(maps[n][k] >= 0.5, levels=[0.5], colors="white", linewidths=0.8)
        r1[j + 1].set_title(
            f"{n} Grad-CAM\nmask (≥ 0.5) = {ex_['mask_area_pct'][n]:.1f}%\nof the image", fontsize=7.5
        )
    r2[0].axis("off")
    txt = "\n".join(f"{p['pair']}: {p['iou']:.3f}" for p in ex_["pairs"])
    band = (
        "strong (0.20 or above)"
        if ex_["eaa_iou"] >= 0.2
        else ("moderate (0.10 to 0.20)" if ex_["eaa_iou"] >= 0.1 else "alert (below 0.10)")
    )
    r2[0].text(
        0,
        0.5,
        f"{chr(65 + k)}. {ex_['title']}\n\nPairwise IoU\n{txt}\n\nEAA-IoU = mean = {ex_['eaa_iou']:.3f}\n{band}",
        fontsize=7.2,
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F4F6FA", ec="#9AA5B8"),
    )
    for j, ((a, b), p) in enumerate(zip(PAIRS, ex_["pairs"])):
        code = (maps[a][k] >= 0.5).astype(int) + 2 * (maps[b][k] >= 0.5).astype(int)
        r2[j + 1].imshow(code, cmap=OVL, vmin=0, vmax=3, interpolation="nearest")
        r2[j + 1].set_title(
            f"{a} vs {b}\n{p['intersection_px']:,} / {p['union_px']:,} px\nIoU = {p['iou']:.3f}", fontsize=7.5
        )
    for a in list(r1) + list(r2):
        a.set_xticks([])
        a.set_yticks([])
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
save(fig, "fig_eaa_example_baseline.png")
print(
    json.dumps(
        [
            {
                k: v
                for k, v in o.items()
                if k in ("idx", "true", "pred", "eaa_iou", "eaa_iou_logged", "ensemble_confidence_pct")
            }
            for o in out
        ],
        indent=1,
    )
)
for o in out:
    print(o["idx"], [p["iou"] for p in o["pairs"]], o["mask_area_pct"])
