"""Architecture and pipeline diagrams, drawn with matplotlib so they can be edited and regenerated."""

import os, sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from project_settings import ASSETS
from plot_style import apply

apply()
FIG = os.path.join(ASSETS, "figures")
GREY, BLUE, GREEN, PINK, YELLOW, SAND = "#F2F2F2", "#DCE6F2", "#DDEBD5", "#F3DCE8", "#FFF2C6", "#F1E4D1"


def box(ax, x, y, w, h, text, fc=BLUE, fs=8, bold=False, rot=0):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="#4A4A4A", lw=0.8))
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        rotation=rot,
        fontweight="bold" if bold else "normal",
        linespacing=1.25,
    )


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, lw=0.9, color="#333333", linestyle=ls
        )
    )


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def layer_chain(layers, name, note):
    fig, ax = canvas(7.2, 2.2, len(layers) * 1.2 + 0.2, 4)
    for i, (text, colour) in enumerate(layers):
        x = 0.2 + i * 1.2
        box(ax, x, 1.0, 0.95, 2.6, text, fc=colour, fs=7.5, rot=90)
        if i:
            arrow(ax, x - 0.25, 2.3, x, 2.3)
    ax.text(0.2, 0.45, note, fontsize=7.5, color="#444444")
    save(fig, name)


# 1. baseline CNN
layer_chain(
    [
        ("Input 256×256×3", GREY),
        ("Conv 3×3, 32", BLUE),
        ("MaxPool 2×2", SAND),
        ("Conv 3×3, 64", BLUE),
        ("MaxPool 2×2", SAND),
        ("Conv 3×3, 64", BLUE),
        ("MaxPool 2×2", SAND),
        ("Conv 3×3, 64", BLUE),
        ("MaxPool 2×2", SAND),
        ("Flatten", GREY),
        ("Dense 256 (features)", GREEN),
        ("Dropout 0.5", GREY),
        ("Dense 4, softmax", PINK),
    ],
    "fig_arch_cnn.png",
    "Baseline CNN. All convolutions use ReLU. Grad-CAM is taken from the last convolution.",
)

# 2. improved CNN
layer_chain(
    [
        ("Input 224×224×3", GREY),
        ("Augmentation", YELLOW),
        ("Conv 32 + BN", BLUE),
        ("MaxPool", SAND),
        ("Conv 64 + BN", BLUE),
        ("MaxPool", SAND),
        ("Conv 128 + BN", BLUE),
        ("MaxPool", SAND),
        ("Conv 256 + BN", BLUE),
        ("MaxPool", SAND),
        ("Global avg. pool", GREY),
        ("Dense 256 (features)", GREEN),
        ("Dropout 0.5", GREY),
        ("Dense 4, softmax", PINK),
    ],
    "fig_arch_cnn_improved.png",
    "Improved CNN. Images are cropped to the head first. BN = batch normalisation; all convolutions are 3×3 with ReLU.",
)

# 3. six-predictor ensemble
fig, ax = canvas(7.2, 4.0, 16, 9)
box(ax, 0.2, 3.9, 2.0, 1.2, "MRI image", fc=GREY, bold=True)
deep = ["Custom CNN", "InceptionV3", "Xception"]
for i, name in enumerate(deep + [f"{d} features\n(Dense 256)" for d in deep]):
    y = 7.9 - i * 1.45
    arrow(ax, 2.2, 4.5, 3.2, y + 0.5)
    box(ax, 3.2, y, 3.6, 1.0, name, fc=BLUE if i < 3 else GREEN, fs=8)
    if i >= 3:
        box(ax, 7.4, y, 2.6, 1.0, "RF + DT + SVM\nsoft vote", fc=YELLOW, fs=8)
        arrow(ax, 6.8, y + 0.5, 7.4, y + 0.5)
        arrow(ax, 10.0, y + 0.5, 11.0, 4.5)
    else:
        arrow(ax, 6.8, y + 0.5, 11.0, 4.5)
box(ax, 11.0, 3.6, 2.4, 1.8, "Six class-\nprobability\nvectors", fc=PINK, fs=8)
box(ax, 14.0, 4.8, 1.9, 1.2, "Majority\nvote", fc=PINK, fs=8)
box(ax, 14.0, 3.0, 1.9, 1.2, "Confidence-\nweighted vote", fc=PINK, fs=7.5)
arrow(ax, 13.4, 4.8, 14.0, 5.3)
arrow(ax, 13.4, 4.2, 14.0, 3.6)
save(fig, "fig_arch_ensemble.png")

# 4. explainability pipeline
fig, ax = canvas(7.2, 3.6, 16, 8)
box(ax, 0.1, 3.4, 2.3, 1.2, "MRI image,\npredicted class", fc=GREY, fs=6.8)
for i, name in enumerate(["CNN", "Xception", "InceptionV3"]):
    y = 6.0 - i * 2.2
    arrow(ax, 2.4, 4.0, 2.9, y + 0.5)
    box(ax, 2.9, y, 2.1, 1.0, name, fs=6.8)
    arrow(ax, 5.0, y + 0.5, 5.4, y + 0.5)
    box(
        ax,
        5.4,
        y - 0.3,
        3.6,
        1.6,
        "Grad-CAM, Grad-CAM++,\nHiResCAM, occlusion,\nLIME" + (", SHAP" if i == 0 else ""),
        fc=GREEN,
        fs=6.8,
    )
    arrow(ax, 9.0, y + 0.5, 9.5, [6.6, 4.0, 1.4][i])
for y, text in [
    (5.8, "Examples per class\n(correct and\nmisclassified)"),
    (3.2, "Faithfulness\n(deletion and\ninsertion AUC)"),
    (0.6, "Sanity check\n(weight\nrandomisation)"),
]:
    box(ax, 9.5, y, 3.0, 1.6, text, fc=YELLOW, fs=6.8)
for y, text in [
    (5.8, "Agreement\nbetween models"),
    (3.2, "Attention outside\nthe brain"),
    (0.6, "SHAP on the\nRandom Forest"),
]:
    box(ax, 13.0, y, 2.9, 1.6, text, fc=PINK, fs=6.8)
    arrow(ax, 12.5, y + 0.8, 13.0, y + 0.8)
save(fig, "fig_xai_pipeline.png")

# 5. conceptual framework with the feedback loop
fig, ax = canvas(7.2, 3.0, 16, 6.6)
stages = [
    ("Data\n7,200 slices,\n4 classes", GREY),
    ("Pre-processing\nresize, scale,\nhead crop*", GREY),
    ("Feature\nlearning\nCNN, IncV3,\nXception", BLUE),
    ("Classification\nsoftmax,\nRF, DT, SVM", GREEN),
    ("Ensemble\nmajority and\nweighted vote", PINK),
    ("Evaluation\nfull and\nclean test sets", YELLOW),
]
for i, (text, colour) in enumerate(stages):
    x = 0.2 + i * 2.65
    box(ax, x, 3.8, 2.3, 2.3, text, fc=colour, fs=6.8)
    if i:
        arrow(ax, x - 0.35, 4.95, x, 4.95)
box(
    ax,
    4.0,
    0.6,
    11.8,
    1.8,
    "Explainability: Grad-CAM family, LIME, SHAP, occlusion\n"
    "faithfulness, sanity check, agreement between models,\nattention outside the brain",
    fc=YELLOW,
    fs=6.8,
)
for x0 in (6.65, 9.3, 11.95):
    arrow(ax, x0, 3.8, x0, 2.4)
arrow(ax, 4.0, 1.5, 1.35, 1.5, ls="--")
arrow(ax, 1.35, 1.5, 1.35, 3.8, ls="--")
ax.text(1.55, 2.95, "findings guide\nthe design", fontsize=6.5, color="#333333", va="center")
ax.text(
    0.2,
    0.05,
    "* head cropping, augmentation and fine-tuning are used in the improved configuration",
    fontsize=6.5,
    color="#555555",
)
save(fig, "fig_conceptual_framework.png")
print("ok")
