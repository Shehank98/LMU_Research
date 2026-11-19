"""Figures 1, 4 and 5: conceptual framework, custom CNN architecture and six-predictor ensemble."""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

from settings import FIG, plot_style

plot_style()
GREY, BLUE, GREEN, PINK, YELLOW, SAND = "#F2F2F2", "#DCE6F2", "#DDEBD5", "#F3DCE8", "#FFF2C6", "#F1E4D1"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def box(ax, x, y, w, h, text, fc=BLUE, fs=8, bold=False, rot=0, round_=False):
    patch = (
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#555555", lw=0.8)
        if round_
        else Rectangle((x, y), w, h, fc=fc, ec="#4A4A4A", lw=0.8)
    )
    ax.add_patch(patch)
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


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333333"))


# ---- Figure 1: conceptual framework
fig, ax = plt.subplots(figsize=(8.0, 3.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")
box(ax, 0.1, 2.0, 1.4, 1.0, "MRI dataset\n7,200 images\n4 classes", "#EEEEEE", 7.2, True, round_=True)
box(ax, 1.9, 2.0, 1.5, 1.0, "Pre-processing\nresize 256 / 128\nscale to [0, 1]", "#E6EEF7", 7.2, round_=True)
arrow(ax, 1.5, 2.5, 1.9, 2.5)
for t, y in [("Custom CNN", 4.0), ("InceptionV3\n(frozen backbone)", 2.95), ("Xception\n(frozen backbone)", 1.9)]:
    box(ax, 3.9, y, 1.6, 0.8, t, "#DCE6F2", 7.2, round_=True)
    arrow(ax, 3.4, 2.5, 3.9, y + 0.4)
box(ax, 3.9, 0.55, 1.6, 0.95, "Deep features\n(Dense 256)\nRF + DT + SVM", "#FFF2CC", 7.2, round_=True)
arrow(ax, 3.4, 2.5, 3.9, 1.0)
box(
    ax,
    6.0,
    1.9,
    1.5,
    1.4,
    "Six-predictor\nensemble\nmajority and\nconfidence-\nweighted vote",
    "#F4DDEA",
    7.2,
    round_=True,
)
for y in (4.4, 3.35, 2.3, 1.0):
    arrow(ax, 5.5, y, 6.0, 2.6)
box(ax, 8.0, 3.2, 1.9, 1.2, "Evaluation\naccuracy, P, R, F1\nconfusion, ROC/PR", "#E2F0D9", 7.2, round_=True)
box(ax, 8.0, 1.6, 1.9, 1.2, "Grad-CAM on the\n3 deep networks\nEAA-IoU agreement", "#E2F0D9", 7.2, round_=True)
box(ax, 8.0, 0.1, 1.9, 1.1, "NeuroScan web app\n(React + FastAPI)", "#FCE4D6", 7.2, round_=True)
arrow(ax, 7.5, 2.6, 8.0, 3.8)
arrow(ax, 7.5, 2.6, 8.0, 2.2)
arrow(ax, 8.95, 1.6, 8.95, 1.2)
save(fig, "fig_framework_final.png")

# ---- Figure 4: custom CNN
layers = [
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
]
fig, ax = plt.subplots(figsize=(7.2, 2.2))
ax.set_xlim(0, len(layers) * 1.2 + 0.2)
ax.set_ylim(0, 4)
ax.axis("off")
for i, (text, colour) in enumerate(layers):
    x = 0.2 + i * 1.2
    box(ax, x, 1.0, 0.95, 2.6, text, colour, 7.5, rot=90)
    if i:
        arrow(ax, x - 0.25, 2.3, x, 2.3)
ax.text(
    0.2,
    0.45,
    "Baseline CNN. All convolutions use ReLU. Grad-CAM is taken from the last convolution.",
    fontsize=7.5,
    color="#444444",
)
save(fig, "fig_arch_cnn.png")

# ---- Figure 5: six-predictor ensemble
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
box(ax, 0.2, 3.9, 2.0, 1.2, "MRI image", GREY, bold=True)
deep = ["Custom CNN", "InceptionV3", "Xception"]
for i, name in enumerate(deep + [f"{d} features\n(Dense 256)" for d in deep]):
    y = 7.9 - i * 1.45
    arrow(ax, 2.2, 4.5, 3.2, y + 0.5)
    box(ax, 3.2, y, 3.6, 1.0, name, BLUE if i < 3 else GREEN)
    if i >= 3:
        box(ax, 7.4, y, 2.6, 1.0, "RF + DT + SVM\nsoft vote", YELLOW)
        arrow(ax, 6.8, y + 0.5, 7.4, y + 0.5)
        arrow(ax, 10.0, y + 0.5, 11.0, 4.5)
    else:
        arrow(ax, 6.8, y + 0.5, 11.0, 4.5)
box(ax, 11.0, 3.6, 2.4, 1.8, "Six class-\nprobability\nvectors", PINK)
box(ax, 14.0, 4.8, 1.9, 1.2, "Majority\nvote", PINK)
box(ax, 14.0, 3.0, 1.9, 1.2, "Confidence-\nweighted vote", PINK, 7.5)
arrow(ax, 13.4, 4.8, 14.0, 5.3)
arrow(ax, 13.4, 4.2, 14.0, 3.6)
save(fig, "fig_arch_ensemble.png")
print("diagrams written to", FIG)
