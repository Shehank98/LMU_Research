"""Figure for Section 5.7: image size by class and accuracy with parts of the image hidden."""

import os, sys, json, numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from project_settings import RES, FIG, CLASSES, LABELS
from plot_style import apply, PALETTE

apply()
d = json.load(open(os.path.join(RES, "shortcut_investigation.json")))
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.4]))
share = [d["size_by_class"][c]["share_512x512_pct"] for c in CLASSES]
bars = a.bar(["Glioma", "Menin-\ngioma", "No\ntumour", "Pituitary"], share, color=PALETTE[:4])
a.bar_label(bars, fmt="%.1f", fontsize=8)
a.set_ylabel("Images that are 512 × 512 px (%)")
a.set_ylim(0, 110)
a.set_title("Image size by class (test set)")
a.tick_params(axis="x", labelsize=8)
models = ["CNN", "Xception", "InceptionV3"]
x = np.arange(3)
w = 0.26
for k, (key, lab, col) in enumerate(
    [
        ("full_image", "Whole image", "#3B6FB6"),
        ("head_only", "Head only", "#4E9F6E"),
        ("outside_head_only", "Outside head only", "#B5487F"),
    ]
):
    bb = b.bar(x + (k - 1) * w, [d[m][key] for m in models], w, label=lab, color=col)
    b.bar_label(bb, fmt="%.1f", fontsize=7)
b.axhline(25, ls=":", color="grey", lw=0.8, label="Chance (25%)")
b.set_xticks(x, models)
b.set_ylim(0, 105)
b.set_ylabel("Test accuracy (%)")
b.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.12))
b.set_title("Accuracy with parts of the image hidden")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_shortcut_investigation.png"))
print("ok")
