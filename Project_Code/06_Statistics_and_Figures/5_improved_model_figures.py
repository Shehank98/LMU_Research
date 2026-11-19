"""Figures for Chapter 6: training curves and confusion matrices of the improved run, and accuracy of both
configurations on the full and duplicate-free test sets."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from project_settings import ROOT, ASSETS, LABELS
from plot_style import apply

apply()
RUN = os.path.join(ROOT, "runs", "v2")
FIG = os.path.join(ASSETS, "figures")

# training curves (head stage and fine-tuning stage are shown as one continuous run)
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex="col")
for j, (key, title) in enumerate([("cnn", "Custom CNN"), ("inceptionv3", "InceptionV3"), ("xception", "Xception")]):
    h = pd.read_csv(os.path.join(RUN, "history", f"{key}.csv"))
    e = np.arange(1, len(h) + 1)
    axs[0, j].plot(e, h.accuracy, color="#3B6FB6", label="train")
    axs[0, j].plot(e, h.val_accuracy, color="#E07B39", label="validation")
    axs[1, j].plot(e, h.loss, color="#3B6FB6")
    axs[1, j].plot(e, h.val_loss, color="#E07B39")
    if "stage" in h and (h.stage == "finetune").any():
        start = int(np.argmax(h.stage.values == "finetune")) + 0.5
        for a in axs[:, j]:
            a.axvline(start, ls=":", color="grey", lw=0.8)
        axs[0, j].text(start + 0.2, 0.52, "fine-tuning", fontsize=7, color="grey")
    axs[0, j].set_title(title)
    axs[1, j].set_xlabel("Epoch")
    axs[0, j].set_ylim(0.5, 1.0)
axs[0, 0].set_ylabel("Accuracy")
axs[1, 0].set_ylabel("Loss")
axs[0, 0].legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_training_improved.png"))
plt.close(fig)

# row-normalised confusion matrices
keys = [("cnn", "CNN"), ("inceptionv3", "InceptionV3"), ("xception", "Xception"), ("final_majority", "Final: majority"),
        ("final_confidence_weighted", "Final: conf.-weighted")]
fig, axs = plt.subplots(1, 5, figsize=(8.8, 1.95))
for a, (k, t) in zip(axs, keys):
    m = pd.read_csv(os.path.join(RUN, "confusion", f"{k}.csv"), index_col=0).values
    n = m / m.sum(1, keepdims=True)
    a.imshow(n, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for jj in range(4):
            a.text(jj, i, f"{n[i, jj]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if n[i, jj] > 0.6 else "black")
    a.set_xticks(range(4), ["G", "M", "N", "P"], fontsize=7)
    a.set_yticks(range(4), ["G", "M", "N", "P"], fontsize=7)
    a.set_title(t, fontsize=8)
    a.set_xlabel("Predicted", fontsize=7)
axs[0].set_ylabel("True", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_cm_improved.png"))
plt.close(fig)

# baseline vs improved, full test set and duplicate-free subset
res = json.load(open(os.path.join(ROOT, "metrics", "clean_subset_results.json")))
names = ["CNN", "InceptionV3", "Xception", "CNN_features", "InceptionV3_features", "Xception_features", "final_majority", "final_confidence_weighted"]
nice = ["CNN", "InceptionV3", "Xception", "CNN feat.", "IncV3 feat.", "Xcp feat.", "Majority", "Conf.-wtd."]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
x = np.arange(len(names))
for a, subset, title in [(axs[0], "full", "Full test set (n = 1,600)"), (axs[1], "clean", "Duplicate-free subset (n = 1,264)")]:
    b = a.bar(x - 0.2, [res["baseline"][n][subset]["acc"] for n in names], 0.4, color="#BFBFBF", label="Baseline")
    i = a.bar(x + 0.2, [res["improved"][n][subset]["acc"] for n in names], 0.4, color="#3B6FB6", label="Improved")
    a.bar_label(i, fmt="%.1f", fontsize=6, padding=1)
    a.set_xticks(x, nice, rotation=40, ha="right", fontsize=7.5)
    a.set_ylim(70, 101)
    a.set_title(title)
axs[0].set_ylabel("Accuracy (%)")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_baseline_vs_improved.png"))
plt.close(fig)
print("ok")
