"""Chapter 4 and Section 5.2 to 5.3: result figures and the numbers quoted in the report.

Reads the logged results in ../2_Results (and outputs/standalone_test_probs.npz for the ROC curves) and writes
the figures to figures/ and every derived number to outputs/report_numbers.json.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

from settings import *

plot_style()
master = pd.read_csv(os.path.join(METRICS, "results_master.csv"))
pred = pd.read_csv(PRED_CSV)
y0 = pred.true.values
numbers = {}
NICE = {
    "CNN_standalone": "CNN",
    "InceptionV3_standalone": "InceptionV3",
    "Xception_standalone": "Xception",
    "cnn_feat_rf": "CNN feat. + RF",
    "cnn_feat_dt": "CNN feat. + DT",
    "cnn_feat_svm": "CNN feat. + SVM",
    "cnn_feat_softvote_ensemble": "CNN feat. + soft vote",
    "inceptionv3_feat_rf": "InceptionV3 feat. + RF",
    "inceptionv3_feat_dt": "InceptionV3 feat. + DT",
    "inceptionv3_feat_svm": "InceptionV3 feat. + SVM",
    "inceptionv3_feat_softvote_ensemble": "InceptionV3 feat. + soft vote",
    "xception_feat_rf": "Xception feat. + RF",
    "xception_feat_dt": "Xception feat. + DT",
    "xception_feat_svm": "Xception feat. + SVM",
    "xception_feat_softvote_ensemble": "Xception feat. + soft vote",
    "raw_rf": "Raw pixels + RF",
    "raw_dt": "Raw pixels + DT",
    "raw_svm": "Raw pixels + SVM",
    "raw_lr": "Raw pixels + LR",
    "raw_softvote_ensemble": "Raw pixels + soft vote",
    "FINAL_majority_vote": "Final 6-model majority vote",
    "FINAL_confidence_weighted": "Final 6-model confidence-weighted vote",
}


def save(f, name):
    f.savefig(os.path.join(FIG, name))
    plt.close(f)


# ---- Figure 2: dataset, Figure 3: sample images
cnt = json.load(open(os.path.join(METRICS, "dataset_counts.json")))
f, ax = plt.subplots(figsize=(6, 2.6))
for k, (split, col) in enumerate([("Training", "#3B6FB6"), ("Testing", "#E07B39")]):
    bars = ax.bar(np.arange(4) + (k - 0.5) * 0.38, [cnt[split][c] for c in CLASSES], 0.38, label=split, color=col)
    ax.bar_label(bars, fontsize=8)
ax.set_xticks(np.arange(4), LABELS)
ax.set_ylabel("Images")
ax.set_ylim(0, 1650)
ax.legend()
save(f, "fig_dataset_distribution.png")
X, y, files = load_numpy("Testing", 256)
rng = np.random.default_rng(7)
f, axs = plt.subplots(2, 4, figsize=(6.4, 3.4))
for c in range(4):
    for r in range(2):
        axs[r, c].imshow(X[rng.choice(np.where(y == c)[0])])
        axs[r, c].axis("off")
    axs[0, c].set_title(LABELS[c])
save(f, "fig_sample_mri.png")
del X

# ---- Figures 7 and 8 / Table 8: training curves
for group, names in [
    (
        "standalone",
        [
            ("cnn_model", "CNN (256 px)"),
            ("inceptionv3_model", "InceptionV3 (256 px)"),
            ("xception_model", "Xception (256 px)"),
        ],
    ),
    (
        "extractors",
        [
            ("cnn_ext", "CNN extractor (128 px)"),
            ("inceptionv3_ext", "InceptionV3 extractor (128 px)"),
            ("xception_ext", "Xception extractor (128 px)"),
        ],
    ),
]:
    f, axs = plt.subplots(2, 3, figsize=(7, 4.2), sharex="col")
    for j, (fn, title) in enumerate(names):
        h = pd.read_csv(os.path.join(HISTORY, fn + "_history.csv"))
        e = h.epoch + 1
        axs[0, j].plot(e, h.accuracy, color="#3B6FB6", label="train")
        axs[0, j].plot(e, h.val_accuracy, color="#E07B39", label="validation")
        axs[1, j].plot(e, h.loss, color="#3B6FB6")
        axs[1, j].plot(e, h.val_loss, color="#E07B39")
        axs[0, j].set_title(title)
        axs[1, j].set_xlabel("Epoch")
        numbers.setdefault("training", {})[fn] = {
            "epochs_recorded": int(len(h)),
            "train_acc": round(float(h.accuracy.iloc[-1]), 4),
            "val_acc": round(float(h.val_accuracy.iloc[-1]), 4),
            "train_loss": round(float(h.loss.iloc[-1]), 4),
            "val_loss": round(float(h.val_loss.iloc[-1]), 4),
            "min_val_loss": round(float(h.val_loss.min()), 4),
            "min_val_loss_epoch": int(h.epoch[h.val_loss.idxmin()] + 1),
        }
    axs[0, 0].set_ylabel("Accuracy")
    axs[1, 0].set_ylabel("Loss")
    axs[0, 0].legend(fontsize=8)
    save(f, f"fig_training_{group}.png")


# ---- Figures 10 to 12: row-normalised confusion matrices
def cmplot(keys, name, ncol):
    nrow = int(np.ceil(len(keys) / ncol))
    f, axs = plt.subplots(nrow, ncol, figsize=(1.75 * ncol, 1.8 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for a, k in zip(axs, keys):
        m = pd.read_csv(os.path.join(CONFUSION, k + "_confusion_matrix.csv"), index_col=0).values
        n = m / m.sum(1, keepdims=True)
        a.imshow(n, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                a.text(
                    j,
                    i,
                    f"{n[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if n[i, j] > 0.6 else "black",
                )
        a.set_xticks(range(4), ["G", "M", "N", "P"], fontsize=7)
        a.set_yticks(range(4), ["G", "M", "N", "P"], fontsize=7)
        title = (
            NICE[k]
            .replace("Final 6-model confidence-weighted vote", "Final: conf.-weighted")
            .replace("Final 6-model majority vote", "Final: majority vote")
            .replace("InceptionV3 feat.", "IncV3 feat.")
            .replace("Xception feat.", "Xcp feat.")
        )
        a.set_title(title, fontsize=7.5)
        a.set_xlabel("Predicted", fontsize=7)
        a.set_ylabel("True", fontsize=7)
    for a in axs[len(keys) :]:
        a.axis("off")
    f.tight_layout()
    save(f, name)


cmplot(
    [
        "CNN_standalone",
        "InceptionV3_standalone",
        "Xception_standalone",
        "FINAL_majority_vote",
        "FINAL_confidence_weighted",
    ],
    "fig_cm_deep_final.png",
    5,
)
cmplot([k for k in NICE if "_feat_" in k], "fig_cm_feature_ensembles.png", 4)
cmplot([k for k in NICE if k.startswith("raw_")], "fig_cm_raw.png", 5)

# ---- Figure 9: accuracy of all 22 configurations
group = lambda k: (
    "Ens" if k.startswith("FINAL") else "Deep" if k.endswith("standalone") else "Feat" if "_feat_" in k else "Raw"
)
COL = {"Deep": "#3B6FB6", "Raw": "#9AA5B8", "Feat": "#E0A458", "Ens": "#B5487F"}
d = master[master.model.isin(NICE)].sort_values("accuracy")
f, ax = plt.subplots(figsize=(6.6, 6.4))
ax.barh([NICE[k] for k in d.model], 100 * d.accuracy, color=[COL[group(k)] for k in d.model])
for i, v in enumerate(100 * d.accuracy):
    ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=7.5)
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
save(f, "fig_accuracy_bars.png")

# ---- Figure 13: per-class F1
fm = master.set_index("model")[["f1_glioma", "f1_meningioma", "f1_notumor", "f1_pituitary"]].loc[list(NICE)]
f, ax = plt.subplots(figsize=(5.2, 6.2))
im = ax.imshow(fm.values, cmap="viridis", vmin=0.65, vmax=1, aspect="auto")
for i in range(len(fm)):
    for j in range(4):
        ax.text(
            j,
            i,
            f"{fm.values[i, j]:.3f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if fm.values[i, j] < 0.85 else "black",
        )
ax.set_xticks(range(4), LABELS)
ax.set_yticks(range(len(fm)), [NICE[k] for k in fm.index], fontsize=7.5)
plt.colorbar(im, ax=ax, shrink=0.6, label="F1-score")
save(f, "fig_f1_heatmap.png")

# ---- Figure 14 / Table 18: ROC and precision-recall (probabilities from 1_test_set_probabilities.py)
P = np.load(os.path.join(OUT, "standalone_test_probs.npz"))
yt = P["y"]
numbers["roc_pr"] = {}
f, axs = plt.subplots(2, 3, figsize=(7, 4.6))
for j, nm in enumerate(["CNN", "InceptionV3", "Xception"]):
    p = P[nm]
    numbers["roc_pr"][nm] = {}
    for c in range(4):
        fpr, tpr, _ = roc_curve(yt == c, p[:, c])
        pr, rc, _ = precision_recall_curve(yt == c, p[:, c])
        a, ap = auc(fpr, tpr), average_precision_score(yt == c, p[:, c])
        numbers["roc_pr"][nm][CLASSES[c]] = {"roc_auc": round(a, 4), "average_precision": round(ap, 4)}
        axs[0, j].plot(fpr, tpr, color=PALETTE[c], lw=1.2, label=f"{LABELS[c]} {a:.3f}")
        axs[1, j].plot(rc, pr, color=PALETTE[c], lw=1.2, label=f"{LABELS[c]} {ap:.3f}")
    numbers["roc_pr"][nm]["macro_roc_auc"] = round(float(roc_auc_score(yt, p, multi_class="ovr", average="macro")), 4)
    axs[0, j].plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
    axs[0, j].set_title(nm)
    axs[0, j].legend(fontsize=6.5, title="ROC AUC", title_fontsize=6.5)
    axs[1, j].legend(fontsize=6.5, title="Average precision", title_fontsize=6.5, loc="lower left")
    axs[0, j].set_xlabel("False positive rate")
    axs[1, j].set_xlabel("Recall")
axs[0, 0].set_ylabel("True positive rate")
axs[1, 0].set_ylabel("Precision")
f.tight_layout()
save(f, "fig_roc_pr.png")

# ---- Section 4.4: the six-predictor ensemble (Tables 14 and 15)
members = ["CNN", "InceptionV3", "Xception", "CNN_ensemble", "InceptionV3_ensemble", "Xception_ensemble"]
votes = np.stack([pred["pred_" + m].values for m in members])
nagree = np.array([np.bincount(votes[:, i], minlength=4).max() for i in range(len(y0))])
numbers["member_accuracy_in_ensemble_pipeline"] = {
    m: round(100 * float((votes[k] == y0).mean()), 2) for k, m in enumerate(members)
}
numbers["votes_for_winning_class"] = {
    f"{k}_of_6": {
        "images": int((nagree == k).sum()),
        "majority_vote_accuracy_pct": round(100 * float((pred.pred_majority.values == y0)[nagree == k].mean()), 2),
    }
    for k in range(2, 7)
    if (nagree == k).any()
}
numbers["all_six_wrong"] = int((~(votes == y0).any(0)).sum())
diff = pred.pred_majority != pred.pred_confidence_weighted
numbers["voting_rules_disagree"] = {
    "images": int(diff.sum()),
    "majority_right": int((pred.pred_majority[diff] == pred.true[diff]).sum()),
    "confidence_weighted_right": int((pred.pred_confidence_weighted[diff] == pred.true[diff]).sum()),
}
z = np.load(os.path.join(METRICS, "final_probs.npz"))
names = [str(n) for n in z["names"]]
order = [names.index(m) for m in members]
numbers["ensemble_build_up_pct"] = {
    " + ".join(members[: k + 1]): round(100 * float((z["P"][order[: k + 1]].sum(0).argmax(1) == z["y"]).mean()), 2)
    for k in range(6)
}

# ---- Figure 18 and Tables 20 to 24: EAA-IoU on the whole test set
e = pd.read_csv(os.path.join(METRICS, "eaa_iou_per_image.csv")).drop_duplicates("idx").sort_values("idx")
f, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
axs[0].boxplot(
    [e[e.true == c].eaa_iou for c in range(4)],
    tick_labels=["G", "M", "N", "P"],
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor="#DCE6F2"),
)
axs[0].set_title("EAA-IoU by true class")
axs[0].set_ylabel("EAA-IoU")
axs[1].boxplot(
    [e[e.correct == 1].eaa_iou, e[e.correct == 0].eaa_iou],
    tick_labels=["Correct", "Misclassified"],
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor="#DCE6F2"),
)
axs[1].set_title("Correct vs misclassified")
fpr, tpr, _ = roc_curve(1 - e.correct, -e.eaa_iou)
axs[2].plot(fpr, tpr, color="#B5487F")
axs[2].plot([0, 1], [0, 1], ":", color="grey")
axs[2].set_title(f"Low EAA-IoU as error flag (AUROC {auc(fpr, tpr):.3f})")
axs[2].set_xlabel("False positive rate")
axs[2].set_ylabel("True positive rate")
f.tight_layout()
save(f, "fig_eaa_iou.png")

err = e.correct.values == 0
conf = pred.sort_values("idx").ensemble_confidence.values
numbers["eaa"] = {
    "mean_all": round(float(e.eaa_iou.mean()), 4),
    "mean_correct": round(float(e[~err].eaa_iou.mean()), 4),
    "mean_misclassified": round(float(e[err].eaa_iou.mean()), 4),
    "median_correct": round(float(e[~err].eaa_iou.median()), 4),
    "median_misclassified": round(float(e[err].eaa_iou.median()), 4),
    "max": round(float(e.eaa_iou.max()), 4),
    "share_zero_pct": round(100 * float((e.eaa_iou == 0).mean()), 2),
    "auroc_low_eaa_predicts_error": round(float(auc(fpr, tpr)), 4),
    "auroc_low_confidence_predicts_error": round(float(roc_auc_score(err, -conf)), 4),
    "pair_means": {c: round(float(e[c].mean()), 4) for c in ["iou_xcp_inc", "iou_cnn_xcp", "iou_cnn_inc"]},
    "by_true_class": {
        CLASSES[c]: {
            "mean_eaa": round(float(e[e.true == c].eaa_iou.mean()), 4),
            "xcp_inc_iou": round(float(e[e.true == c].iou_xcp_inc.mean()), 4),
        }
        for c in range(4)
    },
}
numbers["thresholds"] = {}
for t in [0.05, 0.10, 0.20, 0.40]:
    flag = e.eaa_iou.values < t
    numbers["thresholds"][str(t)] = {
        "flagged_pct": round(100 * float(flag.mean()), 2),
        "errors_caught_pct": round(100 * float((flag & err).sum() / err.sum()), 2),
        "error_rate_flagged_pct": round(100 * float(err[flag].mean()), 2) if flag.any() else None,
        "error_rate_unflagged_pct": round(100 * float(err[~flag].mean()), 2) if (~flag).any() else None,
    }
numbers["bands"] = []
for lo, hi, label in [(0, 0.10, "below 0.10"), (0.10, 0.20, "0.10 to 0.20"), (0.20, 1.01, "0.20 or above")]:
    s = (e.eaa_iou.values >= lo) & (e.eaa_iou.values < hi)
    numbers["bands"].append(
        {
            "band": label,
            "images": int(s.sum()),
            "share_pct": round(100 * float(s.mean()), 2),
            "errors": int((s & err).sum()),
            "error_rate_pct": round(100 * float(err[s].mean()), 2),
            "share_of_errors_pct": round(100 * float((s & err).sum() / err.sum()), 2),
        }
    )
k = int((e.eaa_iou.values < 0.10).sum())  # same number of images flagged by the lowest ensemble confidence
low_conf = np.zeros(len(conf), bool)
low_conf[np.argsort(conf)[:k]] = True
numbers["confidence_flag_same_rate_errors_caught_pct"] = round(100 * float((low_conf & err).sum() / err.sum()), 2)

json.dump(numbers, open(os.path.join(OUT, "report_numbers.json"), "w"), indent=2)
print(json.dumps(numbers, indent=1))
