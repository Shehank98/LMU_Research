"""EAA-IoU (mean pairwise IoU of the binarised Grad-CAM maps of CNN, Xception and InceptionV3) for every test image.
For the baseline this was computed during training (metrics/eaa_iou_per_image.csv); this script does the same
for the improved models and writes the results to paper_assets/xai_results_improved/."""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(__file__))
from common import *
from xai_lib import CamModel, gradcam, load, TARGET


def iou(a, b, t=0.5):
    A, B = a >= t, b >= t
    union = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum() / union) if union else 0.0


pred = pd.read_csv(PRED_CSV)
X, y, _ = load_numpy("Testing", IMG)
cls = pred.pred_confidence_weighted.values
cams = {n: CamModel(load(n), TARGET[n]) for n in ["CNN", "Xception", "InceptionV3"]}
rows = []
for s in range(0, len(X), 16):
    maps = {n: gradcam(c, X[s : s + 16], cls[s : s + 16]) for n, c in cams.items()}
    for k in range(len(maps["CNN"])):
        i = s + k
        p = [iou(maps["CNN"][k], maps["Xception"][k]), iou(maps["CNN"][k], maps["InceptionV3"][k]),
             iou(maps["Xception"][k], maps["InceptionV3"][k])]
        rows.append(dict(idx=i, true=int(y[i]), pred_final=int(cls[i]), correct=int(y[i] == cls[i]), iou_cnn_xcp=p[0],
                         iou_cnn_inc=p[1], iou_xcp_inc=p[2], eaa_iou=float(np.mean(p))))
d = pd.DataFrame(rows).round(4)
d.to_csv(os.path.join(RES, "eaa_iou_per_image.csv"), index=False)
right, wrong = d[d.correct == 1], d[d.correct == 0]
summary = {"n_images": len(d), "n_misclassified": int(len(wrong)), "mean_eaa_all": round(float(d.eaa_iou.mean()), 4),
           "mean_eaa_correct": round(float(right.eaa_iou.mean()), 4),
           "mean_eaa_misclassified": round(float(wrong.eaa_iou.mean()), 4) if len(wrong) else None,
           "mean_eaa_by_true_class": {CLASSES[int(k)]: round(float(v), 4) for k, v in d.groupby("true").eaa_iou.mean().items()},
           "pair_means": {c: round(float(d[c].mean()), 4) for c in ["iou_cnn_xcp", "iou_cnn_inc", "iou_xcp_inc"]}}
if len(wrong) and len(right):
    summary["auroc_low_eaa_predicts_error"] = round(float(roc_auc_score(1 - d.correct, -d.eaa_iou)), 4)
    summary["mann_whitney_p_value"] = float(mannwhitneyu(right.eaa_iou, wrong.eaa_iou, alternative="greater").pvalue)
    summary["confidence_auroc"] = round(float(roc_auc_score(1 - d.correct, -pred.ensemble_confidence.values)), 4)
for t in [0.05, 0.1, 0.2]:
    flag = d.eaa_iou < t
    summary[f"threshold_{t}"] = {"flagged_pct": round(100 * float(flag.mean()), 2),
                                 "errors_caught_pct": round(100 * float((flag & (d.correct == 0)).sum() / max(len(wrong), 1)), 2)}
json.dump(summary, open(os.path.join(RES, "eaa_iou_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=1))
