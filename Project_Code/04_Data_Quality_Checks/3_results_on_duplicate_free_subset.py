"""Accuracy of every predictor on the full test set and on the duplicate-free ('clean') subset,
with bootstrap intervals and McNemar tests between the baseline and improved final ensembles."""

import os, sys, json, numpy as np, pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from project_settings import ROOT, CLASSES

dup = pd.read_csv(os.path.join(ROOT, "metrics", "test_duplicates.csv"))
clean = dup.duplicate.values == 0
rng = np.random.default_rng(42)


def summary(pred, y, mask):
    c = (pred == y)[mask]
    boot = rng.integers(0, len(c), (2000, len(c)))
    lo, hi = np.percentile(c[boot].mean(1), [2.5, 97.5])
    rec = {
        k: round(100 * float(((pred == y) & (y == i))[mask].sum() / max(((y == i) & mask).sum(), 1)), 2)
        for i, k in enumerate(CLASSES)
    }
    return {
        "acc": round(100 * float(c.mean()), 2),
        "ci95": [round(100 * lo, 2), round(100 * hi, 2)],
        "n": int(mask.sum()),
        "recall": rec,
    }


out = {}
base = pd.read_csv(os.path.join(ROOT, "metrics", "final_per_image_predictions.csv"))
cols = {
    "CNN": "pred_CNN",
    "InceptionV3": "pred_InceptionV3",
    "Xception": "pred_Xception",
    "CNN_features": "pred_CNN_ensemble",
    "InceptionV3_features": "pred_InceptionV3_ensemble",
    "Xception_features": "pred_Xception_ensemble",
    "final_majority": "pred_majority",
    "final_confidence_weighted": "pred_confidence_weighted",
}
y = base.true.values
out["baseline"] = {
    k: {"full": summary(base[c].values, y, np.ones(len(y), bool)), "clean": summary(base[c].values, y, clean)}
    for k, c in cols.items()
}
imp_path = os.path.join(ROOT, "runs", "v2", "metrics", "per_image_predictions.csv")
if os.path.exists(imp_path):
    imp = pd.read_csv(imp_path)
    assert (imp.true.values == y).all()
    icols = {
        "CNN": "pred_cnn",
        "InceptionV3": "pred_inceptionv3",
        "Xception": "pred_xception",
        "CNN_features": "pred_cnn_feat_softvote",
        "InceptionV3_features": "pred_inceptionv3_feat_softvote",
        "Xception_features": "pred_xception_feat_softvote",
        "final_majority": "pred_majority",
        "final_confidence_weighted": "pred_confidence_weighted",
    }
    out["improved"] = {
        k: {
            "full": summary(imp[c].values, y, np.ones(len(y), bool)),
            "clean": summary(imp[c].values, y, clean),
        }
        for k, c in icols.items()
    }
    tests = []
    pairs = [
        (
            "improved vs baseline majority (full)",
            imp.pred_majority,
            base.pred_majority,
            np.ones(len(y), bool),
        ),
        ("improved vs baseline majority (clean)", imp.pred_majority, base.pred_majority, clean),
    ]
    pairs += [
        (f"improved majority vs improved {k} (full)", imp.pred_majority, imp[c], np.ones(len(y), bool))
        for k, c in icols.items()
        if not k.startswith("final")
    ]
    for name, a, b, mask in pairs:
        ca, cb = (a.values == y)[mask], (b.values == y)[mask]
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        tests.append(
            {
                "test": name,
                "n01": n01,
                "n10": n10,
                "p": float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0,
            }
        )
    ps = np.array([t["p"] for t in tests])
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    adj[order] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * ps[order]))
    for t, a_ in zip(tests, adj):
        t["p_holm"] = float(a_)
    out["mcnemar_improved"] = tests
json.dump(out, open(os.path.join(ROOT, "metrics", "clean_subset_results.json"), "w"), indent=1)
print(
    json.dumps(
        {
            k: {m: (v["full"]["acc"], v["clean"]["acc"]) for m, v in d.items()}
            for k, d in out.items()
            if k != "mcnemar_improved"
        },
        indent=1,
    )
)
