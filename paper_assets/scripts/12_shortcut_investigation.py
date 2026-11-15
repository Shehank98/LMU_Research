"""Why does the CNN attend outside the brain?  Three checks on the 1,600 test images:
1. do the classes differ in original image size (a source / framing cue)?
2. how well do the models classify when ONLY the brain, or ONLY the area outside the head, is visible?
3. how well can a trivial model predict the class from image size and border statistics alone?"""

import os, sys, json, numpy as np, pandas as pd, cv2

sys.path.insert(0, os.path.dirname(__file__))
from common import *
from xai_lib import brain_mask, load
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rows = []
for c in CLASSES:
    for fn in sorted(os.listdir(os.path.join(DATA, "Testing", c))):
        im = cv2.imread(os.path.join(DATA, "Testing", c, fn))
        h, w = im.shape[:2]
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        b = 10
        border = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
        rows.append(
            dict(
                cls=c,
                w=w,
                h=h,
                aspect=w / h,
                mean=g.mean(),
                std=g.std(),
                border_mean=border.mean(),
                dark_frac=(g < 20).mean(),
            )
        )
meta = pd.DataFrame(rows)
out = {}
out["size_by_class"] = {
    c: {
        "median_w": int(g.w.median()),
        "median_h": int(g.h.median()),
        "share_512x512_pct": round(100 * float(((g.w == 512) & (g.h == 512)).mean()), 1),
        "n_distinct_sizes": int(g.groupby(["w", "h"]).ngroups),
        "mean_dark_frac": round(float(g.dark_frac.mean()), 3),
    }
    for c, g in meta.groupby("cls")
}
y = meta.cls.map({c: i for i, c in enumerate(CLASSES)}).values
feats = meta[["w", "h", "aspect", "mean", "std", "border_mean", "dark_frac"]].values
out["metadata_rf_cv_accuracy_pct"] = round(
    100 * float(cross_val_score(RandomForestClassifier(200, random_state=42), feats, y, cv=5).mean()), 2
)
out["size_only_rf_cv_accuracy_pct"] = round(
    100 * float(cross_val_score(RandomForestClassifier(200, random_state=42), feats[:, :3], y, cv=5).mean()),
    2,
)
X, yt, _ = load_numpy("Testing", IMG)
masks = np.stack([brain_mask(x) for x in X])
inside = X * masks[..., None]
outside = X * (~masks)[..., None]
for n in ["CNN", "Xception", "InceptionV3"]:
    m = load(n)
    out[n] = {
        k: round(100 * float((m.predict(v, batch_size=32, verbose=0).argmax(1) == yt).mean()), 2)
        for k, v in [("full_image", X), ("head_only", inside), ("outside_head_only", outside)]
    }
    p = m.predict(outside, batch_size=32, verbose=0).argmax(1)
    out[n]["outside_only_recall_by_class"] = {
        c: round(100 * float((p[yt == i] == i).mean()), 1) for i, c in enumerate(CLASSES)
    }
    print(n, out[n], flush=True)
json.dump(out, open(os.path.join(RES, "shortcut_investigation.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
