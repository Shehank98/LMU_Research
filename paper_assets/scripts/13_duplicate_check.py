"""Find test images that also appear (possibly re-encoded) in the Training folder.

Candidate pairs share the same 16x16 average hash; a pair counts as a duplicate when the mean absolute
pixel difference at 64x64 (grey, 0-1 scale) is below 0.02.  The result is a 'clean' test subset used to
report accuracy without leakage.
"""

import os, sys, json, numpy as np, cv2

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, CLASSES, ROOT


def small(path, size=64):
    grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA).astype("float32") / 255


def ahash(img64):
    g = cv2.resize(img64, (16, 16), interpolation=cv2.INTER_AREA)
    return (g > g.mean()).tobytes()


train = {}
for c in CLASSES:
    for name in os.listdir(os.path.join(DATA, "Training", c)):
        img = small(os.path.join(DATA, "Training", c, name))
        train.setdefault(ahash(img), []).append((c, img))

rows, idx = [], 0
for label, c in enumerate(CLASSES):
    for name in sorted(os.listdir(os.path.join(DATA, "Testing", c))):
        img = small(os.path.join(DATA, "Testing", c, name))
        matches = train.get(ahash(img), [])
        diff = min((np.abs(img - t).mean() for _, t in matches), default=1.0)
        rows.append(
            dict(
                idx=idx,
                file=f"{c}/{name}",
                true=label,
                min_diff=round(float(diff), 4),
                duplicate=int(diff < 0.02),
                label_conflict=int(
                    diff < 0.02 and all(tc != c for tc, t in matches if np.abs(img - t).mean() < 0.02)
                ),
            )
        )
        idx += 1

import pandas as pd

df = pd.DataFrame(rows)
out_dir = os.path.join(ROOT, "metrics")
df.to_csv(os.path.join(out_dir, "test_duplicates.csv"), index=False)
summary = {
    "threshold": 0.02,
    "n_test": int(len(df)),
    "n_duplicates": int(df.duplicate.sum()),
    "n_clean": int((df.duplicate == 0).sum()),
    "by_class": {c: int(df[df.true == i].duplicate.sum()) for i, c in enumerate(CLASSES)},
    "label_conflicts": int(df.label_conflict.sum()),
}
json.dump(summary, open(os.path.join(out_dir, "test_duplicates_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=1))
