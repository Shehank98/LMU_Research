"""
train_04_classical_raw.py — SVM / RF / DT on raw pixels (comparison baseline)

Trains classical ML on flattened 256×256 greyscale pixels — no deep features.
Results are for research comparison only; this script does NOT produce webapp model files.

Expected results (from research report):
  SVM   88.26%
  DT    80.02%
  RF    79.52%  (was previously broken — used DecisionTree instead)
  LR    84.00%
  Ensemble (hard vote, DT+RF+SVM)  83%

Run:
    cd training
    python train_04_classical_raw.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

from config import DATA_DIR, CLASS_NAMES
from evaluate import print_results, plot_confusion_matrix

SIZE = 256

# ── Load greyscale data ────────────────────────────────────────────────────────
def load_greyscale(split):
    label_to_id = {cls: i for i, cls in enumerate(sorted(CLASS_NAMES))}
    directory   = os.path.join(DATA_DIR, split.capitalize())
    X, y = [], []
    for cls_dir in sorted(glob.glob(os.path.join(directory, '*'))):
        cls = os.path.basename(cls_dir)
        if cls not in label_to_id:
            continue
        for img_path in glob.glob(os.path.join(cls_dir, '*')):
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(cv2.resize(img, (SIZE, SIZE)).flatten())
            y.append(label_to_id[cls])
    return np.array(X, dtype=np.float32) / 255.0, np.array(y, dtype=np.int32)

print('Loading training data (greyscale, 256×256 flattened)...')
X_train, y_train = load_greyscale('training')
print('Loading test data...')
X_test, y_test = load_greyscale('testing')
print(f'Train: {X_train.shape}  Test: {X_test.shape}')

# ── Models ────────────────────────────────────────────────────────────────────
results = {}

print('\n[1/5] Logistic Regression...')
lr = LogisticRegression(C=0.1, max_iter=1000)
lr.fit(X_train, y_train)
results['LR'] = lr.predict(X_test)

print('[2/5] SVM...')
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
results['SVM'] = svm.predict(X_test)

print('[3/5] Decision Tree...')
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
results['DT'] = dt.predict(X_test)

print('[4/5] Random Forest...')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
results['RF'] = rf.predict(X_test)

print('[5/5] Hard-Vote Ensemble (DT + RF + SVM)...')
ens = VotingClassifier(
    estimators=[('dt', dt), ('rf', rf), ('svm', svm)],
    voting='hard',
)
ens.fit(X_train, y_train)
results['Ensemble'] = ens.predict(X_test)

# ── Results ───────────────────────────────────────────────────────────────────
print('\n\n══ Classical ML on Raw Pixels — Summary ══')
for name, preds in results.items():
    print_results(y_test, preds, name)

fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
for ax, (name, preds) in zip(axes, results.items()):
    plot_confusion_matrix(y_test, preds, name, ax=ax)
plt.suptitle('Classical ML on Raw Pixels — Confusion Matrices')
plt.tight_layout()
plt.show()

print('\n✓  train_04_classical_raw.py complete')
