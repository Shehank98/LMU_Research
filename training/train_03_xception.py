"""
train_03_xception.py — Xception Transfer Learning

Produces three files in MODELS/:
  Xception.h5              Keras Xception, 256×256 input, direct classifier
  xcp_ensemble.h5          Keras Xception, 128×128, feature-extraction backbone
  xcp_ensemble_model.pkl   scikit-learn VotingClassifier (RF+DT+SVM) on Xception features

Run:
    cd training
    python train_03_xception.py
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from config import (
    MODELS_DIR, RANDOM_SEED,
    SIZE_STANDALONE, SIZE_ENSEMBLE,
    EPOCHS_TRANSFER,
    BATCH_STANDALONE, BATCH_ENSEMBLE,
    XCP_FEAT_LAYER,
    XCP_STANDALONE_FILE, XCP_FEATURE_FILE, XCP_ENSEMBLE_FILE,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)
from data import get_tf_dataset, load_dataset_numpy
from models.transfer import build_xception_standalone, build_xception_extractor
from models.classical import build_classical_ensemble
from evaluate import print_results, plot_training_history, extract_features, plot_confusion_matrix

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Xception Standalone (256×256)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 1/3  Xception Standalone (256×256) ===')
train_ds, val_ds = get_tf_dataset('training', SIZE_STANDALONE, BATCH_STANDALONE)
test_ds, _       = get_tf_dataset('testing',  SIZE_STANDALONE, BATCH_STANDALONE)

xcp_standalone = build_xception_standalone(SIZE_STANDALONE)
xcp_standalone.summary()

history_standalone = xcp_standalone.fit(
    train_ds,
    epochs=EPOCHS_TRANSFER,
    validation_data=val_ds,
    verbose=1,
)
plot_training_history(history_standalone, 'Xception Standalone (256×256)')

y_true, y_pred = [], []
for X_batch, y_batch in test_ds:
    preds = np.argmax(xcp_standalone.predict(X_batch, verbose=0), axis=1)
    y_pred.extend(preds)
    y_true.extend(y_batch.numpy())

print_results(y_true, y_pred, 'Xception Standalone — Test Set')
plot_confusion_matrix(y_true, y_pred, 'Xception Standalone')

xcp_standalone.save(os.path.join(MODELS_DIR, XCP_STANDALONE_FILE))
print(f'Saved: {XCP_STANDALONE_FILE}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Xception Feature Extractor (128×128)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 2/3  Xception Feature Extractor (128×128) ===')

X_train, y_train = load_dataset_numpy('training', SIZE_ENSEMBLE)
X_test,  y_test  = load_dataset_numpy('testing',  SIZE_ENSEMBLE)

# 10% of training data held out as validation (test set is never seen during training)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=RANDOM_SEED, stratify=y_train
)

xcp_extractor = build_xception_extractor(SIZE_ENSEMBLE)
xcp_extractor.summary()

history_extractor = xcp_extractor.fit(
    X_tr, to_categorical(y_tr),
    batch_size=BATCH_ENSEMBLE,
    epochs=EPOCHS_TRANSFER,
    validation_data=(X_val, to_categorical(y_val)),
    verbose=1,
)
plot_training_history(history_extractor, 'Xception Feature Extractor (128×128)')

xcp_extractor.save(os.path.join(MODELS_DIR, XCP_FEATURE_FILE))
print(f'Saved: {XCP_FEATURE_FILE}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Classical ML Ensemble on Xception Features
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 3/3  Classical Ensemble on Xception Features ===')

train_feats = extract_features(xcp_extractor, XCP_FEAT_LAYER, X_train)
test_feats  = extract_features(xcp_extractor, XCP_FEAT_LAYER, X_test)

print('Fitting RF + DT + SVM ensemble...')
ensemble = build_classical_ensemble()
ensemble.fit(train_feats, y_train)

for name, clf in ensemble.named_estimators_.items():
    preds = clf.predict(test_feats)
    acc   = (preds == y_test).mean() * 100
    print(f'  {name.upper():4s}  {acc:.2f}%')

pred_ens = ensemble.predict(test_feats)
print_results(y_test, pred_ens, 'Xception Ensemble (RF+DT+SVM) — Test Set')
plot_confusion_matrix(y_test, pred_ens, 'Xception Classical Ensemble')

joblib.dump(ensemble, os.path.join(MODELS_DIR, XCP_ENSEMBLE_FILE))
print(f'Saved: {XCP_ENSEMBLE_FILE}')

print('\n✓  train_03_xception.py complete')
