"""
train_01_cnn.py — Custom CNN

Produces three files in MODELS/:
  cnn_standalone.h5        Keras CNN, 256×256 input, direct 4-class classifier
  cnn_ensemble.h5          Keras CNN, 128×128 input, feature-extraction backbone
  cnn_ensemble_model.pkl   scikit-learn VotingClassifier (RF+DT+SVM) on CNN features

Run:
    cd training
    python train_01_cnn.py
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np

from config import (
    MODELS_DIR, RANDOM_SEED,
    SIZE_STANDALONE, SIZE_ENSEMBLE,
    EPOCHS_CNN_STANDALONE, EPOCHS_CNN_ENSEMBLE,
    BATCH_STANDALONE, BATCH_ENSEMBLE,
    CNN_FEAT_LAYER,
    CNN_STANDALONE_FILE, CNN_FEATURE_FILE, CNN_ENSEMBLE_FILE,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)
from data import get_tf_dataset, load_dataset_numpy
from models.cnn import build_cnn
from models.classical import build_classical_ensemble
from evaluate import print_results, plot_training_history, extract_features, plot_confusion_matrix
import matplotlib.pyplot as plt

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CNN Standalone (256×256)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 1/3  CNN Standalone (256×256) ===')
train_ds, val_ds = get_tf_dataset('training', SIZE_STANDALONE, BATCH_STANDALONE)
test_ds, _       = get_tf_dataset('testing',  SIZE_STANDALONE, BATCH_STANDALONE)

cnn_standalone = build_cnn(SIZE_STANDALONE, feat_layer_name='dense')
cnn_standalone.summary()

history_standalone = cnn_standalone.fit(
    train_ds,
    epochs=EPOCHS_CNN_STANDALONE,
    validation_data=val_ds,
    verbose=1,
)
plot_training_history(history_standalone, 'CNN Standalone (256×256)')

# Evaluate on official test set
y_true_standalone, y_pred_standalone = [], []
for X_batch, y_batch in test_ds:
    preds = np.argmax(cnn_standalone.predict(X_batch, verbose=0), axis=1)
    y_pred_standalone.extend(preds)
    y_true_standalone.extend(y_batch.numpy())

print_results(y_true_standalone, y_pred_standalone, 'CNN Standalone — Test Set')
plot_confusion_matrix(y_true_standalone, y_pred_standalone, 'CNN Standalone')

cnn_standalone.save(os.path.join(MODELS_DIR, CNN_STANDALONE_FILE))
print(f'Saved: {CNN_STANDALONE_FILE}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. CNN Feature Extractor (128×128)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 2/3  CNN Feature Extractor (128×128) ===')
train_ds2, val_ds2 = get_tf_dataset('training', SIZE_ENSEMBLE, BATCH_ENSEMBLE)
test_ds2, _        = get_tf_dataset('testing',  SIZE_ENSEMBLE, BATCH_ENSEMBLE)

cnn_extractor = build_cnn(SIZE_ENSEMBLE, feat_layer_name=CNN_FEAT_LAYER)
cnn_extractor.summary()

history_extractor = cnn_extractor.fit(
    train_ds2,
    epochs=EPOCHS_CNN_ENSEMBLE,
    validation_data=val_ds2,
    verbose=1,
)
plot_training_history(history_extractor, 'CNN Feature Extractor (128×128)')

cnn_extractor.save(os.path.join(MODELS_DIR, CNN_FEATURE_FILE))
print(f'Saved: {CNN_FEATURE_FILE}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Classical ML Ensemble on CNN Features
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 3/3  Classical Ensemble on CNN Features ===')
X_train, y_train = load_dataset_numpy('training', SIZE_ENSEMBLE)
X_test,  y_test  = load_dataset_numpy('testing',  SIZE_ENSEMBLE)

train_feats = extract_features(cnn_extractor, CNN_FEAT_LAYER, X_train)
test_feats  = extract_features(cnn_extractor, CNN_FEAT_LAYER, X_test)

print('Fitting RF + DT + SVM ensemble (this may take a few minutes)...')
ensemble = build_classical_ensemble()
ensemble.fit(train_feats, y_train)

# Per-classifier accuracy
for name, clf in ensemble.named_estimators_.items():
    preds = clf.predict(test_feats)
    acc   = (preds == y_test).mean() * 100
    print(f'  {name.upper():4s}  {acc:.2f}%')

pred_ens = ensemble.predict(test_feats)
print_results(y_test, pred_ens, 'CNN Ensemble (RF+DT+SVM) — Test Set')
plot_confusion_matrix(y_test, pred_ens, 'CNN Classical Ensemble')

joblib.dump(ensemble, os.path.join(MODELS_DIR, CNN_ENSEMBLE_FILE))
print(f'Saved: {CNN_ENSEMBLE_FILE}')

print('\n✓  train_01_cnn.py complete')
