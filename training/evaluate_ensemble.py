"""
evaluate_ensemble.py — Final 6-model confidence-weighted ensemble

Loads all 9 model files from MODELS/, runs the same logic as webapp/xai_engine.py
across the entire official test set, and prints per-model and combined results.

Run AFTER all three training scripts have completed:
    cd training
    python evaluate_ensemble.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model

from config import (
    MODELS_DIR,
    SIZE_STANDALONE, SIZE_ENSEMBLE,
    CNN_FEAT_LAYER, INC_FEAT_LAYER, XCP_FEAT_LAYER,
    CNN_STANDALONE_FILE, CNN_FEATURE_FILE, CNN_ENSEMBLE_FILE,
    INC_STANDALONE_FILE, INC_FEATURE_FILE, INC_ENSEMBLE_FILE,
    XCP_STANDALONE_FILE, XCP_FEATURE_FILE, XCP_ENSEMBLE_FILE,
    N_CLASSES,
)
from data import load_dataset_numpy
from evaluate import print_results, plot_confusion_matrix
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading test data...')
X_256, y_test = load_dataset_numpy('testing', SIZE_STANDALONE)
X_128, _      = load_dataset_numpy('testing', SIZE_ENSEMBLE)

# ── Load all models ───────────────────────────────────────────────────────────
def load(filename, fmt='keras'):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f'  [SKIP] {filename} not found')
        return None
    print(f'  Loading {filename}...')
    return load_model(path) if fmt == 'keras' else joblib.load(path)

print('\nLoading models from MODELS/...')
cnn_standalone  = load(CNN_STANDALONE_FILE,  'keras')
cnn_feat_model  = load(CNN_FEATURE_FILE,     'keras')
cnn_ens_model   = load(CNN_ENSEMBLE_FILE,    'joblib')
inc_standalone  = load(INC_STANDALONE_FILE,  'keras')
inc_feat_model  = load(INC_FEATURE_FILE,     'keras')
inc_ens_model   = load(INC_ENSEMBLE_FILE,    'joblib')
xcp_standalone  = load(XCP_STANDALONE_FILE,  'keras')
xcp_feat_model  = load(XCP_FEATURE_FILE,     'keras')
xcp_ens_model   = load(XCP_ENSEMBLE_FILE,    'joblib')


def get_feat_extractor(backbone, layer_name):
    if backbone is None:
        return None
    return Model(inputs=backbone.input,
                 outputs=backbone.get_layer(layer_name).output)


cnn_extractor = get_feat_extractor(cnn_feat_model, CNN_FEAT_LAYER)
inc_extractor = get_feat_extractor(inc_feat_model, INC_FEAT_LAYER)
xcp_extractor = get_feat_extractor(xcp_feat_model, XCP_FEAT_LAYER)


# ── Per-model predictions ─────────────────────────────────────────────────────
def safe_predict_deep(model, X):
    if model is None:
        return None
    return model.predict(X, batch_size=32, verbose=0)   # (N, 4) softmax probs


def safe_predict_classical(extractor, clf, X):
    if extractor is None or clf is None:
        return None
    feats = extractor.predict(X, batch_size=64, verbose=0)
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(feats)   # (N, 4)
    preds = clf.predict(feats)
    return np.eye(N_CLASSES)[preds]


print('\nRunning per-model predictions...')
per_model = {
    'CNN (standalone)':          safe_predict_deep(cnn_standalone, X_256),
    'InceptionV3 (standalone)':  safe_predict_deep(inc_standalone, X_256),
    'Xception (standalone)':     safe_predict_deep(xcp_standalone, X_256),
    'CNN + Ensemble':            safe_predict_classical(cnn_extractor, cnn_ens_model, X_128),
    'InceptionV3 + Ensemble':    safe_predict_classical(inc_extractor, inc_ens_model, X_128),
    'Xception + Ensemble':       safe_predict_classical(xcp_extractor, xcp_ens_model, X_128),
}

# ── Individual model results ──────────────────────────────────────────────────
print('\n\n══ Per-Model Results ══')
for name, probs in per_model.items():
    if probs is None:
        print(f'  {name}: SKIPPED (model file missing)')
        continue
    preds = np.argmax(probs, axis=1)
    print_results(y_test, preds, name)

# ── Confidence-weighted ensemble (sum of probability vectors) ─────────────────
print('\n\n══ 6-Model Confidence-Weighted Ensemble ══')
active = [p for p in per_model.values() if p is not None]
if not active:
    print('No models loaded — cannot run ensemble.')
    sys.exit(1)

combined_probs = sum(active)
ensemble_preds = np.argmax(combined_probs, axis=1)
print_results(y_test, ensemble_preds, 'FINAL ENSEMBLE')

# ── Confusion matrix ──────────────────────────────────────────────────────────
plot_confusion_matrix(y_test, ensemble_preds, 'Final 6-Model Ensemble')

print('\n✓  evaluate_ensemble.py complete')
