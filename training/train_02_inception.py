"""
train_02_inception.py — InceptionV3 Transfer Learning

Produces three files in MODELS/:
  InceptionV3.h5           Keras InceptionV3, 256×256 input, direct classifier
  inc_ensemble.h5          Keras InceptionV3, 128×128, feature-extraction backbone
  inc_ensemble_model.pkl   scikit-learn VotingClassifier (RF+DT+SVM) on InceptionV3 features

Run:
    cd training
    python train_02_inception.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
from tensorflow.keras.utils import to_categorical

from config import (
    MODELS_DIR,
    SIZE_STANDALONE, SIZE_ENSEMBLE,
    EPOCHS_TRANSFER,
    BATCH_STANDALONE, BATCH_ENSEMBLE,
    INC_FEAT_LAYER,
    INC_STANDALONE_FILE, INC_FEATURE_FILE, INC_ENSEMBLE_FILE,
)
from data import get_tf_dataset, load_dataset_numpy
from models.transfer import build_inception_standalone, build_inception_extractor
from models.classical import build_classical_ensemble
from evaluate import print_results, plot_training_history, extract_features, plot_confusion_matrix

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. InceptionV3 Standalone (256×256)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 1/3  InceptionV3 Standalone (256×256) ===')
train_ds, val_ds = get_tf_dataset('training', SIZE_STANDALONE, BATCH_STANDALONE)
test_ds, _       = get_tf_dataset('testing',  SIZE_STANDALONE, BATCH_STANDALONE)

inc_standalone = build_inception_standalone(SIZE_STANDALONE)
inc_standalone.summary()

history_standalone = inc_standalone.fit(
    train_ds,
    epochs=EPOCHS_TRANSFER,
    validation_data=val_ds,
    verbose=1,
)
plot_training_history(history_standalone, 'InceptionV3 Standalone (256×256)')

y_true, y_pred = [], []
for X_batch, y_batch in test_ds:
    preds = np.argmax(inc_standalone.predict(X_batch, verbose=0), axis=1)
    y_pred.extend(preds)
    y_true.extend(y_batch.numpy())

print_results(y_true, y_pred, 'InceptionV3 Standalone — Test Set')
plot_confusion_matrix(y_true, y_pred, 'InceptionV3 Standalone')

inc_standalone.save(os.path.join(MODELS_DIR, INC_STANDALONE_FILE))
print(f'Saved: {INC_STANDALONE_FILE}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. InceptionV3 Feature Extractor (128×128)
# Uses one-hot labels because loss='categorical_crossentropy'
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 2/3  InceptionV3 Feature Extractor (128×128) ===')

X_train, y_train = load_dataset_numpy('training', SIZE_ENSEMBLE)
X_test,  y_test  = load_dataset_numpy('testing',  SIZE_ENSEMBLE)

inc_extractor = build_inception_extractor(SIZE_ENSEMBLE)
inc_extractor.summary()

history_extractor = inc_extractor.fit(
    X_train, to_categorical(y_train),
    batch_size=BATCH_ENSEMBLE,
    epochs=EPOCHS_TRANSFER,
    validation_data=(X_test, to_categorical(y_test)),
    verbose=1,
)
plot_training_history(history_extractor, 'InceptionV3 Feature Extractor (128×128)')

inc_extractor.save(os.path.join(MODELS_DIR, INC_FEATURE_FILE))
print(f'Saved: {INC_FEATURE_FILE}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Classical ML Ensemble on InceptionV3 Features
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 3/3  Classical Ensemble on InceptionV3 Features ===')

train_feats = extract_features(inc_extractor, INC_FEAT_LAYER, X_train)
test_feats  = extract_features(inc_extractor, INC_FEAT_LAYER, X_test)

print('Fitting RF + DT + SVM ensemble...')
ensemble = build_classical_ensemble()
ensemble.fit(train_feats, y_train)

for name, clf in ensemble.named_estimators_.items():
    preds = clf.predict(test_feats)
    acc   = (preds == y_test).mean() * 100
    print(f'  {name.upper():4s}  {acc:.2f}%')

pred_ens = ensemble.predict(test_feats)
print_results(y_test, pred_ens, 'InceptionV3 Ensemble (RF+DT+SVM) — Test Set')
plot_confusion_matrix(y_test, pred_ens, 'InceptionV3 Classical Ensemble')

joblib.dump(ensemble, os.path.join(MODELS_DIR, INC_ENSEMBLE_FILE))
print(f'Saved: {INC_ENSEMBLE_FILE}')

print('\n✓  train_02_inception.py complete')
