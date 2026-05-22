"""
Central configuration — import from here, never hardcode paths or numbers elsewhere.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'MRI_DATASET')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'MODELS')

# ── Classes ───────────────────────────────────────────────────────────────────
# Alphabetical order → matches tf.keras.utils.image_dataset_from_directory default
CLASS_NAMES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
WEBAPP_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
N_CLASSES    = 4

# ── Input sizes ───────────────────────────────────────────────────────────────
SIZE_STANDALONE = 256   # CNN / InceptionV3 / Xception standalone classifiers
SIZE_ENSEMBLE   = 128   # feature extractors that feed classical ML

# ── Architecture ─────────────────────────────────────────────────────────────
FEATURE_DIM = 256       # output units of every feature-extraction Dense layer

# Layer names referenced by webapp/model_loader.py — must not change
CNN_FEAT_LAYER = 'dense'
INC_FEAT_LAYER = 'dense_2'
XCP_FEAT_LAYER = 'dense_3'

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42   # applied to Python random, NumPy, and TensorFlow in every script

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS_CNN_STANDALONE = 20
EPOCHS_CNN_ENSEMBLE   = 10
EPOCHS_TRANSFER       = 10
BATCH_STANDALONE      = 32
BATCH_ENSEMBLE        = 32  # standardised — same batch size across all models

# ── Saved model filenames (all land in MODELS_DIR) ────────────────────────────
CNN_STANDALONE_FILE  = 'cnn_standalone.h5'
CNN_FEATURE_FILE     = 'cnn_ensemble.h5'
CNN_ENSEMBLE_FILE    = 'cnn_ensemble_model.pkl'

INC_STANDALONE_FILE  = 'InceptionV3.h5'
INC_FEATURE_FILE     = 'inc_ensemble.h5'
INC_ENSEMBLE_FILE    = 'inc_ensemble_model.pkl'

XCP_STANDALONE_FILE  = 'Xception.h5'
XCP_FEATURE_FILE     = 'xcp_ensemble.h5'
XCP_ENSEMBLE_FILE    = 'xcp_ensemble_model.pkl'
