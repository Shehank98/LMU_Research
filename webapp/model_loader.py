import os
import joblib
import streamlit as st
from tensorflow.keras.models import load_model, Model

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'MODELS')

HF_REPO_ID = os.environ.get('HF_REPO_ID', '')
HF_TOKEN   = os.environ.get('HF_TOKEN', '')

MODEL_FILES = {
    'cnn':       ('cnn_standalone.h5',       'keras'),
    'inception': ('InceptionV3.h5',          'keras'),
    'xception':  ('Xception.h5',             'keras'),
    'cnn_feat':  ('cnn_ensemble.h5',         'keras'),
    'inc_feat':  ('inc_ensemble.h5',         'keras'),
    'xcp_feat':  ('xcp_ensemble.h5',         'keras'),
    'cnn_ens':   ('cnn_ensemble_model.pkl',  'joblib'),
    'inc_ens':   ('inc_ensemble_model.pkl',  'joblib'),
    'xcp_ens':   ('xcp_ensemble_model.pkl',  'joblib'),
}

FEATURE_LAYERS = {
    'cnn_feat': 'dense',
    'inc_feat': 'dense_2',
    'xcp_feat': 'dense_3',
}


def ensure_models_downloaded():
    """Download any missing model files from HuggingFace Hub if HF_REPO_ID is set."""
    if not HF_REPO_ID:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    for key, (filename, _) in MODEL_FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(dest):
            try:
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    token=HF_TOKEN or None,
                    local_dir=MODELS_DIR,
                )
            except Exception:
                pass


@st.cache_resource(show_spinner="Loading models — this takes a moment…")
def load_all_models():
    ensure_models_downloaded()

    models  = {}
    missing = []

    for key, (filename, loader) in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            missing.append(filename)
            models[key] = None
            continue
        try:
            if loader == 'keras':
                models[key] = load_model(path)
            else:
                models[key] = joblib.load(path)
        except Exception:
            missing.append(filename)
            models[key] = None

    # Build feature-extractor sub-models
    extractors = {}
    for key, layer_name in FEATURE_LAYERS.items():
        if models.get(key) is not None:
            try:
                extractors[key] = Model(
                    inputs=models[key].input,
                    outputs=models[key].get_layer(layer_name).output,
                )
            except Exception:
                pass

    return models, extractors, missing
