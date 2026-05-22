import os
import joblib
import streamlit as st
from tensorflow.keras.models import load_model, Model

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'MODELS')

MODEL_FILES = {
    'cnn':         ('cnn_standalone.h5',        'keras'),
    'inception':   ('InceptionV3.h5',          'keras'),
    'xception':    ('Xception.h5',             'keras'),
    'cnn_feat':    ('cnn_ensemble.h5',         'keras'),
    'inc_feat':    ('inc_ensemble.h5',         'keras'),
    'xcp_feat':    ('xcp_ensemble.h5',         'keras'),
    'cnn_ens':     ('cnn_ensemble_model.pkl',  'joblib'),
    'inc_ens':     ('inc_ensemble_model.pkl',  'joblib'),
    'xcp_ens':     ('xcp_ensemble_model.pkl',  'joblib'),
}

FEATURE_LAYERS = {
    'cnn_feat': 'dense',
    'inc_feat': 'dense_2',
    'xcp_feat': 'dense_3',
}

@st.cache_resource(show_spinner="Loading models — this takes a moment...")
def load_all_models():
    models = {}
    missing = []

    for key, (filename, loader) in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            missing.append(filename)
            models[key] = None
            continue
        if loader == 'keras':
            models[key] = load_model(path)
        else:
            models[key] = joblib.load(path)

    # Build feature extractors from loaded backbone models
    extractors = {}
    for key, layer_name in FEATURE_LAYERS.items():
        if models.get(key) is not None:
            extractors[key] = Model(
                inputs=models[key].input,
                outputs=models[key].get_layer(layer_name).output
            )

    return models, extractors, missing
