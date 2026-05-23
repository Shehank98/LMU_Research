import os, threading, joblib, shutil
from tensorflow.keras.models import load_model, Model

_default_models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'MODELS')
MODELS_DIR = os.environ.get('MODELS_DIR', _default_models_dir)
HF_REPO_ID = os.environ.get('HF_REPO_ID', '')
HF_TOKEN   = os.environ.get('HF_TOKEN', '')

# Exact filenames saved by the standardised training notebooks
MODEL_FILES = {
    'cnn':       ('cnn_model.h5',                    'keras'),
    'inception': ('inceptionv3_model.h5',            'keras'),
    'xception':  ('xception_model.h5',               'keras'),
    'cnn_ens':   ('cnn_ensemble_model.pkl',          'joblib'),
    'inc_ens':   ('inceptionv3_ensemble_model.pkl',  'joblib'),
    'xcp_ens':   ('xception_ensemble_model.pkl',     'joblib'),
}

_lock  = threading.Lock()
_state = {'models': None, 'extractors': None, 'missing': None}


def _download_one(filename):
    """Copy a model file from the HF cache into MODELS_DIR. Returns True on success."""
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        return True
    if not HF_REPO_ID:
        return False
    try:
        from huggingface_hub import hf_hub_download, login as hf_login
        if HF_TOKEN:
            hf_login(token=HF_TOKEN, add_to_git_credential=False)
        # Download to HF cache (no local_dir) → returns the cached path
        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"models/{filename}",
            token=HF_TOKEN or None,
        )
        os.makedirs(MODELS_DIR, exist_ok=True)
        shutil.copy2(cached, dest)
        return True
    except Exception:
        return False


def ensure_models_downloaded():
    if not HF_REPO_ID:
        return
    os.makedirs(MODELS_DIR, exist_ok=True)
    for _, (filename, _) in MODEL_FILES.items():
        _download_one(filename)


def get_models():
    """Returns (models, extractors, missing). Initialises once, thread-safe."""
    with _lock:
        if _state['models'] is not None:
            return _state['models'], _state['extractors'], _state['missing']

        ensure_models_downloaded()
        models, missing = {}, []

        for key, (filename, loader) in MODEL_FILES.items():
            path = os.path.join(MODELS_DIR, filename)
            if not os.path.exists(path):
                missing.append(filename)
                models[key] = None
                continue
            try:
                models[key] = load_model(path) if loader == 'keras' else joblib.load(path)
            except Exception:
                missing.append(filename)
                models[key] = None

        # Build feature extractors in memory from standalone Keras models.
        # Notebooks use 'feature_layer' as the Dense-256 layer name consistently.
        extractors = {}
        for src_key, ext_key in [('cnn', 'cnn_feat'),
                                  ('inception', 'inc_feat'),
                                  ('xception',  'xcp_feat')]:
            base = models.get(src_key)
            if base is None:
                continue
            try:
                extractors[ext_key] = Model(
                    inputs=base.input,
                    outputs=base.get_layer('feature_layer').output,
                )
            except Exception:
                # Fallback: last Dense layer before the softmax
                try:
                    for layer in reversed(base.layers):
                        shape = getattr(layer, 'output_shape', None)
                        if shape and len(shape) == 2:
                            extractors[ext_key] = Model(inputs=base.input,
                                                        outputs=layer.output)
                            break
                except Exception:
                    pass

        _state['models']     = models
        _state['extractors'] = extractors
        _state['missing']    = missing
        return models, extractors, missing
