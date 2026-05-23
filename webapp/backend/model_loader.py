import os, threading, joblib, shutil, logging
from tensorflow.keras.models import load_model, Model

log = logging.getLogger(__name__)

# Use /tmp — guaranteed writable in any Docker/Railway environment.
# Override with MODELS_DIR env var if a persistent volume is mounted.
MODELS_DIR = os.environ.get('MODELS_DIR', '/tmp/neuro_models')
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
_state = {'models': None, 'extractors': None, 'missing': None, 'loading': False}


def _download_one(filename):
    """Download one model file from HF Hub into MODELS_DIR. Returns True on success."""
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        return True
    if not HF_REPO_ID:
        log.warning('HF_REPO_ID not set — cannot download %s', filename)
        return False
    try:
        from huggingface_hub import hf_hub_download, login as hf_login
        if HF_TOKEN:
            hf_login(token=HF_TOKEN, add_to_git_credential=False)
        log.info('Downloading %s from HuggingFace …', filename)
        # Download to HF cache (no local_dir) then copy — avoids nested subdirectory bug
        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f'models/{filename}',
            token=HF_TOKEN or None,
        )
        os.makedirs(MODELS_DIR, exist_ok=True)
        shutil.copy2(cached, dest)
        log.info('Downloaded %s (%.1f MB)', filename, os.path.getsize(dest) / 1e6)
        return True
    except Exception as exc:
        log.error('Failed to download %s: %s', filename, exc)
        return False


def _load_keras(path):
    """
    Load a Keras .h5 model with a Keras 2/3 compatibility shim.

    Colab saves models with Keras 3 config (includes quantization_config in
    Dense layers). If the runtime is Keras 2 it rejects that key. We catch
    the specific error and retry with a patched Dense that silently ignores
    the unknown kwarg — no model re-training or re-saving required.
    """
    try:
        return load_model(path)
    except Exception as exc:
        if 'quantization_config' not in str(exc):
            raise
        log.warning('Keras 2/3 mismatch detected for %s — retrying with compat shim', os.path.basename(path))
        from tensorflow.keras.layers import Dense as _Dense
        class _K3Dense(_Dense):
            def __init__(self, *args, quantization_config=None, **kwargs):
                super().__init__(*args, **kwargs)
        return load_model(path, custom_objects={'Dense': _K3Dense})


def _load_all():
    """Download then load all models. Called in a background thread."""
    with _lock:
        if _state['models'] is not None:
            return
        _state['loading'] = True

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Download phase
    for _, (filename, _) in MODEL_FILES.items():
        _download_one(filename)

    models, missing = {}, []
    for key, (filename, loader) in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            log.warning('Model file not found after download attempt: %s', filename)
            missing.append(filename)
            models[key] = None
            continue
        try:
            models[key] = _load_keras(path) if loader == 'keras' else joblib.load(path)
            log.info('Loaded: %s', filename)
        except Exception as exc:
            log.error('Failed to load %s: %s', filename, exc)
            missing.append(filename)
            models[key] = None

    # Build feature extractors in memory from standalone Keras models
    extractors = {}
    for src_key, ext_key in [('cnn', 'cnn_feat'),
                              ('inception', 'inc_feat'),
                              ('xception', 'xcp_feat')]:
        base = models.get(src_key)
        if base is None:
            continue
        try:
            extractors[ext_key] = Model(
                inputs=base.input,
                outputs=base.get_layer('feature_layer').output,
            )
        except Exception:
            # Fallback: last Dense layer before softmax
            try:
                for layer in reversed(base.layers):
                    shape = getattr(layer, 'output_shape', None)
                    if shape and len(shape) == 2:
                        extractors[ext_key] = Model(inputs=base.input, outputs=layer.output)
                        break
            except Exception:
                pass

    loaded = sum(1 for v in models.values() if v is not None)
    log.info('Model loading complete: %d/%d loaded, missing: %s',
             loaded, len(MODEL_FILES), missing)

    with _lock:
        _state['models']     = models
        _state['extractors'] = extractors
        _state['missing']    = missing
        _state['loading']    = False


def start_loading_background():
    """Kick off model loading in a daemon thread. Returns immediately."""
    t = threading.Thread(target=_load_all, daemon=True, name='model-loader')
    t.start()
    return t


def get_models():
    """
    Returns (models, extractors, missing).
    Blocks until loading is complete if already started.
    Starts loading synchronously if not yet started.
    """
    with _lock:
        if _state['models'] is not None:
            return _state['models'], _state['extractors'], _state['missing']
        already_loading = _state['loading']

    if not already_loading:
        _load_all()
    else:
        # Another thread is loading — wait for it
        import time
        while True:
            with _lock:
                if _state['models'] is not None:
                    break
            time.sleep(0.5)

    with _lock:
        return _state['models'], _state['extractors'], _state['missing']


def peek_state():
    """Non-blocking — returns current state without triggering a download."""
    with _lock:
        return (
            _state['models'] or {},
            _state['extractors'] or {},
            _state['missing'] or [],
            _state['loading'],
        )
