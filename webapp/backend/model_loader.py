"""
Downloads the trained models from Hugging Face and loads them once, in the background.

Files expected in the Hugging Face repo under models/ (written by the Colab notebook
final_code_set_2026-10-22/Brain_Tumor_Train_And_Log.ipynb):

    cnn_model.h5, inceptionv3_model.h5, xception_model.h5      standalone classifiers (256 px)
    cnn_ext.h5, inceptionv3_ext.h5, xception_ext.h5            feature extractors (128 px)
    cnn_ensemble_model.pkl, ...                                 [RF, DT, SVM] on those features
    manifest.json                                               layer names, sizes, versions
"""
import json
import logging
import os
import shutil
import threading

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

log = logging.getLogger(__name__)

MODELS_DIR = os.environ.get('MODELS_DIR', '/tmp/neuro_models')
HF_REPO_ID = os.environ.get('HF_REPO_ID', 'shehank98/brain-tumor-mri-models')
HF_TOKEN = os.environ.get('HF_TOKEN', '')

MODEL_FILES = {
    'cnn':           'cnn_model.h5',
    'inception':     'inceptionv3_model.h5',
    'xception':      'xception_model.h5',
    'cnn_ext':       'cnn_ext.h5',
    'inception_ext': 'inceptionv3_ext.h5',
    'xception_ext':  'xception_ext.h5',
    'cnn_ens':       'cnn_ensemble_model.pkl',
    'inception_ens': 'inceptionv3_ensemble_model.pkl',
    'xception_ens':  'xception_ensemble_model.pkl',
}
DEFAULT_FEATURE_LAYERS = {'cnn': 'dense', 'inception': 'dense_2', 'xception': 'dense_3'}

_lock = threading.Lock()
_state = {'models': None, 'extractors': None, 'missing': None, 'loading': False, 'manifest': {}}


def download_file(path_in_repo):
    """Download one file from the HF repo into MODELS_DIR. Returns the local path or None."""
    dest = os.path.join(MODELS_DIR, os.path.basename(path_in_repo))
    if os.path.exists(dest):
        return dest
    if not HF_REPO_ID:
        return None
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename=path_in_repo, token=HF_TOKEN or None)
        os.makedirs(MODELS_DIR, exist_ok=True)
        shutil.copy2(cached, dest)
        log.info('Downloaded %s (%.1f MB)', path_in_repo, os.path.getsize(dest) / 1e6)
        return dest
    except Exception as exc:
        log.warning('Could not download %s: %s', path_in_repo, exc)
        return None


def _download_model(filename):
    return download_file(f'models/{filename}') or download_file(filename)


def _load_keras(path):
    """Load .h5, tolerating the Keras 2/3 'quantization_config' mismatch from Colab saves."""
    try:
        return load_model(path, compile=False)
    except Exception as exc:
        if 'quantization_config' not in str(exc):
            raise
        from tensorflow.keras.layers import Dense

        class _Dense(Dense):
            def __init__(self, *args, quantization_config=None, **kwargs):
                super().__init__(*args, **kwargs)

        return load_model(path, compile=False, custom_objects={'Dense': _Dense})


def input_size(model, default=256):
    """Image size a model expects. Keras 3 Sequential models without an Input layer
    raise on .input_shape, so fall back to the build config."""
    getters = (lambda: model.input_shape[1],
               lambda: model.get_build_config()['input_shape'][1])
    for get in getters:
        try:
            value = get()
            if value:
                return int(value)
        except Exception:
            continue
    return default


class FeatureExtractor:
    """Returns the activations of one named Dense layer (the 256-d feature vector)."""

    def __init__(self, model, layer_name):
        self.layer_name = layer_name
        self.input_size = input_size(model, 128)
        self._sub = None
        self._model = model
        try:
            self._sub = Model(model.inputs, model.get_layer(layer_name).output)
        except Exception:
            self._sub = None          # fall back to a manual pass for Sequential models

    def predict(self, x):
        if self._sub is not None:
            return self._sub.predict(x, verbose=0)
        out = tf.convert_to_tensor(x, tf.float32)
        for layer in self._model.layers:
            out = layer(out, training=False)
            if layer.name == self.layer_name:
                return out.numpy()
        raise ValueError(f'layer {self.layer_name} not found')


def ensemble_proba(ensemble, features):
    """Ensemble file is a list [RF, DT, SVM] (new) or one sklearn estimator (older uploads)."""
    if isinstance(ensemble, (list, tuple)):
        return np.mean([m.predict_proba(features) for m in ensemble], axis=0)
    if hasattr(ensemble, 'predict_proba'):
        return ensemble.predict_proba(features)
    return np.eye(4)[np.asarray(ensemble.predict(features), dtype=int)]


def _load_all():
    with _lock:
        if _state['models'] is not None or _state['loading']:
            return
        _state['loading'] = True
    try:
        _do_load()
    except Exception as exc:                       # never leave callers waiting forever
        log.exception('Model loading crashed: %s', exc)
        with _lock:
            _state.update(models={k: None for k in MODEL_FILES}, extractors={},
                          missing=list(MODEL_FILES.values()), loading=False)


def _do_load():
    os.makedirs(MODELS_DIR, exist_ok=True)
    log.info('Loading models from Hugging Face repo %s', HF_REPO_ID)

    manifest = {}
    mpath = _download_model('manifest.json')
    if mpath:
        try:
            manifest = json.load(open(mpath))
            log.info('Manifest versions: %s', manifest.get('versions'))
            import sklearn
            trained = (manifest.get('versions') or {}).get('scikit_learn')
            if trained and trained != sklearn.__version__:
                log.warning('Models were trained with scikit-learn %s but the server has %s. Pin '
                            'scikit-learn==%s in requirements.txt to avoid invalid ensemble results.',
                            trained, sklearn.__version__, trained)
        except Exception as exc:
            log.warning('Bad manifest: %s', exc)

    models, missing = {}, []
    for key, filename in MODEL_FILES.items():
        path = _download_model(filename)
        if not path:
            models[key] = None
            missing.append(filename)
            continue
        try:
            models[key] = _load_keras(path) if filename.endswith('.h5') else joblib.load(path)
            log.info('Loaded %s', filename)
        except Exception as exc:
            log.error('Failed to load %s: %s', filename, exc)
            models[key] = None
            missing.append(filename)

    extractors = {}
    for family in ('cnn', 'inception', 'xception'):
        ext = models.get(f'{family}_ext')
        if ext is None:
            continue
        layer = (manifest.get('extractors', {}).get(family, {}).get('layer')
                 or DEFAULT_FEATURE_LAYERS[family])
        try:
            extractors[family] = FeatureExtractor(ext, layer)
        except Exception as exc:
            log.error('Feature extractor %s failed: %s', family, exc)

    loaded = sum(v is not None for v in models.values())
    log.info('Models ready: %d/%d loaded, missing %s', loaded, len(MODEL_FILES), missing)
    with _lock:
        _state.update(models=models, extractors=extractors, missing=missing,
                      loading=False, manifest=manifest)


def start_loading_background():
    t = threading.Thread(target=_load_all, daemon=True, name='model-loader')
    t.start()
    return t


def force_reload():
    """Delete the local copies and download everything again (after a new upload to HF)."""
    with _lock:
        _state.update(models=None, extractors=None, missing=None, loading=False, manifest={})
    shutil.rmtree(MODELS_DIR, ignore_errors=True)
    return start_loading_background()


def get_models():
    """Blocks until loading is done. Returns (models, extractors, missing)."""
    import time
    if _state['models'] is None and not _state['loading']:
        _load_all()
    while _state['models'] is None:
        time.sleep(0.5)
    return _state['models'], _state['extractors'], _state['missing']


def peek_state():
    """Non-blocking snapshot: (models, extractors, missing, loading, manifest)."""
    with _lock:
        return (_state['models'] or {}, _state['extractors'] or {}, _state['missing'] or [],
                _state['loading'], _state['manifest'] or {})
