import base64
import csv
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model_loader import (HF_REPO_ID, HF_TOKEN, MODEL_FILES, force_reload, get_models,
                          peek_state, start_loading_background)
from research_data import DATASET, MODEL_ACCURACY
from xai_engine import CLASS_NAMES, EAA_HIGH, EAA_LOW, explain, predict_all

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
RESULTS_PREFIX = os.environ.get('RESULTS_PREFIX', 'results_2026-10-22')


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_loading_background()          # health check passes while models download
    yield


app = FastAPI(title='NeuroScan', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


def _b64(img):
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


# ── Status ───────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    models, _, _, loading, _ = peek_state()
    return {'status': 'ok', 'models_loaded': sum(v is not None for v in models.values()),
            'models_total': len(MODEL_FILES), 'loading': loading}


@app.get('/api/models/status')
def models_status():
    models, extractors, missing, loading, manifest = peek_state()
    return {
        'repo': HF_REPO_ID,
        'loading': loading,
        'loaded_count': sum(v is not None for v in models.values()),
        'total_count': len(MODEL_FILES),
        'files': {f: models.get(k) is not None for k, f in MODEL_FILES.items()},
        'missing': missing,
        'feature_extractors': sorted(extractors),
        'trained_with': manifest.get('versions'),
    }


@app.post('/api/admin/reload')
def reload_models():
    """Re-download every model from Hugging Face (use after a new upload, no redeploy needed)."""
    force_reload()
    _results_cache['time'] = 0
    return {'status': 'reload started, check /api/models/status in about a minute'}


# ── Prediction ───────────────────────────────────────────────────────────────
@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ('image/jpeg', 'image/jpg', 'image/png'):
        raise HTTPException(400, 'Please upload a JPEG or PNG image.')
    try:
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    except Exception:
        raise HTTPException(400, 'The file could not be read as an image.')

    models, extractors, missing = get_models()
    if not any(v is not None for v in models.values()):
        raise HTTPException(503, f'No models are loaded yet. Check that {HF_REPO_ID} has a models/ folder.')

    final, confidence, per_model, fused = predict_all(image, models, extractors)
    xai = explain(image, models, final, per_model)
    level = 'high' if confidence >= 0.70 else 'moderate' if confidence >= 0.50 else 'low'

    return {
        'final_class': CLASS_NAMES[final],
        'final_class_idx': final,
        'confidence': confidence,
        'confidence_level': level,
        'class_names': CLASS_NAMES,
        'class_probabilities': [float(p) for p in fused],
        'per_model': [{'name': name, 'family': fam,
                       'predicted_class': CLASS_NAMES[int(p.argmax())],
                       'confidence': float(p.max()),
                       'probabilities': [float(v) for v in p]} for name, fam, p in per_model],
        'xai': {
            'original': _b64(image.resize((256, 256))),
            'heatmaps': [{'name': n, 'image': _b64(img)} for n, img in xai['maps'].items()],
            'consensus': _b64(xai['consensus']),
            'pairwise_iou': xai['pairwise_iou'],
            'eaa_iou': xai['eaa_iou'],
            'agreement_level': xai['agreement_level'],
            'alert': xai['alert'],
            'thresholds': {'high': EAA_HIGH, 'low': EAA_LOW},
        },
        'models_used': len(per_model),
        'models_missing': missing,
    }


# ── Results from the latest Colab run (on Hugging Face) ──────────────────────
_results_cache = {'time': 0, 'data': None}

_PRETTY = {
    'CNN_standalone': ('Custom CNN', 'Deep model'),
    'InceptionV3_standalone': ('InceptionV3', 'Deep model'),
    'Xception_standalone': ('Xception', 'Deep model'),
    'FINAL_confidence_weighted': ('Final ensemble, confidence-weighted', 'Final ensemble'),
    'FINAL_majority_vote': ('Final ensemble, majority vote', 'Final ensemble'),
}
_SOURCES = {'cnn_feat_': 'CNN features', 'inceptionv3_feat_': 'InceptionV3 features',
            'xception_feat_': 'Xception features', 'raw_': 'Raw pixels'}
_CLFS = {'rf': 'Random Forest', 'dt': 'Decision Tree', 'svm': 'SVM', 'lr': 'Logistic Regression',
         'softvote_ensemble': 'Soft-vote ensemble'}


def _pretty(name):
    if name in _PRETTY:
        return _PRETTY[name]
    for prefix, source in _SOURCES.items():
        if name.startswith(prefix):
            clf = name[len(prefix):]
            return f'{_CLFS.get(clf, clf)} on {source}', 'Classical ML'
    return name, 'Other'


RESULTS_LOCAL_DIR = os.environ.get('RESULTS_LOCAL_DIR', '')   # optional: a downloaded results folder


def _hf_text(path):
    if RESULTS_LOCAL_DIR:
        return open(os.path.join(RESULTS_LOCAL_DIR, path.split('/', 1)[1])).read()
    from huggingface_hub import hf_hub_download
    return open(hf_hub_download(HF_REPO_ID, path, token=HF_TOKEN or None)).read()


def _load_results():
    try:
        rows = {}
        for r in csv.DictReader(io.StringIO(_hf_text(f'{RESULTS_PREFIX}/metrics/results_master.csv'))):
            rows[r['model']] = r                               # keep the latest row per model
        models = []
        for name, r in rows.items():
            label, group = _pretty(name)
            models.append({'id': name, 'name': label, 'group': group,
                           'accuracy': float(r['accuracy']),
                           'precision': float(r['precision_macro']),
                           'recall': float(r['recall_macro']),
                           'f1': float(r['f1_macro']),
                           'train_time_sec': float(r.get('train_time_sec') or 0)})
        eaa = None
        try:
            eaa = json.loads(_hf_text(f'{RESULTS_PREFIX}/metrics/eaa_iou_summary.json'))
        except Exception:
            pass
        return {'source': 'local' if RESULTS_LOCAL_DIR else 'huggingface', 'run': RESULTS_PREFIX,
                'models': models, 'eaa': eaa}
    except Exception as exc:
        log.info('Live results not available yet (%s); using reference numbers', exc)
        models = [{'id': k, 'name': k, 'group': 'Reference', 'accuracy': v / 100,
                   'precision': None, 'recall': None, 'f1': None, 'train_time_sec': None}
                  for k, v in MODEL_ACCURACY.items()]
        return {'source': 'reference', 'run': None, 'models': models, 'eaa': None}


@app.get('/api/results')
def results():
    if _results_cache['data'] is None or time.time() - _results_cache['time'] > 600:
        _results_cache['data'] = _load_results()
        _results_cache['time'] = time.time()
    return {**_results_cache['data'], 'dataset': DATASET, 'class_names': CLASS_NAMES}


# ── Frontend (built React app) ───────────────────────────────────────────────
_static = os.path.join(os.path.dirname(__file__), 'static')
if os.path.isdir(os.path.join(_static, 'assets')):
    app.mount('/assets', StaticFiles(directory=os.path.join(_static, 'assets')), name='assets')


@app.get('/{full_path:path}', include_in_schema=False)
async def spa(full_path: str):
    index = os.path.join(_static, 'index.html')
    if os.path.exists(index):
        return FileResponse(index, media_type='text/html')
    raise HTTPException(404, 'Frontend not built. Run npm run build in webapp/frontend.')
