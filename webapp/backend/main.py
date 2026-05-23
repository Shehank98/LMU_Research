import base64, io, os, logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

log = logging.getLogger(__name__)

from model_loader import get_models, start_loading_background, peek_state, MODEL_FILES
from xai_engine import (
    CLASS_NAMES, CLASS_COLORS, CONFIDENCE_THRESHOLD,
    preprocess, get_gradcam, overlay_heatmap,
    build_consensus_map, iou_score, average_pairwise_iou,
    confidence_weighted_predict,
)
from research_data import (
    MODEL_ACCURACY, MODEL_FAMILY, FAMILY_COLORS,
    CONFUSION_MATRICES, CLASSIFICATION_REPORTS,
    FEATURE_ML_ACCURACY, RAW_PIXEL_ACCURACY,
    DATASET, ENSEMBLE_BUILDUP, LITERATURE, MODEL_PARAMS,
    CLASS_COLORS as DATA_CLASS_COLORS,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_loading_background()   # non-blocking — health check passes immediately
    yield


app = FastAPI(title='NeuroScan AI', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


# ── Health ────────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    models, _, missing, loading = peek_state()
    loaded = sum(1 for v in models.values() if v is not None)
    return {
        'status': 'ok',
        'models_loaded': loaded,
        'models_total': len(MODEL_FILES),
        'models_loading': loading,
    }


# ── Model status ──────────────────────────────────────────────────────────────
@app.get('/api/models/status')
def models_status():
    models, _, missing = get_models()
    return {
        'status': {k: (v is not None) for k, v in models.items()},
        'missing': missing,
        'loaded_count': sum(1 for v in models.values() if v is not None),
        'total_count': len(MODEL_FILES),
    }


# ── Pre-computed research data ────────────────────────────────────────────────
@app.get('/api/research')
def research_data():
    return {
        'model_accuracy':         MODEL_ACCURACY,
        'model_family':           MODEL_FAMILY,
        'family_colors':          FAMILY_COLORS,
        'confusion_matrices':     CONFUSION_MATRICES,
        'classification_reports': CLASSIFICATION_REPORTS,
        'feature_ml_accuracy':    FEATURE_ML_ACCURACY,
        'raw_pixel_accuracy':     RAW_PIXEL_ACCURACY,
        'dataset':                DATASET,
        'ensemble_buildup':       ENSEMBLE_BUILDUP,
        'literature':             LITERATURE,
        'model_params':           MODEL_PARAMS,
        'class_names':            CLASS_NAMES,
        'class_colors':           CLASS_COLORS,
        'confidence_threshold':   CONFIDENCE_THRESHOLD,
    }


# ── Debug info ───────────────────────────────────────────────────────────────
@app.get('/api/debug/models')
def debug_models():
    """Returns model input/output shapes and a zero-image test prediction."""
    models, extractors, missing = get_models()
    info = {}
    for key, m in models.items():
        if m is None:
            info[key] = None
            continue
        try:
            in_shape  = list(m.input_shape)
            out_shape = list(m.output_shape)
            # Run a zeroed test image through to detect preprocessing/output issues
            size = in_shape[1] or 224
            dummy = np.zeros((1, size, size, 3), dtype=np.float32)
            probs = m.predict(dummy, verbose=0)[0].tolist()
            info[key] = {
                'input_shape':  in_shape,
                'output_shape': out_shape,
                'test_probs':   [round(p, 4) for p in probs],
                'test_pred':    CLASS_NAMES[int(np.argmax(probs))] if len(probs) == 4 else 'n/a',
            }
        except Exception as exc:
            info[key] = {'error': str(exc)}
    return {'models': info, 'class_names': CLASS_NAMES, 'missing': missing}


# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ('image/jpeg', 'image/png', 'image/jpg'):
        raise HTTPException(400, 'Only JPEG/PNG images are accepted')

    data = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        raise HTTPException(400, 'Could not decode image')

    models, extractors, missing = get_models()
    if sum(1 for v in models.values() if v is not None) == 0:
        raise HTTPException(503, 'No models loaded. Configure HF_REPO_ID to auto-download.')

    final_class, ensemble_conf, per_model_raw, combined_probs = \
        confidence_weighted_predict(pil_image, models, extractors)

    # Log raw per-model probabilities so Railway logs show what the model outputs
    for mname, probs in per_model_raw.items():
        log.info('raw probs [%s]: %s → %s (%.1f%%)',
                 mname,
                 [f'{p:.3f}' for p in probs],
                 CLASS_NAMES[int(np.argmax(probs))],
                 float(np.max(probs)) * 100)

    class_name = CLASS_NAMES[final_class]

    per_model_out = [
        {
            'name': name,
            'predicted_class': CLASS_NAMES[int(np.argmax(probs))],
            'predicted_idx':   int(np.argmax(probs)),
            'confidence':      float(np.max(probs)),
            'probabilities':   [float(p) for p in probs],
        }
        for name, probs in per_model_raw.items()
    ]

    # GRAD-CAM uses standalone softmax models (not feature extractors)
    gradcam_sources = {
        'cnn':       models.get('cnn'),
        'xception':  models.get('xception'),
        'inception': models.get('inception'),
    }
    heatmaps_raw, heatmap_confs = {}, {}
    for key, gmodel in gradcam_sources.items():
        if gmodel is None:
            continue
        img_in = preprocess(pil_image, gmodel.input_shape[1])
        try:
            hm = get_gradcam(gmodel, img_in, final_class)
            heatmaps_raw[key]   = hm
            heatmap_confs[key]  = float(np.max(gmodel.predict(img_in, verbose=0)[0]))
        except Exception as exc:
            log.warning('GRAD-CAM failed for %s: %s', key, exc)

    heatmaps_b64 = {k: _pil_to_b64(overlay_heatmap(pil_image, hm))
                    for k, hm in heatmaps_raw.items()}

    iou_scores, consensus_b64, avg_iou = {}, None, 0.0
    if len(heatmaps_raw) >= 2:
        keys    = list(heatmaps_raw.keys())
        maps    = [heatmaps_raw[k] for k in keys]
        weights = [heatmap_confs.get(k, 1.0) for k in keys]
        consensus     = build_consensus_map(maps, weights)
        consensus_b64 = _pil_to_b64(overlay_heatmap(pil_image, consensus))
        avg_iou       = average_pairwise_iou(maps)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                iou_scores[f'{keys[i]}_vs_{keys[j]}'] = iou_score(
                    heatmaps_raw[keys[i]], heatmaps_raw[keys[j]])

    prob_sum = float(combined_probs.sum()) or 1.0
    class_probabilities = [float(p / prob_sum) for p in combined_probs]

    confidence_level = (
        'high'     if ensemble_conf >= CONFIDENCE_THRESHOLD else
        'moderate' if ensemble_conf >= 0.50 else 'low'
    )

    return {
        'final_class':         class_name,
        'final_class_idx':     final_class,
        'confidence':          ensemble_conf,
        'confidence_level':    confidence_level,
        'class_names':         CLASS_NAMES,
        'class_colors':        CLASS_COLORS,
        'class_probabilities': class_probabilities,
        'per_model':           per_model_out,
        'heatmaps':            heatmaps_b64,
        'consensus_heatmap':   consensus_b64,
        'iou_scores':          iou_scores,
        'average_iou':         avg_iou,
        'models_used':         sum(1 for v in models.values() if v is not None),
        'models_missing':      missing,
    }


# Serve React SPA — must be registered last
_static = os.path.join(os.path.dirname(__file__), 'static')
if os.path.isdir(_static):
    app.mount('/', StaticFiles(directory=_static, html=True), name='static')
