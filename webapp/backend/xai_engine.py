"""
Prediction and explainability.

predict_all()   runs the six predictors and fuses them with a confidence-weighted vote.
explain()       builds Grad-CAM maps for CNN, Xception and InceptionV3, the consensus map,
                and the Ensemble Attention Agreement score (EAA-IoU = mean pairwise IoU).
The Grad-CAM code is the same as in the Colab notebook, so the app and the report agree.
"""
import logging

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model

from model_loader import ensemble_proba, input_size

log = logging.getLogger(__name__)

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
MAP_SIZE = 256
IOU_THRESHOLD = 0.5
EAA_HIGH, EAA_LOW = 0.65, 0.40          # >= high: agree; < low: flag for radiologist review
CAM_METHOD = 'hirescam'                  # 'hirescam' (default) or 'gradcam'; must match the notebook

DEEP_MODELS = [('cnn', 'CNN'), ('xception', 'Xception'), ('inception', 'InceptionV3')]
FEATURE_MODELS = [('cnn', 'CNN + Ensemble'), ('inception', 'InceptionV3 + Ensemble'),
                  ('xception', 'Xception + Ensemble')]


def preprocess(pil_image, size):
    img = pil_image.convert('RGB').resize((size, size), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0)[None, ...]


def _input_size(model):
    return input_size(model, 256)


# ── Prediction ───────────────────────────────────────────────────────────────
def predict_all(pil_image, models, extractors):
    """Returns (final_idx, confidence, per_model[(name, family, probs)], fused_probs)."""
    per_model = []
    for key, name in [('cnn', 'CNN'), ('inception', 'InceptionV3'), ('xception', 'Xception')]:
        m = models.get(key)
        if m is not None:
            probs = m.predict(preprocess(pil_image, _input_size(m)), verbose=0)[0]
            per_model.append((name, 'Deep model', np.asarray(probs, dtype=float)))
    for family, name in FEATURE_MODELS:
        ext, ens = extractors.get(family), models.get(f'{family}_ens')
        if ext is None or ens is None:
            continue
        feats = ext.predict(preprocess(pil_image, ext.input_size))
        per_model.append((name, 'Feature ensemble', np.asarray(ensemble_proba(ens, feats)[0], dtype=float)))
    if not per_model:
        raise RuntimeError('no models available')
    fused = np.sum([p for _, _, p in per_model], axis=0)
    final = int(np.argmax(fused))
    confidence = float(fused[final] / fused.sum())
    return final, confidence, per_model, fused / fused.sum()


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
_grad_models = {}


def _last_conv_index(model):
    idx = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
            idx = i
    return idx


def _last_spatial_layer(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError(f'no spatial layer in {model.name}')


def _final_dense(model):
    last = model.layers[-1]
    return last if isinstance(last, tf.keras.layers.Dense) else None


def _logits(dense, h):
    # Pre-softmax class scores. Grad-CAM must use these: when a model is confident the
    # softmax output saturates, its gradients vanish and the heatmap comes out empty.
    z = tf.matmul(h, dense.kernel)
    return z + dense.bias if dense.bias is not None else z


def gradcam(model, image, class_idx):
    """Normalised Grad-CAM map (MAP_SIZE x MAP_SIZE) for one image of shape (1, H, W, 3)."""
    x = tf.convert_to_tensor(image, tf.float32)
    dense = _final_dense(model)
    if isinstance(model, tf.keras.Sequential):
        conv_i = _last_conv_index(model)
        body = model.layers[:-1] if dense is not None else model.layers
        with tf.GradientTape() as tape:
            out, conv = x, None
            for i, layer in enumerate(body):
                out = layer(out, training=False)
                if i == conv_i:
                    conv = out
                    tape.watch(conv)
            scores = _logits(dense, out) if dense is not None else out
            score = scores[:, class_idx]
        grads = tape.gradient(score, conv)
    else:
        key = id(model)
        if key not in _grad_models:
            head_input = dense.input if dense is not None else model.output
            _grad_models[key] = Model(model.inputs,
                                      [model.get_layer(_last_spatial_layer(model)).output, head_input])
        with tf.GradientTape() as tape:
            conv, h = _grad_models[key](x, training=False)
            scores = _logits(dense, h) if dense is not None else h
            score = scores[:, class_idx]
        grads = tape.gradient(score, conv)
    if CAM_METHOD == 'gradcam':
        cam = tf.einsum('bhwc,bc->bhw', conv, tf.reduce_mean(grads, axis=(1, 2)))
    else:
        # HiResCAM: element-wise gradient x activation keeps spatial detail. It equals Grad-CAM for
        # Global-Average-Pooling heads (InceptionV3, Xception) and stays correct for the CNN's
        # Flatten + Dense head, where averaging the gradients cancels out and gives an empty map.
        cam = tf.reduce_sum(grads * conv, axis=-1)
    cam = tf.nn.relu(cam)
    cam = tf.math.divide_no_nan(cam, tf.reduce_max(cam, axis=(1, 2), keepdims=True))
    return tf.image.resize(cam[..., None], (MAP_SIZE, MAP_SIZE))[0, ..., 0].numpy()


def iou(a, b, t=IOU_THRESHOLD):
    A, B = a >= t, b >= t
    union = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum() / union) if union else 0.0


def overlay(pil_image, heatmap, alpha=0.45):
    base = np.asarray(pil_image.convert('RGB').resize((MAP_SIZE, MAP_SIZE)))
    colour = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(base, 1 - alpha, colour, alpha, 0))


def agreement_level(eaa):
    if eaa is None:
        return None
    return 'high' if eaa >= EAA_HIGH else 'moderate' if eaa >= EAA_LOW else 'low'


def explain(pil_image, models, class_idx, per_model):
    """Grad-CAM for the three deep models, consensus map and EAA-IoU."""
    conf = {name: float(p.max()) for name, _, p in per_model}
    maps = {}
    for key, name in DEEP_MODELS:
        m = models.get(key)
        if m is None:
            continue
        try:
            maps[name] = gradcam(m, preprocess(pil_image, _input_size(m)), class_idx)
        except Exception as exc:
            log.warning('Grad-CAM failed for %s: %s', name, exc)
    names = list(maps)
    pairs = [{'pair': f'{names[i]} vs {names[j]}', 'iou': iou(maps[names[i]], maps[names[j]])}
             for i in range(len(names)) for j in range(i + 1, len(names))]
    eaa = float(np.mean([p['iou'] for p in pairs])) if pairs else None
    consensus = None
    if maps:
        w = [conf.get(n, 1.0) for n in names]
        consensus = sum(maps[n] * wi for n, wi in zip(names, w)) / (sum(w) or 1.0)
        consensus = consensus / consensus.max() if consensus.max() > 0 else consensus
    return {
        'maps': {n: overlay(pil_image, maps[n]) for n in names},
        'consensus': overlay(pil_image, consensus) if consensus is not None else None,
        'pairwise_iou': pairs,
        'eaa_iou': eaa,
        'agreement_level': agreement_level(eaa),
        'alert': eaa is not None and eaa < EAA_LOW,
    }
