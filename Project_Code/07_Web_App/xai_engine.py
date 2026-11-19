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

from model_loader import MODEL_SET, ensemble_proba, input_size

log = logging.getLogger(__name__)

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MAP_SIZE = 256
IOU_THRESHOLD = 0.5
EAA_HIGH, EAA_LOW = (
    0.20,
    0.10,
)  # calibrated on the 1,600 test images: < 0.10 flags about half of the cases and catches 82% of errors
CAM_METHOD = "hirescam"  # 'hirescam' (default) or 'gradcam'; must match the notebook

DEEP_MODELS = [("cnn", "CNN"), ("xception", "Xception"), ("inception", "InceptionV3")]
FEATURE_MODELS = [
    ("cnn", "CNN + Ensemble"),
    ("inception", "InceptionV3 + Ensemble"),
    ("xception", "Xception + Ensemble"),
]


def crop_head(rgb, margin=4):
    """Crop to the bounding box of the head (same rule as Project_Code/02_Improved_Models/data_preparation.py, used by the improved models)."""
    grey = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(grey, 45, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rgb
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < 0.2 * rgb.shape[1] or h < 0.2 * rgb.shape[0]:
        return rgb
    return rgb[max(y - margin, 0) : y + h + margin, max(x - margin, 0) : x + w + margin]


def preprocess(pil_image, size):
    if MODEL_SET == "improved":
        rgb = crop_head(np.asarray(pil_image.convert("RGB")))
        img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    else:
        img = np.asarray(pil_image.convert("RGB").resize((size, size), Image.BILINEAR))
    return (np.asarray(img, dtype=np.float32) / 255.0)[None, ...]


def _input_size(model):
    return input_size(model, 256)


# ── Prediction ───────────────────────────────────────────────────────────────
def predict_all(pil_image, models, extractors):
    """Returns (final_idx, confidence, per_model[(name, family, probs)], fused_probs)."""
    per_model = []
    for key, name in [("cnn", "CNN"), ("inception", "InceptionV3"), ("xception", "Xception")]:
        m = models.get(key)
        if m is not None:
            probs = m.predict(preprocess(pil_image, _input_size(m)), verbose=0)[0]
            per_model.append((name, "Deep model", np.asarray(probs, dtype=float)))
    for family, name in FEATURE_MODELS:
        ext, ens = extractors.get(family), models.get(f"{family}_ens")
        if ext is None or ens is None:
            continue
        feats = ext.predict(preprocess(pil_image, ext.input_size))
        per_model.append((name, "Feature ensemble", np.asarray(ensemble_proba(ens, feats)[0], dtype=float)))
    if not per_model:
        raise RuntimeError("no models available")
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
    raise ValueError(f"no spatial layer in {model.name}")


def _nested_backbone(model):
    """The pretrained backbone when it is nested inside the classifier (improved models), else None."""
    for layer in model.layers:
        if isinstance(layer, Model) and layer.name != "augment":
            return layer
    return None


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
    elif _nested_backbone(model) is not None:
        # improved models: the pretrained backbone is a model inside the model, so run the layers by hand
        base = _nested_backbone(model)
        key = id(model)
        if key not in _grad_models:
            _grad_models[key] = Model(
                base.inputs, [base.get_layer(_last_spatial_layer(base)).output, base.output]
            )
        i = model.layers.index(base)
        with tf.GradientTape() as tape:
            h = x
            for layer in model.layers[1:i]:
                if layer.name != "augment":
                    h = layer(h, training=False)
            conv, h = _grad_models[key](h, training=False)
            tape.watch(conv)
            for layer in model.layers[i + 1 : -1]:
                h = layer(h, training=False)
            score = _logits(dense, h)[:, class_idx]
        grads = tape.gradient(score, conv)
    else:
        key = id(model)
        if key not in _grad_models:
            head_input = dense.input if dense is not None else model.output
            _grad_models[key] = Model(
                model.inputs, [model.get_layer(_last_spatial_layer(model)).output, head_input]
            )
        with tf.GradientTape() as tape:
            conv, h = _grad_models[key](x, training=False)
            scores = _logits(dense, h) if dense is not None else h
            score = scores[:, class_idx]
        grads = tape.gradient(score, conv)
    if CAM_METHOD == "gradcam":
        cam = tf.einsum("bhwc,bc->bhw", conv, tf.reduce_mean(grads, axis=(1, 2)))
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
    base = np.asarray(pil_image.convert("RGB").resize((MAP_SIZE, MAP_SIZE)))
    colour = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(base, 1 - alpha, colour, alpha, 0))


def agreement_level(eaa):
    if eaa is None:
        return None
    return "high" if eaa >= EAA_HIGH else "moderate" if eaa >= EAA_LOW else "low"


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
            log.warning("Grad-CAM failed for %s: %s", name, exc)
    names = list(maps)
    pairs = [
        {"pair": f"{names[i]} vs {names[j]}", "iou": iou(maps[names[i]], maps[names[j]])}
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    eaa = float(np.mean([p["iou"] for p in pairs])) if pairs else None
    consensus = None
    if maps:
        w = [conf.get(n, 1.0) for n in names]
        consensus = sum(maps[n] * wi for n, wi in zip(names, w)) / (sum(w) or 1.0)
        consensus = consensus / consensus.max() if consensus.max() > 0 else consensus
    # heatmaps are drawn on the image the models actually saw (cropped to the head for the improved set)
    display = (
        Image.fromarray(crop_head(np.asarray(pil_image.convert("RGB"))))
        if MODEL_SET == "improved"
        else pil_image
    )
    return {
        "maps": {n: overlay(display, maps[n]) for n in names},
        "consensus": overlay(display, consensus) if consensus is not None else None,
        "pairwise_iou": pairs,
        "eaa_iou": eaa,
        "agreement_level": agreement_level(eaa),
        "alert": eaa is not None and eaa < EAA_LOW,
    }
