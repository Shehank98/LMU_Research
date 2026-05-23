import numpy as np
import cv2
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

log = logging.getLogger(__name__)

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_COLORS = {
    'Glioma':      '#e74c3c',
    'Meningioma':  '#e67e22',
    'No Tumor':    '#27ae60',
    'Pituitary':   '#2980b9',
}
CONFIDENCE_THRESHOLD = 0.70


def preprocess(pil_image, size):
    img = pil_image.convert('RGB').resize((size, size))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, size, size, 3)


def _last_conv_name(model):
    """Return the name of the last layer with a 4-D spatial output."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
            if isinstance(shape, list):
                shape = shape[0]
            if len(shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError(f"No conv layer found in model {model.name}")


def get_gradcam(model, image_arr, class_idx):
    """
    Returns a normalised (0–1) heatmap of shape (H, W).
    image_arr: shape (1, H, W, 3), already normalised to [0, 1].
    """
    conv_name = _last_conv_name(model)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_name).output, model.output],
    )
    img = tf.cast(image_arr, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        # conv_outputs is an intermediate tensor (not a tf.Variable) so it must
        # be watched explicitly for tape.gradient to return a value.
        tape.watch(conv_outputs)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError(
            f'GradCAM: tape.gradient returned None for layer {conv_name}. '
            'Model may not support gradient computation in this TF version.'
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_outputs[0] @ pooled_grads[..., tf.newaxis]).numpy().squeeze()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    h, w = image_arr.shape[1], image_arr.shape[2]
    return cv2.resize(heatmap, (w, h))


def overlay_heatmap(pil_image, heatmap, alpha=0.45):
    """Overlay a heatmap on a PIL image. Returns a PIL image."""
    orig = np.array(pil_image.convert('RGB').resize((256, 256)))
    cam = np.uint8(255 * heatmap)
    cam_color = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig, 1 - alpha, cam_color, alpha, 0)
    return Image.fromarray(overlay)


def build_consensus_map(heatmaps, weights):
    """
    heatmaps : list of (H, W) arrays
    weights  : list of floats (confidences), same length
    Returns  : weighted-average (H, W) normalised 0–1
    """
    total = sum(weights) or 1.0
    consensus = sum(h * w for h, w in zip(heatmaps, weights)) / total
    if consensus.max() > 0:
        consensus /= consensus.max()
    return consensus


def iou_score(map_a, map_b, threshold=0.5):
    a = (map_a >= threshold).astype(np.uint8)
    b = (map_b >= threshold).astype(np.uint8)
    intersection = np.logical_and(a, b).sum()
    union        = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def average_pairwise_iou(heatmaps, threshold=0.5):
    n = len(heatmaps)
    scores = [iou_score(heatmaps[i], heatmaps[j], threshold)
              for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(scores)) if scores else 0.0


def confidence_weighted_predict(pil_image, models, extractors):
    """
    Runs all 6 ensemble components.  Each model uses its own trained input size
    (read from model.input_shape[1]) — no hardcoded dimensions.

    Returns (final_class_idx, ensemble_confidence, per_model_dict, combined_probs).
    """
    per_model     = {}
    combined_probs = np.zeros(4)

    def _add(probs, name):
        per_model[name] = probs
        combined_probs[:] += probs

    # --- Standalone DL models ---
    for model_key, display_name in [('cnn',       'CNN'),
                                     ('inception', 'InceptionV3'),
                                     ('xception',  'Xception')]:
        m = models.get(model_key)
        if m is None:
            continue
        size = m.input_shape[1]          # actual trained input size
        img  = preprocess(pil_image, size)
        _add(m.predict(img, verbose=0)[0], display_name)

    # --- Feature extractor → classical ensemble ---
    for feat_key, ens_key, display_name in [
        ('cnn_feat', 'cnn_ens', 'CNN + Ensemble'),
        ('inc_feat', 'inc_ens', 'InceptionV3 + Ensemble'),
        ('xcp_feat', 'xcp_ens', 'Xception + Ensemble'),
    ]:
        feat_model = extractors.get(feat_key)
        clf        = models.get(ens_key)
        if feat_model is None or clf is None:
            continue
        size  = feat_model.input_shape[1]
        img   = preprocess(pil_image, size)
        feats = feat_model.predict(img, verbose=0)
        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(feats)[0]
        else:
            prob = np.eye(4)[int(clf.predict(feats)[0])]
        _add(prob, display_name)

    final_class = int(np.argmax(combined_probs))
    total       = float(np.sum(combined_probs))
    ensemble_confidence = float(np.max(combined_probs) / total) if total > 0 else 0.0

    return final_class, ensemble_confidence, per_model, combined_probs
