import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

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
    return np.expand_dims(arr, axis=0)  # shape (1, size, size, 3)


def get_gradcam(model, image_arr, class_idx):
    """
    Returns a normalised (0–1) heatmap of shape (H, W) using
    gradient-weighted class activation mapping.
    image_arr: shape (1, H, W, 3), already normalised.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.layers[-3].output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_arr)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    h, w = image_arr.shape[1], image_arr.shape[2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    return heatmap_resized


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
    weights  : list of floats (softmax confidences), same length
    Returns  : weighted-average (H, W) array normalised 0–1
    """
    total = sum(weights) or 1.0
    consensus = sum(h * w for h, w in zip(heatmaps, weights)) / total
    if consensus.max() > 0:
        consensus /= consensus.max()
    return consensus


def iou_score(map_a, map_b, threshold=0.5):
    """Intersection over Union between two binarised heatmaps."""
    a = (map_a >= threshold).astype(np.uint8)
    b = (map_b >= threshold).astype(np.uint8)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def average_pairwise_iou(heatmaps, threshold=0.5):
    """Mean IoU across all pairs of heatmaps."""
    n = len(heatmaps)
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(iou_score(heatmaps[i], heatmaps[j], threshold))
    return float(np.mean(scores)) if scores else 0.0


def confidence_weighted_predict(pil_image, models, extractors):
    """
    Runs all 6 models on the image and returns:
      - final_class (int)
      - ensemble_confidence (float 0–1)
      - per_model dict with class probabilities and predicted class per model
      - combined probability array (4,)
    """
    per_model = {}
    combined_probs = np.zeros(4)

    # --- Deep model predictions (CNN classification model uses predict_proba via joblib) ---
    img_256 = preprocess(pil_image, 256)
    img_128 = preprocess(pil_image, 128)

    # CNN standalone — Keras model, takes (1, 256, 256, 3) directly
    cnn_model = models.get('cnn')
    if cnn_model is not None:
        prob = cnn_model.predict(img_256, verbose=0)[0]
        per_model['CNN'] = prob
        combined_probs += prob

    # InceptionV3 standalone
    inc_model = models.get('inception')
    if inc_model is not None:
        prob = inc_model.predict(img_256, verbose=0)[0]
        per_model['InceptionV3'] = prob
        combined_probs += prob

    # Xception standalone
    xcp_model = models.get('xception')
    if xcp_model is not None:
        prob = xcp_model.predict(img_256, verbose=0)[0]
        per_model['Xception'] = prob
        combined_probs += prob

    # CNN feature extractor → classical ensemble
    cnn_feat = extractors.get('cnn_feat')
    cnn_ens = models.get('cnn_ens')
    if cnn_feat is not None and cnn_ens is not None:
        feats = cnn_feat.predict(img_128, verbose=0)
        if hasattr(cnn_ens, 'predict_proba'):
            prob = cnn_ens.predict_proba(feats)[0]
        else:
            pred = int(cnn_ens.predict(feats)[0])
            prob = np.eye(4)[pred]
        per_model['CNN + Ensemble'] = prob
        combined_probs += prob

    # InceptionV3 feature extractor → classical ensemble
    inc_feat = extractors.get('inc_feat')
    inc_ens = models.get('inc_ens')
    if inc_feat is not None and inc_ens is not None:
        feats = inc_feat.predict(img_128, verbose=0)
        if hasattr(inc_ens, 'predict_proba'):
            prob = inc_ens.predict_proba(feats)[0]
        else:
            pred = int(inc_ens.predict(feats)[0])
            prob = np.eye(4)[pred]
        per_model['InceptionV3 + Ensemble'] = prob
        combined_probs += prob

    # Xception feature extractor → classical ensemble
    xcp_feat = extractors.get('xcp_feat')
    xcp_ens = models.get('xcp_ens')
    if xcp_feat is not None and xcp_ens is not None:
        feats = xcp_feat.predict(img_128, verbose=0)
        if hasattr(xcp_ens, 'predict_proba'):
            prob = xcp_ens.predict_proba(feats)[0]
        else:
            pred = int(xcp_ens.predict(feats)[0])
            prob = np.eye(4)[pred]
        per_model['Xception + Ensemble'] = prob
        combined_probs += prob

    final_class = int(np.argmax(combined_probs))
    ensemble_confidence = float(np.max(combined_probs) / np.sum(combined_probs)) \
        if np.sum(combined_probs) > 0 else 0.0

    return final_class, ensemble_confidence, per_model, combined_probs
