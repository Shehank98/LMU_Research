"""Loading, head cropping and the train/validation split."""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from settings import DATA_DIR, OUT_DIR, CLASSES, IMG_SIZE, SEED, VAL_FRACTION


def crop_head(img, margin=4):
    """Crop an MRI slice to the bounding box of the head.

    The scans come from several collections with different borders and image sizes.
    Cropping removes most of the black background so the models cannot use it as a cue.
    """
    grey = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(grey, 45, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < 0.2 * img.shape[1] or h < 0.2 * img.shape[0]:
        return img  # thresholding failed, keep the full image
    y0, x0 = max(y - margin, 0), max(x - margin, 0)
    return img[y0 : y + h + margin, x0 : x + w + margin]


def load_split(split, size=IMG_SIZE, crop=True):
    """Return images (uint8, RGB), labels and file names for 'Training' or 'Testing'.

    Files are read in sorted order, so test indices match the saved per-image predictions.
    Results are cached as .npz because decoding 7,200 JPEGs takes a while.
    """
    cache = os.path.join(OUT_DIR, "cache", f"{split}_{size}_{int(crop)}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["y"], list(d["files"])
    X, y, files = [], [], []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, split, cls)
        for name in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if crop:
                img = crop_head(img)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            X.append(img[..., ::-1])  # BGR -> RGB
            y.append(label)
            files.append(f"{cls}/{name}")
    X, y = np.asarray(X, dtype=np.uint8), np.asarray(y)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, X=X, y=y, files=np.array(files))
    return X, y, files


def train_val_split(X, y):
    """Stratified 90/10 split of the Training folder (the Testing folder is never touched)."""
    return train_test_split(X, y, test_size=VAL_FRACTION, stratify=y, random_state=SEED)
