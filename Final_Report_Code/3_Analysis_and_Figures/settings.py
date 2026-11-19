"""Paths and the test-image loader shared by the analysis scripts.

MRI_DATA   folder with Training/ and Testing/ (Kaggle Brain Tumor MRI Dataset)
MODEL_DIR  folder with the trained models (cnn_model.h5, inceptionv3_model.h5, xception_model.h5),
           downloaded from the Hugging Face repository shehank98/brain-tumor-mri-models
"""

import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "2_Results")
METRICS = os.path.join(RESULTS, "metrics")
CONFUSION = os.path.join(RESULTS, "confusion")
HISTORY = os.path.join(RESULTS, "history")
OUT = os.path.join(HERE, "outputs")  # numbers computed by these scripts
FIG = os.path.join(HERE, "figures")  # figures used in the report
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

DATA = os.environ.get("MRI_DATA", "MRI_DATASET")
MODELS = os.environ.get("MODEL_DIR", "models")
PRED_CSV = os.path.join(METRICS, "final_per_image_predictions.csv")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LABELS = ["Glioma", "Meningioma", "No tumour", "Pituitary"]
PALETTE = ["#3B6FB6", "#E07B39", "#4E9F6E", "#B5487F"]


def load_numpy(split, size=256):
    """Images in sorted file order, resized with OpenCV, RGB, scaled to [0, 1] (same as the training notebook)."""
    X, y, files = [], [], []
    for idx, c in enumerate(CLASSES):
        folder = os.path.join(DATA, split, c)
        for fn in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_COLOR)
            if img is None:
                continue
            X.append(cv2.resize(img, (size, size)))
            y.append(idx)
            files.append(f"{c}/{fn}")
    X = np.asarray(X, dtype="float32")[..., ::-1] / 255.0
    return np.ascontiguousarray(X), np.asarray(y), files


def plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )
