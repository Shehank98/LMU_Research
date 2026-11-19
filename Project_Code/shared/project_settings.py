"""Shared helpers for the paper's analysis scripts (reproduce the training code's preprocessing exactly)."""

import os, numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "paper_assets")
VERSION = os.environ.get("MODEL_SET", "baseline")  # 'baseline' (first run) or 'improved' (02_Improved_Models)
SUFFIX = "" if VERSION == "baseline" else "_improved"
FIG = os.path.join(ASSETS, "figures" + SUFFIX)
RES = os.path.join(ASSETS, "xai_results" + SUFFIX)
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
DATA = os.environ.get("MRI_DATA", "/home/user/MRI_DATASET")
MODELS = os.environ.get(
    "MODEL_DIR",
    "/home/user/hf_models" if VERSION == "baseline" else os.path.join(ROOT, "runs", "v2", "models"),
)
IMG = 256 if VERSION == "baseline" else 224
# per-image predictions and the final confusion matrix of each configuration
PRED_CSV = (os.path.join(ROOT, "metrics", "final_per_image_predictions.csv") if VERSION == "baseline"
            else os.path.join(ROOT, "runs", "v2", "metrics", "per_image_predictions.csv"))
FINAL_CM = (os.path.join(ROOT, "confusion", "FINAL_majority_vote_confusion_matrix.csv") if VERSION == "baseline"
            else os.path.join(ROOT, "runs", "v2", "confusion", "final_majority.csv"))
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LABELS = ["Glioma", "Meningioma", "No tumour", "Pituitary"]


def load_numpy(split, size=None):
    """Test images exactly as each pipeline sees them, in sorted file order, scaled to [0, 1]."""
    if VERSION != "baseline":
        import sys

        sys.path.insert(0, os.path.join(ROOT, "Project_Code", "02_Improved_Models"))
        os.environ.setdefault("RUN_DIR", os.path.join(ROOT, "runs", "v2"))
        from data_preparation import load_split

        X, y, files = load_split(split)
        return X.astype("float32") / 255.0, y, files
    size = size or 256
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
