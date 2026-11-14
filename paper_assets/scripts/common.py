"""Shared helpers for the paper's analysis scripts (reproduce the training code's preprocessing exactly)."""
import os, numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, 'paper_assets')
FIG = os.path.join(ASSETS, 'figures'); RES = os.path.join(ASSETS, 'xai_results')
DATA = os.environ.get('MRI_DATA', '/home/user/MRI_DATASET')
MODELS = os.environ.get('MODEL_DIR', '/home/user/hf_models')
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
LABELS = ['Glioma', 'Meningioma', 'No tumour', 'Pituitary']

def load_numpy(split, size):
    """Same as load_numpy() in the training notebook: sorted file order, cv2 resize, /255, BGR->RGB."""
    X, y, files = [], [], []
    for idx, c in enumerate(CLASSES):
        folder = os.path.join(DATA, split, c)
        for fn in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_COLOR)
            if img is None: continue
            X.append(cv2.resize(img, (size, size))); y.append(idx); files.append(f'{c}/{fn}')
    X = np.asarray(X, dtype='float32')[..., ::-1] / 255.0
    return np.ascontiguousarray(X), np.asarray(y), files
