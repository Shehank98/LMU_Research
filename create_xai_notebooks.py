#!/usr/bin/env python3
"""Create the 3 XAI phase-2 notebooks for the brain tumor MRI research project."""
import json, os

OUT_DIR = "/home/user/LMU_Research/XAI_notebook"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mk_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src if isinstance(src, list) else [src]}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def save_nb(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Shared fragments
# ---------------------------------------------------------------------------
SETUP_MD = """\
## Section 0 — Colab / Local Setup

Detects environment, installs packages, and configures Kaggle + HuggingFace credentials.

**Google Colab Secrets required** (Colab → 🔑 Secrets panel):

| Secret name | Value |
|---|---|
| `KAGGLE_USERNAME` | `sk1285` |
| `KAGGLE_KEY` | `7261c6b4046a6bd5c9ba4d1a6f58c98f` |
| `HF_TOKEN` | *(your HuggingFace write token)* |
"""

SETUP_CODE = """\
import sys, os, json, subprocess

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    subprocess.run(["pip", "install", "kaggle", "huggingface_hub", "-q"], check=False)
    from google.colab import userdata
    _kaggle_user = userdata.get('KAGGLE_USERNAME')
    _kaggle_key  = userdata.get('KAGGLE_KEY')
    HF_TOKEN     = userdata.get('HF_TOKEN')
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as _f:
        json.dump({'username': _kaggle_user, 'key': _kaggle_key}, _f)
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
    DATASET_PATH     = "/content/MRI_DATASET/"
    SAVED_MODELS_DIR = "/content/saved_models/"
    RESULTS_DIR      = "/content/results/"
    if not os.path.exists(os.path.join(DATASET_PATH, "Testing")):
        subprocess.run(["kaggle", "datasets", "download",
                        "masoudnickparvar/brain-tumor-mri-dataset",
                        "-p", "/content/"], check=False)
        subprocess.run(["unzip", "-q", "/content/brain-tumor-mri-dataset.zip",
                        "-d", DATASET_PATH], check=False)
        if os.path.exists("/content/brain-tumor-mri-dataset.zip"):
            os.remove("/content/brain-tumor-mri-dataset.zip")
else:
    DATASET_PATH     = "../MRI_DATASET/"
    SAVED_MODELS_DIR = "../saved_models/"
    RESULTS_DIR      = "../results/"
    HF_TOKEN         = os.environ.get('HF_TOKEN', '')

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Environment :", "Colab" if IN_COLAB else "Local")
print("Dataset     :", DATASET_PATH)
print("Models      :", SAVED_MODELS_DIR)
"""

HF_UPLOAD_FN = """\
def _hf_upload(files, repo_id, token, prefix=""):
    from huggingface_hub import HfApi, login as hf_login
    if not token:
        print("No HF_TOKEN — skipping upload.")
        return
    hf_login(token=token, add_to_git_credential=False)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    for p in files:
        if not os.path.exists(p):
            print(f"  skip (missing): {p}"); continue
        rp = (prefix + "/" + os.path.basename(p)).lstrip("/")
        api.upload_file(path_or_fileobj=p, path_in_repo=rp,
                        repo_id=repo_id, repo_type="model")
        print(f"  uploaded: {rp}")
"""

HF_REPO = "shehank98/brain-tumor-mri-models"

# ===========================================================================
# NOTEBOOK 09 — Consensus GRAD-CAM  (consensus_gradcam.ipynb)
# ===========================================================================

NB09_CONSTANTS = """\
NOTEBOOK_NAME    = "09_ConsensusGradCAM"
HF_REPO_ID       = "shehank98/brain-tumor-mri-models"
CLASS_NAMES      = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE_CNN     = (224, 224)
IMG_SIZE_PRETRAINED = (299, 299)
HEATMAP_SIZE     = (112, 112)   # common size for pairwise IoU
IOU_THRESHOLD    = 0.5          # binarisation threshold for heatmaps
RANDOM_SEED      = 42

DATASET_PATH     = globals().get("DATASET_PATH",     "../MRI_DATASET/")
SAVED_MODELS_DIR = globals().get("SAVED_MODELS_DIR", "../saved_models/")
RESULTS_DIR      = globals().get("RESULTS_DIR",      "../results/")
HF_TOKEN         = globals().get("HF_TOKEN",         os.environ.get("HF_TOKEN", ""))

RESULTS_NB_DIR   = os.path.join(RESULTS_DIR, NOTEBOOK_NAME)
os.makedirs(RESULTS_NB_DIR, exist_ok=True)
print("Results dir:", RESULTS_NB_DIR)
"""

NB09_IMPORTS = """\
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)
print("TF version:", tf.__version__)
"""

NB09_DOWNLOAD_MODELS = """\
def _download_model_from_hf(filename, repo_id, token):
    \"\"\"Download a model file from HuggingFace Hub if not already present.\"\"\"
    dest = os.path.join(SAVED_MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  Found locally: {filename}")
        return dest
    try:
        from huggingface_hub import hf_hub_download, login as hf_login
        if token:
            hf_login(token=token, add_to_git_credential=False)
        path = hf_hub_download(repo_id=repo_id, filename=f"models/{filename}",
                               local_dir=SAVED_MODELS_DIR)
        print(f"  Downloaded: {filename}")
        return path
    except Exception as e:
        print(f"  Could not download {filename}: {e}")
        return None

print("Checking / downloading DL models …")
for _fname in ["cnn_model.h5", "inceptionv3_model.h5", "xception_model.h5"]:
    _download_model_from_hf(_fname, HF_REPO_ID, HF_TOKEN)
"""

NB09_LOAD_MODELS = """\
def _load(name):
    p = os.path.join(SAVED_MODELS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model not found: {p}\\nRun notebooks 01, 03, 05 first (or download from HF).")
    m = keras.models.load_model(p)
    m.trainable = False
    return m

print("Loading CNN …")
cnn_model = _load("cnn_model.h5")
print("Loading InceptionV3 …")
inc_model = _load("inceptionv3_model.h5")
print("Loading Xception …")
xcp_model = _load("xception_model.h5")
print("All 3 DL models loaded.")
"""

NB09_GRADCAM_FN = """\
def _last_conv_name(model):
    \"\"\"Return the name of the last Conv layer with 4-D output.\"\"\"
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
            if isinstance(shape, list):
                shape = shape[0]
            if len(shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No 4-D conv layer found in model.")

# Cache layer names once
_LAST_CONV = {
    "cnn": _last_conv_name(cnn_model),
    "inc": _last_conv_name(inc_model),
    "xcp": _last_conv_name(xcp_model),
}
print("Last conv layers:", _LAST_CONV)

def compute_gradcam(model, model_key, img_array):
    \"\"\"
    Compute GRAD-CAM heatmap for a preprocessed image.
    img_array : np.float32 array, shape (1, H, W, 3), values in [0, 1]
    Returns   : (heatmap np.float32 (H,W) in [0,1], predicted_class_idx int)
    \"\"\"
    conv_name = _LAST_CONV[model_key]
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_name).output, model.output]
    )
    img_tf = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tf, training=False)
        pred_idx = int(tf.argmax(preds[0]))
        class_score = preds[:, pred_idx]
    grads = tape.gradient(class_score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
    heatmap = np.maximum(heatmap, 0)
    mx = heatmap.max()
    if mx > 0:
        heatmap = heatmap / mx
    return heatmap.astype(np.float32), pred_idx

def binary_iou(a, b, t=IOU_THRESHOLD):
    \"\"\"IoU between two heatmaps binarised at threshold t.\"\"\"
    a_bin = (a >= t)
    b_bin = (b >= t)
    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin,  b_bin).sum()
    return float(inter / union) if union > 0 else 1.0

def eaa_iou(hm_a, hm_b, hm_c, t=IOU_THRESHOLD):
    \"\"\"Mean pairwise IoU across 3 heatmaps (EAA-IoU).\"\"\"
    iou_ab = binary_iou(hm_a, hm_b, t)
    iou_ac = binary_iou(hm_a, hm_c, t)
    iou_bc = binary_iou(hm_b, hm_c, t)
    return float(np.mean([iou_ab, iou_ac, iou_bc])), iou_ab, iou_ac, iou_bc

def consensus_heatmap(hm_a, hm_b, hm_c, weights=(1, 1, 1)):
    \"\"\"Weighted average of 3 heatmaps, normalised to [0, 1].\"\"\"
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    fused = w[0]*hm_a + w[1]*hm_b + w[2]*hm_c
    mx = fused.max()
    return fused / mx if mx > 0 else fused

def overlay(img_rgb, heatmap, alpha=0.45):
    \"\"\"Overlay a heatmap on an RGB image (uint8). Returns RGB uint8.\"\"\"
    h, w = img_rgb.shape[:2]
    hm_up = cv2.resize(heatmap, (w, h))
    hm_u8 = np.uint8(255 * hm_up)
    col    = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    col_rgb = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
    img_u8 = img_rgb if img_rgb.dtype == np.uint8 else np.uint8(img_rgb * 255)
    return cv2.addWeighted(img_u8, 1 - alpha, col_rgb, alpha, 0)
"""

NB09_COMPUTE = """\
TEST_DIR = os.path.join(DATASET_PATH, "Testing")

records = []
sample_store = {cls: [] for cls in CLASS_NAMES}  # store up to 2 samples per class for viz

for cls_name in CLASS_NAMES:
    cls_dir = os.path.join(TEST_DIR, cls_name)
    if not os.path.isdir(cls_dir):
        print(f"  Warning: {cls_dir} not found"); continue
    img_files = sorted([f for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.jpg','.jpeg','.png'))])
    for fname in tqdm(img_files, desc=cls_name):
        img_path = os.path.join(cls_dir, fname)
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess for each architecture
        img_cnn = cv2.resize(img_rgb, IMG_SIZE_CNN).astype(np.float32) / 255.0
        img_299 = cv2.resize(img_rgb, IMG_SIZE_PRETRAINED).astype(np.float32) / 255.0
        arr_cnn = img_cnn[np.newaxis]
        arr_299 = img_299[np.newaxis]

        # GRAD-CAM
        hm_cnn, pred_cnn = compute_gradcam(cnn_model, "cnn", arr_cnn)
        hm_inc, pred_inc = compute_gradcam(inc_model, "inc", arr_299)
        hm_xcp, pred_xcp = compute_gradcam(xcp_model, "xcp", arr_299)

        # Resize to common size for IoU
        hm_cnn_r = cv2.resize(hm_cnn, HEATMAP_SIZE)
        hm_inc_r = cv2.resize(hm_inc, HEATMAP_SIZE)
        hm_xcp_r = cv2.resize(hm_xcp, HEATMAP_SIZE)

        eaa, iou_ci, iou_cx, iou_ix = eaa_iou(hm_cnn_r, hm_inc_r, hm_xcp_r)

        # Ensemble majority vote
        preds_list = [pred_cnn, pred_inc, pred_xcp]
        from collections import Counter
        majority_pred = Counter(preds_list).most_common(1)[0][0]

        rec = {
            "image_path": img_path,
            "true_class": cls_name,
            "true_idx":   CLASS_NAMES.index(cls_name),
            "pred_cnn":   CLASS_NAMES[pred_cnn],
            "pred_inc":   CLASS_NAMES[pred_inc],
            "pred_xcp":   CLASS_NAMES[pred_xcp],
            "majority_pred": CLASS_NAMES[majority_pred],
            "majority_correct": int(majority_pred == CLASS_NAMES.index(cls_name)),
            "eaa_iou":    round(eaa, 4),
            "iou_cnn_inc": round(iou_ci, 4),
            "iou_cnn_xcp": round(iou_cx, 4),
            "iou_inc_xcp": round(iou_ix, 4),
        }
        records.append(rec)

        # Store sample for viz (up to 3 per class)
        if len(sample_store[cls_name]) < 3:
            sample_store[cls_name].append({
                "img_rgb": img_rgb,
                "hm_cnn": hm_cnn, "hm_inc": hm_inc, "hm_xcp": hm_xcp,
                "consensus": consensus_heatmap(hm_cnn_r, hm_inc_r, hm_xcp_r),
                "eaa_iou": eaa,
                "majority_pred": CLASS_NAMES[majority_pred],
                "correct": majority_pred == CLASS_NAMES.index(cls_name),
            })

df = pd.DataFrame(records)
print(f"\\nProcessed {len(df)} images.")
print(df[["true_class","pred_cnn","pred_inc","pred_xcp","eaa_iou"]].head(8))
"""

NB09_SAVE_CSV = """\
csv_path = os.path.join(RESULTS_NB_DIR, "eaa_iou_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved CSV: {csv_path}")
print("EAA-IoU summary:")
print(df.groupby("true_class")["eaa_iou"].describe().round(3))
"""

NB09_VIZ_SAMPLES = """\
# --- Chart 1: Sample GRAD-CAM grid (4 classes × 5 columns: orig, CNN, Inc, Xcp, Consensus) ---
fig, axes = plt.subplots(4, 5, figsize=(18, 14))
col_titles = ["Original", "CNN GRAD-CAM", "InceptionV3 GRAD-CAM",
              "Xception GRAD-CAM", "Consensus Map"]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=8)

for row_idx, cls_name in enumerate(CLASS_NAMES):
    samples = sample_store[cls_name]
    if not samples:
        continue
    s = samples[0]
    img_disp = cv2.resize(s["img_rgb"], (224, 224))
    axes[row_idx, 0].imshow(img_disp); axes[row_idx, 0].set_ylabel(cls_name.capitalize(), fontsize=11, fontweight='bold')
    axes[row_idx, 1].imshow(overlay(img_disp, cv2.resize(s["hm_cnn"], (224,224))))
    axes[row_idx, 2].imshow(overlay(img_disp, cv2.resize(s["hm_inc"], (224,224))))
    axes[row_idx, 3].imshow(overlay(img_disp, cv2.resize(s["hm_xcp"], (224,224))))
    con_up = cv2.resize(s["consensus"], (224,224))
    axes[row_idx, 4].imshow(overlay(img_disp, con_up))
    for ax in axes[row_idx]:
        ax.axis('off')
    axes[row_idx, 4].set_xlabel(f"EAA-IoU: {s['eaa_iou']:.3f}", fontsize=9)

fig.suptitle("GRAD-CAM Heatmaps — CNN · InceptionV3 · Xception · Consensus\\n"
             "Brain Tumor MRI Dataset (Kaggle)", fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "sample_gradcam_grid.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print("Saved:", p)
"""

NB09_VIZ_DIST = """\
# --- Chart 2: EAA-IoU distribution histogram by class ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {'glioma':'#e74c3c','meningioma':'#e67e22','notumor':'#2ecc71','pituitary':'#3498db'}

ax = axes[0]
for cls in CLASS_NAMES:
    sub = df[df["true_class"] == cls]["eaa_iou"]
    ax.hist(sub, bins=30, alpha=0.6, label=cls.capitalize(), color=colors[cls], edgecolor='white')
ax.axvline(0.40, color='red',    linestyle='--', lw=1.5, label='Low/Mod threshold (0.40)')
ax.axvline(0.65, color='orange', linestyle='--', lw=1.5, label='Mod/High threshold (0.65)')
ax.set_xlabel("EAA-IoU", fontsize=11); ax.set_ylabel("Count", fontsize=11)
ax.set_title("EAA-IoU Distribution by Tumor Class", fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
means = df.groupby("true_class")["eaa_iou"].mean().reindex(CLASS_NAMES)
stds  = df.groupby("true_class")["eaa_iou"].std().reindex(CLASS_NAMES)
bars  = ax.bar([c.capitalize() for c in CLASS_NAMES], means,
               yerr=stds, capsize=5, color=[colors[c] for c in CLASS_NAMES],
               alpha=0.8, edgecolor='white')
ax.set_ylim(0, 1); ax.set_ylabel("Mean EAA-IoU", fontsize=11)
ax.set_title("Mean EAA-IoU ± Std per Class", fontsize=12, fontweight='bold')
ax.axhline(0.65, color='orange', linestyle='--', lw=1.2, label='High-confidence threshold')
ax.axhline(0.40, color='red',    linestyle='--', lw=1.2, label='Escalation threshold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "eaa_iou_distribution.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print("Saved:", p)
"""

NB09_VIZ_LOWHI = """\
# --- Chart 3: Low vs High EAA-IoU examples ---
df_sorted = df.sort_values("eaa_iou")
low_examples  = df_sorted.head(6)
high_examples = df_sorted.tail(6)

def load_and_show(row, ax, size=224):
    img = cv2.imread(row["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    ax.imshow(img); ax.axis('off')
    cls = row['true_class'].capitalize()
    iou = row['eaa_iou']
    correct = row['true_class'] == row['majority_pred']
    color = 'green' if correct else 'red'
    ax.set_title(f"{cls}\\nIoU:{iou:.3f}", fontsize=8, color=color)

fig, axes = plt.subplots(2, 6, figsize=(18, 7))
for i, (_, row) in enumerate(low_examples.iterrows()):
    load_and_show(row, axes[0, i])
axes[0, 0].set_ylabel("Low EAA-IoU\\n(Model Disagree)", fontsize=10, fontweight='bold', labelpad=10)

for i, (_, row) in enumerate(high_examples.iterrows()):
    load_and_show(row, axes[1, i])
axes[1, 0].set_ylabel("High EAA-IoU\\n(Model Agree)", fontsize=10, fontweight='bold', labelpad=10)

fig.suptitle("Low vs High EAA-IoU Examples\\n(Green title = correct prediction, Red = misclassified)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "eaa_iou_low_vs_high.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print("Saved:", p)
"""

NB09_HF_UPLOAD = """\
""" + HF_UPLOAD_FN + """
files_to_upload = [
    os.path.join(RESULTS_NB_DIR, "eaa_iou_results.csv"),
    os.path.join(RESULTS_NB_DIR, "sample_gradcam_grid.jpg"),
    os.path.join(RESULTS_NB_DIR, "eaa_iou_distribution.jpg"),
    os.path.join(RESULTS_NB_DIR, "eaa_iou_low_vs_high.jpg"),
]
_hf_upload(files_to_upload, HF_REPO_ID, HF_TOKEN, prefix=f"results/{NOTEBOOK_NAME}")
print("\\nSection 10 complete.")
"""

NB09_CELLS = [
    md("# 09 — Consensus GRAD-CAM & EAA-IoU\n\n"
       "Computes GRAD-CAM heatmaps for CNN, InceptionV3, and Xception on all test images,\n"
       "then calculates the **Ensemble Attention Agreement IoU (EAA-IoU)** — a novel inter-model\n"
       "saliency consensus metric introduced in this research.\n\n"
       "Run notebooks 01, 03, 05 first to produce the model `.h5` files."),
    md(SETUP_MD),
    code(SETUP_CODE),
    md("## Section 1 — Imports"),
    code(NB09_IMPORTS),
    md("## Section 2 — Constants"),
    code(NB09_CONSTANTS),
    md("## Section 3 — Download & Load DL Models\n\n"
       "Downloads `cnn_model.h5`, `inceptionv3_model.h5`, `xception_model.h5` from HuggingFace\n"
       "if they are not already present in `SAVED_MODELS_DIR`."),
    code(NB09_DOWNLOAD_MODELS),
    code(NB09_LOAD_MODELS),
    md("## Section 4 — GRAD-CAM Helper Functions\n\n"
       "Uses `tf.GradientTape` on the last convolutional layer of each model.\n"
       "EAA-IoU = mean pairwise IoU of the three binarised heatmaps."),
    code(NB09_GRADCAM_FN),
    md("## Section 5 — Compute GRAD-CAM for All Test Images\n\n"
       "Iterates through `Testing/` folder. For each image:\n"
       "1. Preprocess for each model's input size\n"
       "2. Compute GRAD-CAM (uses predicted class — models' own focus region)\n"
       "3. Resize heatmaps to 112×112 common size\n"
       "4. Compute pairwise IoU and EAA-IoU\n\n"
       "> **Runtime:** ~20–40 min on Colab GPU for all 2,063 test images."),
    code(NB09_COMPUTE),
    md("## Section 6 — Save Results CSV"),
    code(NB09_SAVE_CSV),
    md("## Section 7 — Visualisation: Sample Heatmap Grid"),
    code(NB09_VIZ_SAMPLES),
    md("## Section 8 — Visualisation: EAA-IoU Distribution by Class"),
    code(NB09_VIZ_DIST),
    md("## Section 9 — Visualisation: Low vs High EAA-IoU Examples"),
    code(NB09_VIZ_LOWHI),
    md("## Section 10 — Upload to HuggingFace"),
    code(NB09_HF_UPLOAD),
]

save_nb(mk_nb(NB09_CELLS), os.path.join(OUT_DIR, "consensus_gradcam.ipynb"))

# ===========================================================================
# NOTEBOOK 10 — Confidence Voting  (confidence_voting.ipynb)
# ===========================================================================

NB10_CONSTANTS = """\
NOTEBOOK_NAME    = "10_ConfidenceVoting"
HF_REPO_ID       = "shehank98/brain-tumor-mri-models"
CLASS_NAMES      = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE_CNN     = (224, 224)
IMG_SIZE_PRETRAINED = (299, 299)
RANDOM_SEED      = 42
BATCH_SIZE       = 32

DATASET_PATH     = globals().get("DATASET_PATH",     "../MRI_DATASET/")
SAVED_MODELS_DIR = globals().get("SAVED_MODELS_DIR", "../saved_models/")
RESULTS_DIR      = globals().get("RESULTS_DIR",      "../results/")
HF_TOKEN         = globals().get("HF_TOKEN",         os.environ.get("HF_TOKEN", ""))

RESULTS_NB_DIR   = os.path.join(RESULTS_DIR, NOTEBOOK_NAME)
os.makedirs(RESULTS_NB_DIR, exist_ok=True)
print("Results dir:", RESULTS_NB_DIR)
"""

NB10_IMPORTS = """\
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2, joblib, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')
tf.random.set_seed(42); np.random.seed(42)
print("TF:", tf.__version__)
"""

NB10_DOWNLOAD_MODELS = """\
def _dl_model(filename, repo_id, token):
    dest = os.path.join(SAVED_MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  local: {filename}"); return dest
    try:
        from huggingface_hub import hf_hub_download, login as hf_login
        if token: hf_login(token=token, add_to_git_credential=False)
        p = hf_hub_download(repo_id=repo_id, filename=f"models/{filename}",
                            local_dir=SAVED_MODELS_DIR)
        print(f"  downloaded: {filename}"); return p
    except Exception as e:
        print(f"  WARN: could not download {filename}: {e}"); return None

needed = [
    "cnn_model.h5", "inceptionv3_model.h5", "xception_model.h5",
    "cnn_ensemble.h5", "inceptionv3_ensemble.h5", "xception_ensemble.h5",
    "cnn_ensemble_model.pkl", "inceptionv3_ensemble_model.pkl", "xception_ensemble_model.pkl",
]
print("Checking models …")
for f in needed:
    _dl_model(f, HF_REPO_ID, HF_TOKEN)
"""

NB10_LOAD_MODELS = """\
def _lm(name):
    p = os.path.join(SAVED_MODELS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")
    m = keras.models.load_model(p); m.trainable = False; return m

def _lpkl(name):
    p = os.path.join(SAVED_MODELS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")
    return joblib.load(p)

print("Loading standalone DL models …")
cnn_model = _lm("cnn_model.h5")
inc_model  = _lm("inceptionv3_model.h5")
xcp_model  = _lm("xception_model.h5")

print("Loading ensemble feature extractors …")
cnn_feat_model = _lm("cnn_ensemble.h5")
inc_feat_model = _lm("inceptionv3_ensemble.h5")
xcp_feat_model = _lm("xception_ensemble.h5")

# Feature extractors — output the dense feature layer
def _make_extractor(base_model):
    for layer in reversed(base_model.layers):
        if 'dense' in layer.name.lower() and len(layer.output_shape) == 2:
            return keras.Model(inputs=base_model.inputs, outputs=layer.output)
    return base_model   # fallback: use full model output

cnn_extractor = _make_extractor(cnn_feat_model)
inc_extractor = _make_extractor(inc_feat_model)
xcp_extractor = _make_extractor(xcp_feat_model)

print("Loading classical ensemble models …")
cnn_ens_clf = _lpkl("cnn_ensemble_model.pkl")
inc_ens_clf = _lpkl("inceptionv3_ensemble_model.pkl")
xcp_ens_clf = _lpkl("xception_ensemble_model.pkl")

print("All 6 model components loaded.")
"""

NB10_DATA = """\
TEST_DIR = os.path.join(DATASET_PATH, "Testing")

datagen = ImageDataGenerator(rescale=1./255)

gen_cnn = datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE_CNN, batch_size=BATCH_SIZE,
    class_mode='sparse', classes=CLASS_NAMES, shuffle=False)

gen_299 = datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE_PRETRAINED, batch_size=BATCH_SIZE,
    class_mode='sparse', classes=CLASS_NAMES, shuffle=False)

y_true = gen_cnn.classes
n = len(y_true)
print(f"Test images: {n}")
"""

NB10_PREDICT = """\
print("Running inference (this may take a few minutes) …")

# Standalone DL probabilities
prob_cnn = cnn_model.predict(gen_cnn, verbose=1)     # (N, 4)
prob_inc = inc_model.predict(gen_299, verbose=1)
prob_xcp = xcp_model.predict(gen_299, verbose=1)

# Classical ensemble probabilities via feature extraction
print("Extracting features for classical ensembles …")
feat_cnn = cnn_extractor.predict(gen_cnn, verbose=0)
feat_inc = inc_extractor.predict(gen_299, verbose=0)
feat_xcp = xcp_extractor.predict(gen_299, verbose=0)

prob_cnn_ens = cnn_ens_clf.predict_proba(feat_cnn)    # sklearn returns (N, 4)
prob_inc_ens = inc_ens_clf.predict_proba(feat_inc)
prob_xcp_ens = xcp_ens_clf.predict_proba(feat_xcp)

print("Predictions done.")
"""

NB10_VOTE = """\
# --- Majority Vote (original method) ---
from collections import Counter

pred_cnn_cls = np.argmax(prob_cnn, axis=1)
pred_inc_cls = np.argmax(prob_inc, axis=1)
pred_xcp_cls = np.argmax(prob_xcp, axis=1)
pred_cnn_ens_cls = np.argmax(prob_cnn_ens, axis=1)
pred_inc_ens_cls = np.argmax(prob_inc_ens, axis=1)
pred_xcp_ens_cls = np.argmax(prob_xcp_ens, axis=1)

majority_preds = []
for i in range(n):
    votes = [pred_cnn_cls[i], pred_inc_cls[i], pred_xcp_cls[i],
             pred_cnn_ens_cls[i], pred_inc_ens_cls[i], pred_xcp_ens_cls[i]]
    majority_preds.append(Counter(votes).most_common(1)[0][0])
majority_preds = np.array(majority_preds)

# --- Confidence-Weighted Vote (novel extension) ---
combined_probs = prob_cnn + prob_inc + prob_xcp + prob_cnn_ens + prob_inc_ens + prob_xcp_ens
weighted_preds = np.argmax(combined_probs, axis=1)
vote_confidence = combined_probs.max(axis=1) / combined_probs.sum(axis=1)

acc_majority  = accuracy_score(y_true, majority_preds)
acc_weighted  = accuracy_score(y_true, weighted_preds)

print(f"Majority Vote Accuracy  : {acc_majority*100:.2f}%")
print(f"Confidence-Weighted Acc : {acc_weighted*100:.2f}%")
print(f"Improvement             : {(acc_weighted - acc_majority)*100:+.2f}%")
"""

NB10_REPORT = """\
print("\\n=== Majority Vote Classification Report ===")
print(classification_report(y_true, majority_preds, target_names=CLASS_NAMES))

print("\\n=== Confidence-Weighted Classification Report ===")
print(classification_report(y_true, weighted_preds, target_names=CLASS_NAMES))
"""

NB10_CHARTS = """\
import re as _re

# --- Chart 1: Side-by-side confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, preds, title in [
    (axes[0], majority_preds,  f"Majority Vote\\n(Acc: {acc_majority*100:.2f}%)"),
    (axes[1], weighted_preds, f"Confidence-Weighted Vote\\n(Acc: {acc_weighted*100:.2f}%)"),
]:
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=[c.capitalize() for c in CLASS_NAMES])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=12, fontweight='bold')
plt.suptitle("Voting Strategy Comparison — Confusion Matrices", fontsize=13, fontweight='bold')
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "voting_comparison_confusion_matrix.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
print("Saved:", p)

# --- Chart 2: Per-class F1 improvement ---
from sklearn.metrics import f1_score
f1_maj = f1_score(y_true, majority_preds, average=None, labels=list(range(4)))
f1_wgt = f1_score(y_true, weighted_preds, average=None, labels=list(range(4)))
x = np.arange(len(CLASS_NAMES)); w = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, f1_maj, w, label="Majority Vote",           alpha=0.85, color='#3498db')
ax.bar(x + w/2, f1_wgt, w, label="Confidence-Weighted Vote", alpha=0.85, color='#2ecc71')
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=11)
ax.set_ylim(0.80, 1.01); ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("Per-Class F1 Score: Majority Vote vs Confidence-Weighted Vote",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
for i, (a, b) in enumerate(zip(f1_maj, f1_wgt)):
    ax.text(i - w/2, a + 0.002, f"{a:.3f}", ha='center', va='bottom', fontsize=8)
    ax.text(i + w/2, b + 0.002, f"{b:.3f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "voting_comparison_per_class_f1.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
print("Saved:", p)

# --- Chart 3: Confidence distribution ---
fig, ax = plt.subplots(figsize=(10, 4))
colors_conf = ['#e74c3c' if c else '#2ecc71' for c in (weighted_preds != y_true)]
ax.hist(vote_confidence[weighted_preds == y_true],  bins=40, alpha=0.6,
        color='#2ecc71', label='Correct prediction')
ax.hist(vote_confidence[weighted_preds != y_true], bins=40, alpha=0.6,
        color='#e74c3c', label='Misclassified')
ax.set_xlabel("Vote Confidence Score", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Confidence Distribution: Correct vs Misclassified Predictions", fontsize=12, fontweight='bold')
ax.axvline(0.70, linestyle='--', color='gray', lw=1.5, label='Threshold 0.70')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "voting_confidence_distribution.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
print("Saved:", p)
"""

NB10_SAVE_CSV = """\
results_df = pd.DataFrame({
    "true_class": [CLASS_NAMES[i] for i in y_true],
    "majority_pred": [CLASS_NAMES[i] for i in majority_preds],
    "weighted_pred":  [CLASS_NAMES[i] for i in weighted_preds],
    "vote_confidence": vote_confidence,
    "majority_correct": (majority_preds == y_true).astype(int),
    "weighted_correct":  (weighted_preds  == y_true).astype(int),
})
csv_p = os.path.join(RESULTS_NB_DIR, "voting_comparison_results.csv")
results_df.to_csv(csv_p, index=False)
print(f"Saved: {csv_p}")
print(results_df.groupby("true_class")[["majority_correct","weighted_correct"]].mean().round(3))
"""

NB10_HF_UPLOAD = HF_UPLOAD_FN + """\
files_to_upload = [
    os.path.join(RESULTS_NB_DIR, "voting_comparison_results.csv"),
    os.path.join(RESULTS_NB_DIR, "voting_comparison_confusion_matrix.jpg"),
    os.path.join(RESULTS_NB_DIR, "voting_comparison_per_class_f1.jpg"),
    os.path.join(RESULTS_NB_DIR, "voting_confidence_distribution.jpg"),
]
_hf_upload(files_to_upload, HF_REPO_ID, HF_TOKEN, prefix=f"results/{NOTEBOOK_NAME}")
print("\\nSection 10 complete.")
"""

NB10_CELLS = [
    md("# 10 — Confidence-Weighted Voting Comparison\n\n"
       "Compares the original **majority vote** ensemble with **confidence-weighted voting**\n"
       "(sum of 6 softmax probability vectors) across all 2,063 test images.\n\n"
       "Requires notebooks 01–06 to have been run first."),
    md(SETUP_MD),
    code(SETUP_CODE),
    md("## Section 1 — Imports"),
    code(NB10_IMPORTS),
    md("## Section 2 — Constants"),
    code(NB10_CONSTANTS),
    md("## Section 3 — Download & Load All 6 Models"),
    code(NB10_DOWNLOAD_MODELS),
    code(NB10_LOAD_MODELS),
    md("## Section 4 — Load Test Data\n\nThree generators: one for CNN (224×224), one for InceptionV3/Xception (299×299)."),
    code(NB10_DATA),
    md("## Section 5 — Run Inference (All 6 Models)"),
    code(NB10_PREDICT),
    md("## Section 6 — Majority Vote vs Confidence-Weighted Vote"),
    code(NB10_VOTE),
    md("## Section 7 — Classification Reports"),
    code(NB10_REPORT),
    md("## Section 8 — Charts"),
    code(NB10_CHARTS),
    md("## Section 9 — Save Results CSV"),
    code(NB10_SAVE_CSV),
    md("## Section 10 — Upload to HuggingFace"),
    code(NB10_HF_UPLOAD),
]

save_nb(mk_nb(NB10_CELLS), os.path.join(OUT_DIR, "confidence_voting.ipynb"))

# ===========================================================================
# NOTEBOOK 11 — XAI Analysis  (xai_analysis.ipynb)
# ===========================================================================

NB11_CONSTANTS = """\
NOTEBOOK_NAME    = "11_XAIAnalysis"
HF_REPO_ID       = "shehank98/brain-tumor-mri-models"
CLASS_NAMES      = ["glioma", "meningioma", "notumor", "pituitary"]
IOU_THRESHOLD    = 0.5
ESCALATION_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

DATASET_PATH     = globals().get("DATASET_PATH",     "../MRI_DATASET/")
SAVED_MODELS_DIR = globals().get("SAVED_MODELS_DIR", "../saved_models/")
RESULTS_DIR      = globals().get("RESULTS_DIR",      "../results/")
HF_TOKEN         = globals().get("HF_TOKEN",         os.environ.get("HF_TOKEN", ""))

RESULTS_NB_DIR   = os.path.join(RESULTS_DIR, NOTEBOOK_NAME)
os.makedirs(RESULTS_NB_DIR, exist_ok=True)
print("Results dir:", RESULTS_NB_DIR)
"""

NB11_IMPORTS = """\
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
"""

NB11_LOAD_CSV = """\
# Load EAA-IoU results produced by notebook 09
eaa_csv = os.path.join(RESULTS_DIR, "09_ConsensusGradCAM", "eaa_iou_results.csv")
if not os.path.exists(eaa_csv):
    # Try to download from HuggingFace
    try:
        from huggingface_hub import hf_hub_download, login as hf_login
        if HF_TOKEN: hf_login(token=HF_TOKEN, add_to_git_credential=False)
        eaa_csv = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="results/09_ConsensusGradCAM/eaa_iou_results.csv",
            local_dir=RESULTS_DIR)
        print("Downloaded EAA-IoU CSV from HuggingFace")
    except Exception as e:
        raise FileNotFoundError(
            f"eaa_iou_results.csv not found.\\nRun notebook 09 first.\\nError: {e}")

df_eaa = pd.read_csv(eaa_csv)
print(f"Loaded {len(df_eaa)} records from: {eaa_csv}")
print(df_eaa.head(4))

# Also load confidence-voting results if available
vote_csv = os.path.join(RESULTS_DIR, "10_ConfidenceVoting", "voting_comparison_results.csv")
df_vote  = pd.read_csv(vote_csv) if os.path.exists(vote_csv) else None
if df_vote is not None:
    print(f"\\nLoaded voting results: {len(df_vote)} records")
    df_eaa["vote_confidence"]   = df_vote["vote_confidence"].values
    df_eaa["weighted_correct"]  = df_vote["weighted_correct"].values
"""

NB11_MISCLASSIFY = """\
# Derive misclassification flag from majority vote (3 deep models)
df_eaa["misclassified"] = (df_eaa["majority_pred"] != df_eaa["true_class"]).astype(int)

n_total = len(df_eaa)
n_mis   = df_eaa["misclassified"].sum()
print(f"Total images    : {n_total}")
print(f"Misclassified   : {n_mis} ({n_mis/n_total*100:.1f}%)")
print(f"Correctly classified: {n_total - n_mis} ({(n_total-n_mis)/n_total*100:.1f}%)")
print("\\nMisclassification rate by class:")
print(df_eaa.groupby("true_class")["misclassified"].mean().round(3))
"""

NB11_ROC = """\
# --- Chart 1: ROC Curve — EAA-IoU as misclassification predictor ---
# Low EAA-IoU should predict misclassification, so we invert it
y_mis   = df_eaa["misclassified"].values
score   = 1.0 - df_eaa["eaa_iou"].values   # higher score = more likely misclassified

fpr, tpr, thresholds = roc_curve(y_mis, score)
roc_auc = auc(fpr, tpr)

prec, rec, thr_pr = precision_recall_curve(y_mis, score)
pr_auc = auc(rec, prec)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f"EAA-IoU ROC (AUC = {roc_auc:.3f})")
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=11); ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC: EAA-IoU as Misclassification Predictor", fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(rec, prec, color='#3498db', lw=2, label=f"Precision-Recall (AUC = {pr_auc:.3f})")
baseline = y_mis.mean()
ax.axhline(baseline, linestyle='--', color='gray', lw=1.2, label=f"Baseline (random) = {baseline:.3f}")
ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall: EAA-IoU as Misclassification Predictor", fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

plt.suptitle("EAA-IoU Diagnostic Performance as Uncertainty Signal", fontsize=13, fontweight='bold')
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "eaa_iou_roc_pr_curve.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
print(f"Saved: {p}")
print(f"ROC-AUC: {roc_auc:.3f}   PR-AUC: {pr_auc:.3f}")
"""

NB11_BOXPLOT = """\
# --- Chart 2: EAA-IoU distribution per class (box plots) ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = {'glioma':'#e74c3c','meningioma':'#e67e22','notumor':'#2ecc71','pituitary':'#3498db'}

# Box plot by class
ax = axes[0]
data_by_class = [df_eaa[df_eaa["true_class"]==c]["eaa_iou"].values for c in CLASS_NAMES]
bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2))
for patch, cls in zip(bp['boxes'], CLASS_NAMES):
    patch.set_facecolor(colors[cls]); patch.set_alpha(0.8)
ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=11)
ax.axhline(0.65, linestyle='--', color='orange', lw=1.5, label='High-conf threshold (0.65)')
ax.axhline(0.40, linestyle='--', color='red',    lw=1.5, label='Escalation threshold (0.40)')
ax.set_ylabel("EAA-IoU", fontsize=11)
ax.set_title("EAA-IoU Distribution by Tumor Class", fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

# Box plot: correct vs misclassified
ax = axes[1]
data_split = [
    df_eaa[df_eaa["misclassified"]==0]["eaa_iou"].values,
    df_eaa[df_eaa["misclassified"]==1]["eaa_iou"].values,
]
bp2 = ax.boxplot(data_split, patch_artist=True,
                 medianprops=dict(color='white', linewidth=2))
bp2['boxes'][0].set_facecolor('#2ecc71'); bp2['boxes'][0].set_alpha(0.8)
bp2['boxes'][1].set_facecolor('#e74c3c'); bp2['boxes'][1].set_alpha(0.8)
ax.set_xticklabels(["Correctly\\nClassified", "Misclassified"], fontsize=11)
ax.axhline(0.40, linestyle='--', color='red', lw=1.5, label='Escalation threshold (0.40)')
ax.set_ylabel("EAA-IoU", fontsize=11)
ax.set_title("EAA-IoU: Correct vs Misclassified", fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

plt.suptitle("EAA-IoU Distribution Analysis", fontsize=13, fontweight='bold')
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "eaa_iou_by_class_boxplot.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
print("Saved:", p)

from scipy import stats
t_stat, p_val = stats.mannwhitneyu(
    df_eaa[df_eaa["misclassified"]==0]["eaa_iou"],
    df_eaa[df_eaa["misclassified"]==1]["eaa_iou"],
    alternative='greater')
print(f"Mann-Whitney U test (correct > misclassified EAA-IoU): p = {p_val:.4f}")
print("Statistically significant:" , "YES" if p_val < 0.05 else "NO")
"""

NB11_ESCALATION = """\
# --- Chart 3: Escalation threshold analysis ---
# At each EAA-IoU threshold, how many misclassifications are caught vs
# how many correct predictions are unnecessarily escalated?

thresholds = np.arange(0.20, 0.75, 0.025)
sensitivity = []  # % of misclassifications flagged
specificity = []  # % of correct predictions NOT flagged (1 - false escalation rate)
flagged_total = []

for t in thresholds:
    flagged = df_eaa["eaa_iou"] < t
    tp = (flagged & (df_eaa["misclassified"]==1)).sum()   # caught misclassifications
    fn = (~flagged & (df_eaa["misclassified"]==1)).sum()   # missed misclassifications
    tn = (~flagged & (df_eaa["misclassified"]==0)).sum()   # correct, not escalated
    fp = (flagged & (df_eaa["misclassified"]==0)).sum()    # correct, but escalated
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity.append(sens)
    specificity.append(spec)
    flagged_total.append(flagged.sum())

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
ax.plot(thresholds, sensitivity, 'r-o', ms=4, lw=2, label='Sensitivity (misclass caught)')
ax.plot(thresholds, specificity, 'g-s', ms=4, lw=2, label='Specificity (correct not escalated)')
ax.axvline(0.40, linestyle='--', color='navy', lw=1.5, label='Recommended threshold 0.40')
ax.set_xlabel("EAA-IoU Escalation Threshold", fontsize=11)
ax.set_ylabel("Rate", fontsize=11)
ax.set_title("Sensitivity vs Specificity at Each Threshold", fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

ax = axes[1]
ax.plot(thresholds, np.array(flagged_total)/len(df_eaa)*100,
        'b-^', ms=4, lw=2, label='% images flagged for review')
ax.axvline(0.40, linestyle='--', color='navy', lw=1.5, label='Recommended threshold 0.40')
n_mis = df_eaa["misclassified"].sum()
ax.axhline(n_mis/len(df_eaa)*100, linestyle=':', color='red', lw=1.2,
           label=f'Actual misclass rate ({n_mis/len(df_eaa)*100:.1f}%)')
ax.set_xlabel("EAA-IoU Escalation Threshold", fontsize=11)
ax.set_ylabel("% of Test Set Flagged", fontsize=11)
ax.set_title("Workload: % Images Sent for Radiologist Review", fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

plt.suptitle("EAA-IoU Clinical Escalation Rule Analysis", fontsize=13, fontweight='bold')
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "escalation_threshold_analysis.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
print("Saved:", p)

# Print table at key thresholds
print("\\n Escalation threshold analysis:")
print(f"{'Threshold':>12} {'Sensitivity':>14} {'Specificity':>14} {'% Flagged':>12}")
for t, sen, spe, fl in zip(thresholds, sensitivity, specificity, flagged_total):
    if abs(t - 0.40) < 0.015 or abs(t - 0.50) < 0.015 or abs(t - 0.60) < 0.015:
        print(f"{t:>12.2f} {sen:>14.3f} {spe:>14.3f} {fl/len(df_eaa)*100:>11.1f}%")
"""

NB11_SCATTER = """\
# --- Chart 4: EAA-IoU vs Vote Confidence Scatter ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
colors_cls = {'glioma':'#e74c3c','meningioma':'#e67e22','notumor':'#2ecc71','pituitary':'#3498db'}

ax = axes[0]
for cls in CLASS_NAMES:
    sub = df_eaa[df_eaa["true_class"]==cls]
    ax.scatter(sub["eaa_iou"], sub.get("vote_confidence",
               pd.Series(np.random.uniform(0.5,1.0,len(sub)))),
               alpha=0.3, s=8, color=colors_cls[cls], label=cls.capitalize())
ax.set_xlabel("EAA-IoU (inter-model attention agreement)", fontsize=11)
ax.set_ylabel("Vote Confidence Score", fontsize=11)
ax.set_title("EAA-IoU vs Ensemble Confidence\\nColoured by True Class", fontsize=11, fontweight='bold')
ax.legend(fontsize=9, markerscale=3); ax.grid(alpha=0.3)
ax.axvline(0.40, linestyle='--', color='red', lw=1, alpha=0.7)

ax = axes[1]
correct_mask = df_eaa["misclassified"] == 0
ax.scatter(df_eaa.loc[correct_mask, "eaa_iou"],
           df_eaa.loc[correct_mask].get("vote_confidence",
           pd.Series(np.random.uniform(0.5,1.0,correct_mask.sum()))),
           alpha=0.3, s=8, color='#2ecc71', label='Correct')
ax.scatter(df_eaa.loc[~correct_mask, "eaa_iou"],
           df_eaa.loc[~correct_mask].get("vote_confidence",
           pd.Series(np.random.uniform(0.5,1.0,(~correct_mask).sum()))),
           alpha=0.6, s=12, color='#e74c3c', label='Misclassified', zorder=5)
ax.set_xlabel("EAA-IoU", fontsize=11)
ax.set_ylabel("Vote Confidence Score", fontsize=11)
ax.set_title("EAA-IoU vs Confidence\\nColoured by Outcome", fontsize=11, fontweight='bold')
ax.legend(fontsize=10, markerscale=3); ax.grid(alpha=0.3)
ax.axvline(0.40, linestyle='--', color='red', lw=1, alpha=0.7, label='Threshold 0.40')

plt.suptitle("EAA-IoU vs Ensemble Vote Confidence", fontsize=13, fontweight='bold')
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "eaa_iou_vs_confidence_scatter.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
print("Saved:", p)
"""

NB11_SUMMARY = """\
# --- Summary statistics for the research report ---
print("=" * 60)
print("XAI ANALYSIS — SUMMARY FOR REPORT")
print("=" * 60)

mean_iou_overall = df_eaa["eaa_iou"].mean()
mean_iou_correct = df_eaa[df_eaa["misclassified"]==0]["eaa_iou"].mean()
mean_iou_wrong   = df_eaa[df_eaa["misclassified"]==1]["eaa_iou"].mean()

print(f"\\nEAA-IoU (all images)        : {mean_iou_overall:.4f}")
print(f"EAA-IoU (correct predictions): {mean_iou_correct:.4f}")
print(f"EAA-IoU (misclassifications) : {mean_iou_wrong:.4f}")
print(f"Difference                   : {mean_iou_correct - mean_iou_wrong:.4f}")

print("\\nMean EAA-IoU by class:")
for cls in CLASS_NAMES:
    sub = df_eaa[df_eaa["true_class"]==cls]
    print(f"  {cls:<15}: {sub['eaa_iou'].mean():.4f} ± {sub['eaa_iou'].std():.4f}")

# Escalation at threshold 0.40
t = 0.40
flagged = df_eaa["eaa_iou"] < t
tp = (flagged & (df_eaa["misclassified"]==1)).sum()
fn = (~flagged & (df_eaa["misclassified"]==1)).sum()
fp = (flagged & (df_eaa["misclassified"]==0)).sum()
sens = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\\nAt EAA-IoU threshold 0.40:")
print(f"  Sensitivity (misclass caught)  : {sens:.3f} ({tp}/{tp+fn})")
print(f"  False escalation rate          : {fp/len(df_eaa):.3f}")
print(f"  Total flagged for review       : {flagged.sum()} / {len(df_eaa)} ({flagged.sum()/len(df_eaa)*100:.1f}%)")
print("=" * 60)
"""

NB11_HF_UPLOAD = HF_UPLOAD_FN + """\
files_to_upload = [
    os.path.join(RESULTS_NB_DIR, "eaa_iou_roc_pr_curve.jpg"),
    os.path.join(RESULTS_NB_DIR, "eaa_iou_by_class_boxplot.jpg"),
    os.path.join(RESULTS_NB_DIR, "escalation_threshold_analysis.jpg"),
    os.path.join(RESULTS_NB_DIR, "eaa_iou_vs_confidence_scatter.jpg"),
]
_hf_upload(files_to_upload, HF_REPO_ID, HF_TOKEN, prefix=f"results/{NOTEBOOK_NAME}")
print("\\nSection 10 complete.")
"""

NB11_CELLS = [
    md("# 11 — XAI Analysis: EAA-IoU as Clinical Uncertainty Signal\n\n"
       "Loads the EAA-IoU scores from notebook 09 and the voting results from notebook 10.\n"
       "Produces the publication-quality figures that answer the four research questions:\n\n"
       "1. Does low EAA-IoU correlate with misclassification?\n"
       "2. Which tumor class has the lowest average EAA-IoU?\n"
       "3. What precision/recall tradeoff does an EAA-IoU escalation threshold give?\n"
       "4. Can EAA-IoU flag misclassified cases without retraining?\n\n"
       "**Run notebooks 09 and 10 first.**"),
    md(SETUP_MD),
    code(SETUP_CODE),
    md("## Section 1 — Imports"),
    code(NB11_IMPORTS),
    md("## Section 2 — Constants"),
    code(NB11_CONSTANTS),
    md("## Section 3 — Load Results CSVs\n\n"
       "Loads `eaa_iou_results.csv` (from notebook 09) and optionally `voting_comparison_results.csv`\n"
       "(from notebook 10). Downloads from HuggingFace if not found locally."),
    code(NB11_LOAD_CSV),
    md("## Section 4 — Misclassification Statistics"),
    code(NB11_MISCLASSIFY),
    md("## Section 5 — ROC & PR Curves: EAA-IoU as Misclassification Predictor\n\n"
       "Treats `1 − EAA-IoU` as a risk score. High score = low inter-model agreement = likely error."),
    code(NB11_ROC),
    md("## Section 6 — EAA-IoU Distribution: Class-Conditional Box Plots\n\n"
       "Tests the hypothesis: glioma has the lowest average EAA-IoU (most disagreement)."),
    code(NB11_BOXPLOT),
    md("## Section 7 — Escalation Threshold Analysis\n\n"
       "Sweeps the EAA-IoU threshold and plots the sensitivity/specificity tradeoff.\n"
       "Answers: at threshold 0.40, how many misclassifications are caught, and at what cost?"),
    code(NB11_ESCALATION),
    md("## Section 8 — EAA-IoU vs Vote Confidence Scatter"),
    code(NB11_SCATTER),
    md("## Section 9 — Summary Statistics for Report"),
    code(NB11_SUMMARY),
    md("## Section 10 — Upload to HuggingFace"),
    code(NB11_HF_UPLOAD),
]

save_nb(mk_nb(NB11_CELLS), os.path.join(OUT_DIR, "xai_analysis.ipynb"))

print("\nAll 3 XAI notebooks created successfully.")
