#!/usr/bin/env python3
"""Regenerate consensus_gradcam.ipynb with full 6-model GRAD-CAM support."""
import json, os

OUT_DIR = "/home/user/LMU_Research/XAI_notebook"

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

SETUP_MD = """\
## Section 0 — Colab / Local Setup

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

IMPORTS = """\
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2, itertools, tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
tf.random.set_seed(42); np.random.seed(42)
print("TF version:", tf.__version__)
"""

CONSTANTS = """\
NOTEBOOK_NAME    = "09_ConsensusGradCAM"
HF_REPO_ID       = "shehank98/brain-tumor-mri-models"
CLASS_NAMES      = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE_CNN     = (224, 224)
IMG_SIZE_PRETRAINED = (299, 299)
HEATMAP_SIZE     = (112, 112)   # common size for pairwise IoU
IOU_THRESHOLD    = 0.5

# 6 model keys: 3 standalone DL + 3 feature-extractor (ensemble backbone)
MODEL_KEYS = ["cnn", "inc", "xcp", "cnn_ens", "inc_ens", "xcp_ens"]
MODEL_LABELS = {
    "cnn":     "CNN Standalone",
    "inc":     "InceptionV3 Standalone",
    "xcp":     "Xception Standalone",
    "cnn_ens": "CNN + Ensemble",
    "inc_ens": "InceptionV3 + Ensemble",
    "xcp_ens": "Xception + Ensemble",
}

DATASET_PATH     = globals().get("DATASET_PATH",     "../MRI_DATASET/")
SAVED_MODELS_DIR = globals().get("SAVED_MODELS_DIR", "../saved_models/")
RESULTS_DIR      = globals().get("RESULTS_DIR",      "../results/")
HF_TOKEN         = globals().get("HF_TOKEN",         os.environ.get("HF_TOKEN", ""))

RESULTS_NB_DIR   = os.path.join(RESULTS_DIR, NOTEBOOK_NAME)
os.makedirs(RESULTS_NB_DIR, exist_ok=True)
print("Results dir:", RESULTS_NB_DIR)
"""

DOWNLOAD_MODELS = """\
def _dl(filename):
    dest = os.path.join(SAVED_MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  local: {filename}"); return
    try:
        from huggingface_hub import hf_hub_download, login as hf_login
        if HF_TOKEN: hf_login(token=HF_TOKEN, add_to_git_credential=False)
        hf_hub_download(repo_id=HF_REPO_ID, filename=f"models/{filename}",
                        local_dir=SAVED_MODELS_DIR)
        print(f"  downloaded: {filename}")
    except Exception as e:
        print(f"  WARN: {filename}: {e}")

# All 6 model files needed
for f in ["cnn_model.h5", "inceptionv3_model.h5", "xception_model.h5",
          "cnn_ensemble.h5", "inceptionv3_ensemble.h5", "xception_ensemble.h5"]:
    _dl(f)
"""

LOAD_MODELS = """\
def _lm(name):
    p = os.path.join(SAVED_MODELS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing: {p}\\n"
            f"Run the corresponding training notebook first.")
    m = keras.models.load_model(p)
    m.trainable = False
    return m

print("Loading 6 models …")
models = {
    "cnn":     _lm("cnn_model.h5"),          # standalone CNN (outputs softmax)
    "inc":     _lm("inceptionv3_model.h5"),   # standalone InceptionV3 (outputs softmax)
    "xcp":     _lm("xception_model.h5"),      # standalone Xception (outputs softmax)
    "cnn_ens": _lm("cnn_ensemble.h5"),        # CNN feature extractor (outputs 256-d features)
    "inc_ens": _lm("inceptionv3_ensemble.h5"),# InceptionV3 feature extractor
    "xcp_ens": _lm("xception_ensemble.h5"),   # Xception feature extractor
}

# Determine output type per model
SOFTMAX_KEYS  = ["cnn", "inc", "xcp"]         # output shape: (N, 4)
EXTRACTOR_KEYS = ["cnn_ens", "inc_ens", "xcp_ens"]  # output shape: (N, 256)

# Input sizes
IMG_SIZES = {
    "cnn":     IMG_SIZE_CNN,
    "inc":     IMG_SIZE_PRETRAINED,
    "xcp":     IMG_SIZE_PRETRAINED,
    "cnn_ens": IMG_SIZE_CNN,
    "inc_ens": IMG_SIZE_PRETRAINED,
    "xcp_ens": IMG_SIZE_PRETRAINED,
}
print("All 6 models loaded.")
for k, m in models.items():
    out = m.output_shape
    print(f"  {k:<10}: input {m.input_shape[1:]}  output {out}")
"""

GRADCAM_FN = """\
def _last_conv(model):
    \"\"\"Name of the last layer with 4-D output (spatial conv feature map).\"\"\"
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
            if isinstance(shape, list): shape = shape[0]
            if len(shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No 4-D conv layer found.")

# Cache last conv layer names
_LAST_CONV = {k: _last_conv(m) for k, m in models.items()}
print("Last conv layers:")
for k, v in _LAST_CONV.items():
    print(f"  {k:<10}: {v}")

def compute_gradcam(model_key, img_array):
    \"\"\"
    GRAD-CAM for any of the 6 models.

    For softmax models  (cnn / inc / xcp):
        target = class score of predicted class  →  standard GRAD-CAM
    For feature extractor models (cnn_ens / inc_ens / xcp_ens):
        target = sum of all output features  →  shows which spatial region
                 drives the feature representation used by the sklearn ensemble

    img_array : np.float32, shape (1, H, W, 3), values in [0, 1]
    Returns   : (heatmap float32 (H, W) in [0, 1],  predicted_class_idx int)
    \"\"\"
    model    = models[model_key]
    conv_name = _LAST_CONV[model_key]

    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_name).output, model.output]
    )
    img_tf = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_out, model_out = grad_model(img_tf, training=False)

        if model_key in SOFTMAX_KEYS:
            pred_idx = int(tf.argmax(model_out[0]))
            target   = model_out[:, pred_idx]
        else:
            # Feature extractor: gradient of total feature activation
            pred_idx = -1          # no direct class prediction
            target   = tf.reduce_sum(model_out)

    grads  = tape.gradient(target, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
    heatmap  = np.maximum(heatmap, 0)
    mx = heatmap.max()
    return (heatmap / mx if mx > 0 else heatmap).astype(np.float32), pred_idx

def binary_iou(a, b, t=IOU_THRESHOLD):
    inter = np.logical_and(a >= t, b >= t).sum()
    union = np.logical_or( a >= t, b >= t).sum()
    return float(inter / union) if union > 0 else 1.0

def eaa_iou_6(heatmaps_dict, t=IOU_THRESHOLD):
    \"\"\"Mean pairwise IoU across all C(6,2)=15 pairs of the 6 heatmaps.\"\"\"
    keys = MODEL_KEYS
    pairs = list(itertools.combinations(keys, 2))
    ious  = {f"iou_{a}_{b}": binary_iou(heatmaps_dict[a], heatmaps_dict[b], t)
             for a, b in pairs}
    eaa   = float(np.mean(list(ious.values())))
    return eaa, ious

def consensus_heatmap(heatmaps_dict):
    \"\"\"Equal-weight average of all 6 heatmaps, normalised to [0, 1].\"\"\"
    stacked = np.stack([cv2.resize(hm, HEATMAP_SIZE)
                        for hm in heatmaps_dict.values()], axis=0)
    fused = stacked.mean(axis=0)
    mx = fused.max()
    return fused / mx if mx > 0 else fused

def overlay(img_rgb, heatmap, alpha=0.45):
    h, w = img_rgb.shape[:2]
    hm_u8  = np.uint8(255 * cv2.resize(heatmap, (w, h)))
    col    = cv2.cvtColor(cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    src    = img_rgb if img_rgb.dtype == np.uint8 else np.uint8(img_rgb * 255)
    return cv2.addWeighted(src, 1 - alpha, col, alpha, 0)
"""

COMPUTE = """\
TEST_DIR = os.path.join(DATASET_PATH, "Testing")

records      = []
sample_store = {cls: [] for cls in CLASS_NAMES}   # up to 2 samples per class for viz

for cls_name in CLASS_NAMES:
    cls_dir    = os.path.join(TEST_DIR, cls_name)
    if not os.path.isdir(cls_dir):
        print(f"  Warning: {cls_dir} not found"); continue
    img_files  = sorted([f for f in os.listdir(cls_dir)
                         if f.lower().endswith(('.jpg','.jpeg','.png'))])

    for fname in tqdm(img_files, desc=cls_name):
        img_path = os.path.join(cls_dir, fname)
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None: continue
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess for each input size
        imgs = {
            "cnn":     np.expand_dims(cv2.resize(img_rgb, IMG_SIZE_CNN).astype(np.float32) / 255.0, 0),
            "inc":     np.expand_dims(cv2.resize(img_rgb, IMG_SIZE_PRETRAINED).astype(np.float32) / 255.0, 0),
            "xcp":     np.expand_dims(cv2.resize(img_rgb, IMG_SIZE_PRETRAINED).astype(np.float32) / 255.0, 0),
            "cnn_ens": np.expand_dims(cv2.resize(img_rgb, IMG_SIZE_CNN).astype(np.float32) / 255.0, 0),
            "inc_ens": np.expand_dims(cv2.resize(img_rgb, IMG_SIZE_PRETRAINED).astype(np.float32) / 255.0, 0),
            "xcp_ens": np.expand_dims(cv2.resize(img_rgb, IMG_SIZE_PRETRAINED).astype(np.float32) / 255.0, 0),
        }

        # GRAD-CAM for all 6 models
        heatmaps = {}
        preds    = {}
        for k in MODEL_KEYS:
            hm, pred_idx = compute_gradcam(k, imgs[k])
            heatmaps[k]  = cv2.resize(hm, HEATMAP_SIZE)
            preds[k]     = pred_idx

        # EAA-IoU across all 15 pairs
        eaa, pair_ious = eaa_iou_6(heatmaps)

        # Majority vote from the 3 softmax models (standalone)
        from collections import Counter
        votes = [preds["cnn"], preds["inc"], preds["xcp"]]
        majority_pred = Counter(votes).most_common(1)[0][0]
        true_idx      = CLASS_NAMES.index(cls_name)

        rec = {
            "image_path":      img_path,
            "true_class":      cls_name,
            "true_idx":        true_idx,
            "pred_cnn":        CLASS_NAMES[preds["cnn"]],
            "pred_inc":        CLASS_NAMES[preds["inc"]],
            "pred_xcp":        CLASS_NAMES[preds["xcp"]],
            "majority_pred":   CLASS_NAMES[majority_pred],
            "majority_correct": int(majority_pred == true_idx),
            "eaa_iou":         round(eaa, 4),
        }
        rec.update({k: round(v, 4) for k, v in pair_ious.items()})
        records.append(rec)

        if len(sample_store[cls_name]) < 2:
            sample_store[cls_name].append({
                "img_rgb":      img_rgb,
                "heatmaps":     heatmaps,
                "consensus":    consensus_heatmap(heatmaps),
                "eaa_iou":      eaa,
                "majority_pred": CLASS_NAMES[majority_pred],
                "correct":      majority_pred == true_idx,
            })

df = pd.DataFrame(records)
print(f"\\nProcessed {len(df)} images.")
print(df[["true_class","pred_cnn","pred_inc","pred_xcp","eaa_iou"]].head(6))
"""

SAVE_CSV = """\
csv_path = os.path.join(RESULTS_NB_DIR, "eaa_iou_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
print("\\nEAA-IoU summary by class:")
print(df.groupby("true_class")["eaa_iou"].describe().round(3))
"""

VIZ_GRID = """\
# Chart 1: 6-model GRAD-CAM grid  (4 classes × 8 columns)
# Columns: Original | CNN | InceptionV3 | Xception | CNN+Ens | Inc+Ens | Xcp+Ens | Consensus
col_keys    = MODEL_KEYS + ["consensus"]
col_titles  = [MODEL_LABELS[k] for k in MODEL_KEYS] + ["Consensus Map"]

fig = plt.figure(figsize=(28, 14))
n_rows = len(CLASS_NAMES)
n_cols = 1 + len(col_keys)   # original + 6 heatmaps + consensus = 8

gs = fig.add_gridspec(n_rows, n_cols, hspace=0.08, wspace=0.04)

for row_idx, cls_name in enumerate(CLASS_NAMES):
    samples = sample_store[cls_name]
    if not samples: continue
    s = samples[0]
    img_disp = cv2.resize(s["img_rgb"], (224, 224))

    # Original
    ax = fig.add_subplot(gs[row_idx, 0])
    ax.imshow(img_disp); ax.axis('off')
    ax.set_ylabel(cls_name.capitalize(), fontsize=11, fontweight='bold', labelpad=6)
    if row_idx == 0:
        ax.set_title("Original MRI", fontsize=9, fontweight='bold', pad=4)

    # 6 GRAD-CAM heatmaps
    for col_offset, key in enumerate(MODEL_KEYS):
        ax = fig.add_subplot(gs[row_idx, col_offset + 1])
        hm_up = cv2.resize(s["heatmaps"][key], (224, 224))
        ax.imshow(overlay(img_disp, hm_up)); ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[col_offset], fontsize=8, fontweight='bold', pad=4)

    # Consensus map
    ax = fig.add_subplot(gs[row_idx, 7])
    con_up = cv2.resize(s["consensus"], (224, 224))
    ax.imshow(overlay(img_disp, con_up)); ax.axis('off')
    ax.set_xlabel(f"EAA-IoU: {s['eaa_iou']:.3f}", fontsize=8)
    if row_idx == 0:
        ax.set_title("Consensus Map", fontsize=8, fontweight='bold', pad=4)

fig.suptitle(
    "All-6-Model GRAD-CAM Heatmaps — Standalone DL + Ensemble Feature Extractors + Consensus\\n"
    "Brain Tumor MRI Dataset (Kaggle, 4 classes)",
    fontsize=12, fontweight='bold', y=1.01
)
p = os.path.join(RESULTS_NB_DIR, "sample_gradcam_grid.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print("Saved:", p)
"""

VIZ_DIST = """\
# Chart 2: EAA-IoU distribution (all-6-model) + per-class means
colors = {'glioma':'#e74c3c','meningioma':'#e67e22','notumor':'#2ecc71','pituitary':'#3498db'}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for cls in CLASS_NAMES:
    sub = df[df["true_class"]==cls]["eaa_iou"]
    ax.hist(sub, bins=30, alpha=0.6, label=cls.capitalize(),
            color=colors[cls], edgecolor='white')
ax.axvline(0.40, color='red',    linestyle='--', lw=1.5, label='Low/Mod threshold (0.40)')
ax.axvline(0.65, color='orange', linestyle='--', lw=1.5, label='Mod/High threshold (0.65)')
ax.set_xlabel("EAA-IoU (mean of 15 pairwise IoUs)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("EAA-IoU Distribution\\n(All 6 models, C(6,2)=15 pairs)", fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
means = df.groupby("true_class")["eaa_iou"].mean().reindex(CLASS_NAMES)
stds  = df.groupby("true_class")["eaa_iou"].std().reindex(CLASS_NAMES)
bars  = ax.bar([c.capitalize() for c in CLASS_NAMES], means, yerr=stds, capsize=5,
               color=[colors[c] for c in CLASS_NAMES], alpha=0.8, edgecolor='white')
ax.set_ylim(0, 1); ax.set_ylabel("Mean EAA-IoU", fontsize=11)
ax.set_title("Mean EAA-IoU ± Std per Class\\n(All 6 models)", fontsize=11, fontweight='bold')
ax.axhline(0.65, color='orange', linestyle='--', lw=1.2, label='High-conf threshold')
ax.axhline(0.40, color='red',    linestyle='--', lw=1.2, label='Escalation threshold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle("EAA-IoU Distribution — 6-Model Ensemble Attention Agreement", fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "eaa_iou_distribution.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
print("Saved:", p)
"""

VIZ_PAIRWISE = """\
# Chart 3: Pairwise IoU heatmap — average IoU between every model pair
pair_cols = [c for c in df.columns if c.startswith("iou_")]
pair_means = df[pair_cols].mean()

# Build 6×6 matrix
iou_matrix = np.eye(6)
for i, ki in enumerate(MODEL_KEYS):
    for j, kj in enumerate(MODEL_KEYS):
        if i >= j: continue
        col = f"iou_{ki}_{kj}"
        if col not in pair_means: col = f"iou_{kj}_{ki}"
        val = pair_means.get(col, 0)
        iou_matrix[i, j] = val
        iou_matrix[j, i] = val

short_labels = ["CNN", "Inc", "Xcp", "CNN+Ens", "Inc+Ens", "Xcp+Ens"]
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(iou_matrix, vmin=0, vmax=1, cmap='RdYlGn')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(6)); ax.set_xticklabels(short_labels, fontsize=10, rotation=30, ha='right')
ax.set_yticks(range(6)); ax.set_yticklabels(short_labels, fontsize=10)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{iou_matrix[i,j]:.2f}", ha='center', va='center',
                fontsize=9, color='black' if iou_matrix[i,j] > 0.5 else 'white')
ax.set_title("Average Pairwise GRAD-CAM IoU\\nAcross All Test Images (All 6 Models)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "pairwise_iou_heatmap.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
print("Saved:", p)
print("\\nPairwise IoU matrix:")
print(pd.DataFrame(iou_matrix, index=short_labels, columns=short_labels).round(3))
"""

VIZ_LOWHI = """\
# Chart 4: Low vs high EAA-IoU examples
df_sorted = df.sort_values("eaa_iou")
low_ex  = df_sorted.head(6)
high_ex = df_sorted.tail(6)

def _show(row, ax):
    img = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
    ax.imshow(cv2.resize(img, (224, 224))); ax.axis('off')
    ok = row['true_class'] == row['majority_pred']
    ax.set_title(f"{row['true_class'].capitalize()}\\nIoU:{row['eaa_iou']:.3f}",
                 fontsize=8, color='green' if ok else 'red')

fig, axes = plt.subplots(2, 6, figsize=(18, 7))
for i, (_, row) in enumerate(low_ex.iterrows()):  _show(row, axes[0, i])
for i, (_, row) in enumerate(high_ex.iterrows()): _show(row, axes[1, i])
axes[0, 0].set_ylabel("Low EAA-IoU\\n(Disagree)", fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel("High EAA-IoU\\n(Agree)", fontsize=10, fontweight='bold')
fig.suptitle("Low vs High EAA-IoU Examples  (green=correct, red=misclassified)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(RESULTS_NB_DIR, "eaa_iou_low_vs_high.jpg")
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.show(); plt.close()
print("Saved:", p)
"""

HF_UPLOAD = """\
def _hf_upload(files, repo_id, token, prefix=""):
    from huggingface_hub import HfApi, login as hf_login
    if not token:
        print("No HF_TOKEN — skipping upload."); return
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

files_to_upload = [
    os.path.join(RESULTS_NB_DIR, "eaa_iou_results.csv"),
    os.path.join(RESULTS_NB_DIR, "sample_gradcam_grid.jpg"),
    os.path.join(RESULTS_NB_DIR, "eaa_iou_distribution.jpg"),
    os.path.join(RESULTS_NB_DIR, "pairwise_iou_heatmap.jpg"),
    os.path.join(RESULTS_NB_DIR, "eaa_iou_low_vs_high.jpg"),
]
_hf_upload(files_to_upload, HF_REPO_ID, HF_TOKEN, prefix=f"results/{NOTEBOOK_NAME}")
print("\\nSection 10 complete.")
"""

CELLS = [
    md("# 09 — Consensus GRAD-CAM & EAA-IoU (All 6 Models)\n\n"
       "Computes GRAD-CAM heatmaps for **all 6 models** used in the final ensemble:\n\n"
       "| # | Model key | File | Output type |\n"
       "|---|-----------|------|-------------|\n"
       "| 1 | `cnn` | `cnn_model.h5` | Softmax (4 classes) |\n"
       "| 2 | `inc` | `inceptionv3_model.h5` | Softmax (4 classes) |\n"
       "| 3 | `xcp` | `xception_model.h5` | Softmax (4 classes) |\n"
       "| 4 | `cnn_ens` | `cnn_ensemble.h5` | 256-dim features |\n"
       "| 5 | `inc_ens` | `inceptionv3_ensemble.h5` | 256-dim features |\n"
       "| 6 | `xcp_ens` | `xception_ensemble.h5` | 256-dim features |\n\n"
       "For models 1–3 (softmax): GRAD-CAM targets the predicted class score.\n"
       "For models 4–6 (feature extractors): GRAD-CAM targets the sum of all 256 feature\n"
       "activations — this shows which spatial region drives the feature representation\n"
       "that the sklearn ensemble classifier acts on.\n\n"
       "**EAA-IoU** = mean of all C(6,2) = **15 pairwise IoUs** between the 6 heatmaps.\n\n"
       "Run notebooks 01–06 first."),
    md(SETUP_MD),
    code(SETUP_CODE),
    md("## Section 1 — Imports"),
    code(IMPORTS),
    md("## Section 2 — Constants"),
    code(CONSTANTS),
    md("## Section 3 — Download & Load All 6 Models\n\n"
       "Downloads from HuggingFace if not present locally."),
    code(DOWNLOAD_MODELS),
    code(LOAD_MODELS),
    md("## Section 4 — GRAD-CAM Functions\n\n"
       "- **Standalone models** (cnn / inc / xcp): standard GRAD-CAM on predicted class\n"
       "- **Feature extractor models** (cnn_ens / inc_ens / xcp_ens): GRAD-CAM on sum of 256 features\n"
       "- **EAA-IoU**: mean pairwise IoU across all 15 model pairs (C(6,2))"),
    code(GRADCAM_FN),
    md("## Section 5 — Compute GRAD-CAM for All Test Images\n\n"
       "Loops through `Testing/` directory and computes 6 heatmaps per image.\n\n"
       "> **Runtime:** ~40–60 min on Colab GPU for all 2,063 images × 6 models."),
    code(COMPUTE),
    md("## Section 6 — Save Results CSV"),
    code(SAVE_CSV),
    md("## Section 7 — Visualisation: All-6-Model GRAD-CAM Grid"),
    code(VIZ_GRID),
    md("## Section 8 — Visualisation: EAA-IoU Distribution by Class"),
    code(VIZ_DIST),
    md("## Section 9 — Visualisation: Pairwise IoU Matrix & Low/High Examples"),
    code(VIZ_PAIRWISE),
    code(VIZ_LOWHI),
    md("## Section 10 — Upload to HuggingFace"),
    code(HF_UPLOAD),
]

path = os.path.join(OUT_DIR, "consensus_gradcam.ipynb")
with open(path, "w") as f:
    json.dump(mk_nb(CELLS), f, indent=1)
print(f"Saved: {path}")
