# LMU Research — Brain Tumor MRI Classification

**Student:** Pattiyage Shehan Kavishka (E187041)  
**Institution:** ESOFT Metro Campus  
**Title:** A Comparative Analysis of Ensemble Learning, Fine-Tuned Models and CNNs for Multi-Class Disease Prediction in Medical Imaging: An Integrated Framework with Systematic Optimization.

---

## Project Summary

This research builds an integrated multi-class brain tumor classification system for MRI images. It systematically compares traditional machine learning, CNN architectures, transfer-learning models, and an ensemble that combines all of them. The final ensemble achieves **98.20% accuracy** on the test set.

**4 Classes:** `glioma` | `meningioma` | `notumor` | `pituitary`

---

## Dataset

**Source:** Kaggle Brain Tumor MRI Dataset  
**Path (local):** `../MRI_DATASET/`  
**Structure:**
```
MRI_DATASET/
  Training/   — 6,984 images (4 classes)
  Testing/    — 2,063 images (4 classes)
```
Image format: `.jpg`, resized to 256×256 or 128×128 depending on model.

> The dataset is NOT committed to this repo (too large). Download from Kaggle and place at the path above before running any notebook.

---

## Repository Structure

```
LMU_Research/
├── CNN/                          # Custom CNN (4-layer)
│   └── cnn_model.ipynb
├── ALAEXNET/                     # AlexNet variant
│   └── Alexnet.ipynb
├── Xception_model/               # Xception transfer learning
│   └── Xception_model.ipynb
├── InceptionV3_model/            # InceptionV3 transfer learning
│   └── InceptionV3_model.ipynb
├── SVM_RF_DT_model/              # Raw pixel + classical ML
│   └── SVM_RF_DT_Model.ipynb
├── CNN_WITH_SVM_RF_DT/           # CNN features + classical ML
│   └── CNN_WITH_SVM_RF_DT.ipynb
├── Xception + Ensemble/          # Xception features + RF/DT/SVM ensemble
│   └── Xception +Ensemble.ipynb
├── InceptionV3 + Ensemble/       # InceptionV3 features + RF/DT/SVM ensemble
│   └── InceptionV3+Ensemble.ipynb
├── Final_model_Research/         # Full 6-model majority-vote ensemble
│   └── Final_model.ipynb
└── MODELS/                       # Saved model files (partial — heavy files excluded)
    ├── cnn_ensemble.h5
    └── cnn_ensemble_model.pkl
```

---

## Models & Results

### 1. Traditional ML on Raw Pixels (`SVM_RF_DT_model`)
Pixels flattened to 65,536 features, normalised 0–1. No PCA applied.

| Model              | Test Accuracy |
|--------------------|---------------|
| Logistic Regression | 84%          |
| SVM                | 88%           |
| Decision Tree      | 80%           |
| Random Forest      | 80%           |
| Ensemble (DT+RF+SVM, hard vote) | 83% |

### 2. Custom CNN (`CNN/`)
4× Conv2D→MaxPool blocks, Dense(64), Softmax(4). Input 256×256×3. Trained 20 epochs.

| Metric | Value |
|--------|-------|
| Test Accuracy | **91.18%** |
| Training Accuracy | ~99% (overfitting visible) |

### 3. Xception Transfer Learning (`Xception_model/`)
Frozen ImageNet weights + Flatten + Dense(4, softmax). Input 256×256×3. Trained 5 epochs.

| Test Accuracy |
|---------------|
| **86.91%** |

### 4. InceptionV3 Transfer Learning (`InceptionV3_model/`)
Same approach as Xception. Trained 5 epochs.

| Test Accuracy |
|---------------|
| **84.63%** |

### 5. Xception + Classical ML Ensemble (`Xception + Ensemble/`)
Xception fine-tuned (10 epochs, 128×128), Dense(256) layer used as feature extractor → RF/DT/SVM.

| Classifier        | Accuracy |
|-------------------|----------|
| Random Forest     | 73.68%   |
| Decision Tree     | 68.20%   |
| SVM               | 56.71%   |
| Soft Vote Ensemble | 70.29%  |

### 6. InceptionV3 + Classical ML Ensemble (`InceptionV3 + Ensemble/`)
Same pipeline as Xception+Ensemble but with InceptionV3 backbone.

### 7. Final Integrated Ensemble (`Final_model_Research/`)
Majority vote across **6 predictors**:
- CNN model direct prediction
- InceptionV3 direct prediction
- Xception direct prediction
- CNN features → traditional ensemble
- InceptionV3 features → traditional ensemble
- Xception features → traditional ensemble

| Metric    | Value  |
|-----------|--------|
| **Accuracy** | **98.20%** |
| Precision | 0.98   |
| Recall    | 0.98   |
| F1-Score  | 0.98   |

Confusion matrix (test set, 6,985 samples — augmented evaluation):
```
Glioma    [1688   34    6    5]
Meningioma[  42 1636    8   13]
No Tumor  [   5    5 1837    0]
Pituitary [   4    4    0 1698]
```

---

## Saved Models (MODELS/)

The following models must exist in `MODELS/` for `Final_model.ipynb` to run:

| File | Description |
|------|-------------|
| `MRI_model.sav` | Pickled CNN classification model |
| `InceptionV3.h5` | InceptionV3 Keras model |
| `Xception.h5` | Xception Keras model |
| `cnn_ensemble.h5` | CNN feature extractor (committed) |
| `inc_ensemble.h5` | InceptionV3 feature extractor |
| `xcp_ensemble.h5` | Xception feature extractor |
| `cnn_ensemble_model.pkl` | CNN → classical ensemble (committed) |
| `inc_ensemble_model.pkl` | InceptionV3 → classical ensemble |
| `xcp_ensemble_model.pkl` | Xception → classical ensemble |
| `ensemble_model.pkl` | Final voting ensemble |

> `.h5` files for InceptionV3 and Xception are 80–200 MB and exceed GitHub's 100 MB limit.  
> See **"Uploading Heavy Files"** section below for how to handle this.

---

## Environment Setup

```bash
# Python 3.8+
pip install tensorflow==2.x keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow joblib tqdm streamlit
```

**GPU strongly recommended** — each training run is 5–20 epochs and takes 300–800 s/epoch on CPU.

Run notebooks in this order if reproducing from scratch:
1. `SVM_RF_DT_model/SVM_RF_DT_Model.ipynb`
2. `CNN/cnn_model.ipynb`
3. `Xception_model/Xception_model.ipynb`
4. `InceptionV3_model/InceptionV3_model.ipynb`
5. `Xception + Ensemble/Xception +Ensemble.ipynb`
6. `InceptionV3 + Ensemble/InceptionV3+Ensemble.ipynb`
7. `Final_model_Research/Final_model.ipynb`

---

## Uploading Heavy Files to GitHub

GitHub enforces a **100 MB hard limit** per file. The `.h5` model files and the MRI dataset are far too large. Use one of these approaches:

### Option A — Git LFS (Recommended for models)
```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.h5"
git lfs track "*.pkl"
git lfs track "*.sav"
git add .gitattributes
git add MODELS/
git commit -m "add models via Git LFS"
git push
```
GitHub gives 1 GB LFS storage free. Files >2 GB need a paid plan.

### Option B — Store on Google Drive / Kaggle
Upload `.h5` and `.pkl` files to Google Drive or a Kaggle dataset, then add a download cell at the top of each notebook:
```python
# Example: gdown for Google Drive
!pip install gdown
import gdown
gdown.download("https://drive.google.com/uc?id=FILE_ID", "MODELS/InceptionV3.h5")
```

### Option C — HuggingFace Hub (Best for ML models)
```bash
pip install huggingface_hub
huggingface-cli login
# Upload model files to a HuggingFace repository
```

### Option D — .gitignore heavy files, document download steps
Add to `.gitignore`:
```
*.h5
MRI_DATASET/
```
Document exact download/training steps so anyone can reproduce from scratch.

---

## Known Issues & Notes

- `SVM_RF_DT_model`: The "Random Forest" cell accidentally instantiates `DecisionTreeClassifier` instead of `RandomForestClassifier` — results for RF and DT are identical (both 80%).
- `CNN/cnn_model.ipynb`: Early cells have a `TypeError` (pth used as string before fix) — later cells use `tf.keras.preprocessing.image_dataset_from_directory` correctly.
- All notebooks were developed on Windows (paths use `\\`). On Linux/Mac use `../MRI_DATASET/Training/`.
- Xception/InceptionV3 standalone models show high validation loss despite reasonable accuracy — consider fine-tuning more layers or adding dropout.
