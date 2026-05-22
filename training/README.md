# Training Scripts

Clean Python reimplementation of all model training code.
Replaces the original Jupyter notebooks with runnable, reproducible Python scripts.

---

## Setup

### 1. Install dependencies
```bash
pip install tensorflow scikit-learn opencv-python-headless numpy matplotlib seaborn joblib plotly pillow
```

### 2. Place the dataset
Download from Kaggle:
```bash
curl -L -o ~/Downloads/brain-tumor-mri-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset
```
Extract so the structure is:
```
LMU_Research/
└── MRI_DATASET/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        └── ...
```

---

## Run order

Run each script from inside the `training/` folder:
```bash
cd training
```

| Script | What it produces | Time (CPU) |
|--------|-----------------|-----------|
| `python train_01_cnn.py` | `cnn_standalone.h5`, `cnn_ensemble.h5`, `cnn_ensemble_model.pkl` | ~2–4 h |
| `python train_02_inception.py` | `InceptionV3.h5`, `inc_ensemble.h5`, `inc_ensemble_model.pkl` | ~3–5 h |
| `python train_03_xception.py` | `Xception.h5`, `xcp_ensemble.h5`, `xcp_ensemble_model.pkl` | ~3–5 h |
| `python train_04_classical_raw.py` | *(no files — comparison only)* | ~1–2 h |
| `python evaluate_ensemble.py` | Prints final ensemble accuracy | ~5 min |

Scripts 01–03 can run in parallel on separate machines or GPUs.
Script 04 is optional (research comparison baseline, not required for the webapp).

---

## Expected results

| Model | Expected accuracy |
|-------|------------------|
| CNN standalone | ~91% |
| InceptionV3 standalone | ~85% |
| Xception standalone | ~87% |
| CNN + classical ensemble | ~84% |
| InceptionV3 + classical ensemble | ~71% |
| Xception + classical ensemble | ~70% |
| **6-model ensemble** | **~98%** |

---

## File structure

```
training/
├── config.py                  All hyperparameters and paths
├── data.py                    Dataset loaders (tf.data + numpy)
├── evaluate.py                Shared metrics, plots, feature extraction
├── models/
│   ├── cnn.py                 Custom CNN builder
│   ├── transfer.py            InceptionV3 + Xception builders
│   └── classical.py          RF + DT + SVM VotingClassifier
├── train_01_cnn.py
├── train_02_inception.py
├── train_03_xception.py
├── train_04_classical_raw.py
└── evaluate_ensemble.py
```

---

## After training

All 9 model files land in `../MODELS/`. Start the webapp:
```bash
cd ../webapp
streamlit run app.py
```

---

## Key design decisions

- **Single normalisation**: all images divided by 255.0 exactly once in `data.py`
- **Explicit layer names**: `Dense(..., name='dense')`, `name='dense_2'`, `name='dense_3'` guarantee
  webapp compatibility regardless of Keras session state
- **Consistent class order**: alphabetical sort → glioma=0, meningioma=1, notumor=2, pituitary=3
- **float32 throughout**: halves memory vs float64 (avoids MemoryError on 6,984 training images)
