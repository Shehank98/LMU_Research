# Code for the Final Project Report

**A Comparative Analysis of Ensemble Learning, Fine-Tuned Models and CNNs for Multi-Class Disease Prediction in
Medical Imaging: An Integrated Framework with Systematic Optimization**
Pattiyage Shehan Kavishka (E187041), CS6P05ES Computing Project, London Metropolitan University / ESOFT Metro Campus

This folder contains only the code behind the Final Project Report. Every number in the report comes from the
result files in `2_Results/`, which were written by the training notebook in `1_Training_and_Evaluation/`.

| Folder | What it contains | Report |
|---|---|---|
| `1_Training_and_Evaluation/` | Training and evaluation of all 22 configurations, the six-predictor ensemble and EAA-IoU (Google Colab, T4 GPU) | Chapter 3, Chapter 4 |
| `2_Results/` | Result files logged by the notebook: metrics, confusion matrices, training histories, per-image predictions, EAA-IoU | All result tables |
| `3_Analysis_and_Figures/` | Scripts that turn the result files into the report's figures and derived numbers, and compute Grad-CAM | Chapters 2 to 5 |
| `4_NeuroScan_Web_App/` | The NeuroScan decision-support web application (React front end, FastAPI back end, Docker / Railway) | Sections 3.8 and 5.4 |

## 1_Training_and_Evaluation

| File | What it does |
|---|---|
| `Brain_Tumor_Training_and_Evaluation.ipynb` | All-in-one notebook: data loading, the custom CNN, InceptionV3 and Xception (256 px), the three 128-px feature extractors, RF / DT / SVM / LR on raw pixels and on deep features, the six-predictor majority and confidence-weighted ensemble, Grad-CAM and EAA-IoU for every test image, and the upload of models and results to Hugging Face |
| `individual_models/01_CNN.ipynb` ... `10_Upload_to_HuggingFace.ipynb` | The same code split into one notebook per model (see `individual_models/README.md` for the run order) |
| `training_log.txt` | Log of the Colab run, including the first failed attempt (duplicate layer name), its fix and the two resumed CNN sessions |

Settings used everywhere: 256 × 256 inputs for the standalone networks, 128 × 128 for the feature extractors,
pixels scaled to [0, 1], batch size 32, Adam, seed 42, official Testing folder (1,600 images) used once.

## 2_Results

| File | Used for |
|---|---|
| `metrics/results_master.csv` | Accuracy, macro precision / recall / F1, per-class F1 and training time of every configuration (Tables 9 to 13) |
| `metrics/*_classification_report.json` | Per-class precision and recall (Table 16) |
| `confusion/*_confusion_matrix.csv` | Confusion matrices (Table 17, Figures 10 to 12) |
| `history/*_history.csv` | Training curves (Table 8, Figures 7 and 8) |
| `metrics/final_per_image_predictions.csv`, `metrics/final_probs.npz` | Per-image votes and probabilities of the six predictors (Tables 14 and 15) |
| `metrics/eaa_iou_per_image.csv`, `metrics/eaa_iou_summary.json` | EAA-IoU of every test image (Tables 20 to 24, Figure 18) |
| `metrics/dataset_counts.json` | Images per class (Table 3, Figure 2) |

## 3_Analysis_and_Figures

Run in this order (needs the dataset and the trained models, see "How to run"):

| Script | Produces | Report |
|---|---|---|
| `1_test_set_probabilities.py` | Re-runs the saved networks on the test set, checks they match the logged predictions (100% agreement), saves their probabilities | Section 4.6 |
| `2_results_figures.py` | Dataset, sample, training, confusion, accuracy, F1, ROC/PR and EAA-IoU figures; every derived number in `outputs/report_numbers.json` (vote agreement, ensemble build-up, EAA-IoU statistics, thresholds, bands) | Figures 2, 3, 7 to 14, 18; Tables 8, 14, 15, 18, 20 to 24 |
| `3_gradcam_figures.py` | Grad-CAM of the three networks for correct and misclassified examples, and the EAA-IoU worked example (checked against the logged values) | Figures 15 to 17; Table 19 |
| `4_diagrams.py` | Conceptual framework, CNN architecture, six-predictor ensemble | Figures 1, 4, 5 |
| `settings.py`, `gradcam.py` | Shared paths and image loader; Grad-CAM and IoU (same method as the notebook) | Section 3.7 |

## 4_NeuroScan_Web_App

| Path | What it does |
|---|---|
| `backend/main.py` | FastAPI server: `/health`, `/api/models/status`, `/api/predict`, `/api/results`, `/api/patients`; also serves the built React site |
| `backend/xai_engine.py` | Six-predictor confidence-weighted prediction, Grad-CAM, consensus map, EAA-IoU, alert below 0.10 |
| `backend/model_loader.py` | Downloads and loads the nine model files from Hugging Face in the background |
| `backend/analysis.py` | One full analysis of an image (used by the Diagnose page and by patient scans) |
| `backend/patients.py` | Patient records API: list, profile with scan history, new patient, new scan (analysed and saved), radiologist review |
| `backend/demo_patients/` | Five fictitious demonstration patients with Sinhala names; their scans are public test images analysed by the trained models (`seed_demo_patients.py`) |
| `frontend/src/pages/Patients.jsx`, `PatientDetail.jsx` | Patients list and patient record (profile, scan history, saved analyses, review form, new-scan upload) |
| `frontend/src/pages/Diagnose.jsx` | Upload page: prediction, confidence, EAA-IoU, alert, heatmaps, per-model votes |
| `frontend/src/pages/Results.jsx` | Results page: all configurations, EAA-IoU summary, thresholds, dataset |
| `Dockerfile`, `railway.toml` | Two-stage build (React with Node, then Python) and Railway deployment |

## How to run

```bash
# 1. data and trained models
#    dataset: Kaggle "Brain Tumor MRI Dataset" (Nickparvar) with Training/ and Testing/ folders
huggingface-cli download shehank98/brain-tumor-mri-models --include "models/*" --local-dir hf
export MRI_DATA=/path/to/MRI_DATASET
export MODEL_DIR=$PWD/hf/models

# 2. figures and numbers of the report
cd 3_Analysis_and_Figures
pip install -r requirements.txt
python 1_test_set_probabilities.py
python 2_results_figures.py
python 3_gradcam_figures.py
python 4_diagrams.py

# 3. NeuroScan on your own computer (models are downloaded from Hugging Face automatically)
cd ../4_NeuroScan_Web_App
docker build -t neuroscan . && docker run -p 8000:8000 neuroscan      # then open http://localhost:8000
```

To retrain everything from scratch, open `1_Training_and_Evaluation/Brain_Tumor_Training_and_Evaluation.ipynb`
in Google Colab with a GPU and run all cells (the dataset path is set in the first cells).
