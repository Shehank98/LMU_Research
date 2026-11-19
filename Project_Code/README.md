# Project code: brain tumour MRI classification with explainable AI

This folder holds the code behind every result in the report, organised by step. Start with the folder that
matches the report section you want to check. Each script saves its results as files (CSV / JSON / PNG), and the
report quotes the numbers from those files.

| Folder | What it covers | Report |
|---|---|---|
| `01_Baseline_Models/` | First configuration: training and evaluation of all 22 models | Chapter 4 |
| `02_Improved_Models/` | Improved configuration: head cropping, augmentation, fine-tuning, ensemble | Sections 3.6 to 3.7, Chapter 6 |
| `03_Model_Verification/` | Re-runs the saved models to confirm the logged results | Section 4.9 |
| `04_Data_Quality_Checks/` | Duplicate test images and the background (shortcut) cue | Sections 3.5, 4.10, 5.7 |
| `05_Explainable_AI/` | Grad-CAM, Grad-CAM++, HiResCAM, occlusion, LIME, SHAP and their evaluation | Section 3.10, Chapter 5, Section 6.5 |
| `06_Statistics_and_Figures/` | Confidence intervals, McNemar tests and the figures in the report | Chapters 3 to 6 |
| `07_Web_App/` | Back end of the web demonstrator (prediction, heatmaps, agreement alert) | Sections 5.8 and 5.11 |
| `shared/` | Settings and helper functions used by folders 03 to 06 | |

## 01_Baseline_Models
| File | What it does | Results |
|---|---|---|
| `Baseline_Training_and_Evaluation.ipynb` | Loads the data, trains the custom CNN, InceptionV3 and Xception, the classical classifiers on raw pixels and on deep features, and the six-model ensemble; computes EAA-IoU. Run on Google Colab (T4 GPU). | `metrics/`, `confusion/`, `history/` |
| `baseline_training_log.txt` | Log of the Colab run, including the first failed attempt (duplicate layer name) and its fix. | |

## 02_Improved_Models
Run `./run_full_pipeline.sh` (finished steps are skipped). Results go to `runs/v2/`.

| File | What it does |
|---|---|
| `settings.py` | Image size, seed, epochs, data and output folders |
| `data_preparation.py` | Loads the images, crops each one to the head, makes the stratified validation split (Listing 1) |
| `model_architectures.py` | Improved CNN and the pretrained networks with a new head (Listings 3 and 4) |
| `train_deep_models.py` | Trains one deep model with augmentation and two-stage fine-tuning, saves its test probabilities and features |
| `train_classical_models.py` | Random Forest, Decision Tree and SVM on the deep features, combined by soft voting (Listing 5) |
| `build_ensemble.py` | Six-predictor majority and confidence-weighted vote (Listing 6) |
| `evaluation_metrics.py` | Accuracy, classification report and confusion matrix for every model |

## 03_Model_Verification
| File | What it does | Results |
|---|---|---|
| `1_verify_saved_models.py` | Reloads the saved models and checks that they give the logged predictions | `paper_assets/xai_results/verification.json` |
| `2_rerun_final_ensemble.py` | Rebuilds the six-model ensemble from the saved models only | `final_ensemble_rerun.json` |

## 04_Data_Quality_Checks
| File | What it does | Results |
|---|---|---|
| `1_find_duplicate_test_images.py` | Finds test images that also appear in the training set (336 found) | `metrics/test_duplicates.csv` |
| `2_background_shortcut_investigation.py` | Image size by class, accuracy with only the head or only the background visible, class from image statistics | `shortcut_investigation.json` |
| `3_results_on_duplicate_free_subset.py` | Accuracy on the 1,264 clean images, bootstrap intervals, baseline vs improved McNemar tests | `metrics/clean_subset_results.json` |

## 05_Explainable_AI
| File | What it does | Report |
|---|---|---|
| `1_xai_examples_all_methods.py` | All six methods on correct and misclassified example images | Sections 5.1 to 5.3 |
| `2_xai_faithfulness_and_consistency.py` | Deletion and insertion AUC, agreement between models, attention outside the head | Sections 5.4, 5.6 |
| `3_xai_sanity_check_and_summary.py` | Cascading weight-randomisation sanity check (split from file 2 because of memory) | Section 5.5 |
| `4_shap_random_forest.py` | SHAP TreeExplainer on the Random Forest that uses CNN features | Section 5.10 |
| `5_eaa_iou_all_test_images.py` | EAA-IoU for all 1,600 test images and the alert thresholds | Sections 5.8, 6.5 |
| `6_eaa_iou_worked_example.py` | Worked example on three images and the table of errors per agreement band | Figure 8, Tables 8 and 21 |

The methods themselves (Grad-CAM, Grad-CAM++, HiResCAM, occlusion, LIME, deletion/insertion) are in
`shared/xai_methods.py`.

## 06_Statistics_and_Figures
| File | What it does |
|---|---|
| `1_statistics_and_result_figures.py` | Wilson and bootstrap intervals, McNemar tests with Holm correction, ROC/PR curves, results figures |
| `2_xai_example_figures.py` | Heatmap figures for the example images |
| `3_xai_evaluation_figures.py` | Deletion/insertion, sanity check and agreement figures |
| `4_shortcut_figure.py` | Image size and background-only accuracy figure |
| `5_improved_model_figures.py` | Training curves, confusion matrices and baseline vs improved accuracy |
| `6_architecture_diagrams.py` | Framework, pipeline and architecture diagrams |

## 07_Web_App
`api_server.py` (FastAPI) receives an MRI image and returns the prediction of all six models, the Grad-CAM heatmaps,
the consensus map and the EAA-IoU agreement score. `xai_engine.py` holds the prediction and explanation logic
(alert below 0.10, strong agreement from 0.20), and `model_loader.py` loads the saved models.

## How to run
```bash
pip install tensorflow scikit-learn opencv-python pandas matplotlib scipy shap lime scikit-image joblib
export MRI_DATA=/path/to/MRI_DATASET          # Training/ and Testing/ folders
cd 02_Improved_Models && ./run_full_pipeline.sh
cd ../05_Explainable_AI && MODEL_SET=improved python 6_eaa_iou_worked_example.py
```
Scripts in folders 03 to 06 analyse the baseline models by default; set `MODEL_SET=improved` for the improved ones.

The same code is used by the report build (`src/` and `paper_assets/scripts/` in the repository); only the file
names and import lines differ here.
