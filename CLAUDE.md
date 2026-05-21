# LMU Research — Brain Tumor MRI Classification

**Student:** Pattiyage Shehan Kavishka (E187041)  
**Institution:** ESOFT Metro Campus  
**Programme:** BTEC Higher National Diploma in Computing — HND in Data Analytics  
**Unit:** Unit 16: Computing Research Project (Pearson Set)  
**Supervisor:** Mr. Dileepa Lakshan  
**Submission Date:** 2024-10-18  
**Title:** A Comparative Analysis of Ensemble Learning, Fine-Tuned Models and CNNs for Multi-Class Disease Prediction in Medical Imaging: An Integrated Framework with Systematic Optimization.

---

## Abstract

This study examines how well fine-tuned models perform in feature extraction when compared to ordinary Convolutional Neural Networks (CNNs) and how their integration with ensemble techniques affects the precision and dependability of identifying brain tumors from MRI images. The purpose is to assess the performance of machine learning classifiers such as Support Vector Machines (SVM), Random Forest (RF), and Decision Tree (DT), as well as more sophisticated models like Xception and Inception V3. A mixed-methods approach was used, combining qualitative views from healthcare professionals with quantitative indicators like accuracy, precision, recall, and F1 score. The final ensemble model, composed of six saved models, outperformed the CNN's highest individual model accuracy of 94% with an accuracy of **98.20%**.

**Keywords:** CNN, Ensemble Learning, Inception V3, Xception, Medical Imaging, Brain Tumor Detection

---

## Research Objectives

**Main Research Question:** How well do fine-tuned models perform for feature extraction compared to standard CNNs, and what effect does their integration with ensemble approaches have on the accuracy and dependability of brain tumor identification from MRI images?

**Hypothesis:** Fine-tuned models will outperform conventional CNNs in identifying distinguishing characteristics from MRI images. Furthermore, using an ensemble architecture to integrate outputs of many classifiers will result in higher classification accuracy than individual classifiers alone.

**Primary Objectives:**
1. Investigate how well conventional CNNs perform feature extraction for brain tumor identification from MRI images
2. Assess how well deep classifiers (RF/DT/SVM) perform when combined with CNN-extracted features
3. Investigate fine-tuned models (Xception, Inception V3) for feature extraction in MRI-based brain tumor identification
4. Evaluate different ML classifiers using features retrieved from optimised models
5. Investigate how well ensemble approaches work at combining classifier results to increase detection accuracy
6. Compare accuracy, computational effectiveness, and robustness of CNN-based vs fine-tuned model-based feature extraction

---

## Dataset

**Source:** Kaggle Brain Tumor MRI Dataset  
**Local path:** `../MRI_DATASET/`  
**Total images:** 9,047 MRI scans in JPG format  
**Classes:** glioma | meningioma | notumor | pituitary

| Class | Train Images | Test Images |
|-------|-------------|-------------|
| Glioma | 1,733 | 504 |
| Meningioma | 1,699 | 510 |
| No Tumor | 1,847 | 545 |
| Pituitary | 1,705 | 504 |
| **Total** | **6,984** | **2,063** |

> The dataset is NOT committed to this repo (too large). Download from Kaggle and place at `../MRI_DATASET/` before running any notebook.

---

## Repository Structure

```
LMU_Research/
├── CNN/                          # Custom CNN (4-layer)
│   └── cnn_model.ipynb
├── ALAEXNET/                     # AlexNet variant
│   └── Alexnet.ipynb
├── Xception_model/               # Xception transfer learning (standalone)
│   └── Xception_model.ipynb
├── InceptionV3_model/            # InceptionV3 transfer learning (standalone)
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
    ├── cnn_ensemble.h5           (2.1 MB — committed)
    └── cnn_ensemble_model.pkl    (15 MB — committed)
```

---

## Models & Results

### Layer / Feature Extraction Reference

| Model | Input Shape | Layer Name | Features Extracted |
|-------|------------|-----------|-------------------|
| Inception V3 (standalone) | 256×256 | — | — |
| Inception V3 + Ensemble | 128×128 | `dense_2` | 256 |
| Xception (standalone) | 256×256 | — | — |
| Xception + Ensemble | 128×128 | `dense_3` | 256 |
| CNN (standalone) | 256×256 | — | — |
| CNN + Ensemble | 128×128 | `dense` | 256 |
| DT / RF / SVM (raw pixels) | 256×256 flattened | — | 65,536 |

---

### 1. Traditional ML on Raw Pixels (`SVM_RF_DT_model/`)
Pixels flattened to 65,536 features, normalised 0–1. No PCA applied (explicitly skipped to avoid feature loss).

| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | 84% |
| SVM | 88.26% |
| Decision Tree | 80.02% |
| Random Forest | 79.52% |
| Ensemble (DT + RF + SVM, hard vote) | 83% |

> Note: The "Random Forest" cell in the notebook accidentally instantiates `DecisionTreeClassifier` — RF and DT results are therefore identical (both ~80%).

---

### 2. Custom CNN (`CNN/`)
Architecture: 4× (Conv2D → MaxPool2D), Dense(64, relu), Dense(4, softmax). Input 256×256×3. Trained 20 epochs, Adam, SparseCategoricalCrossentropy.

| Metric | Value |
|--------|-------|
| Test Accuracy | **91.18% (reported as 94% in research report)** |
| Training Accuracy | ~99% (overfitting visible after epoch 10) |

---

### 3. Xception Transfer Learning (`Xception_model/`)
Frozen ImageNet weights, Flatten → Dense(4, softmax). Input 256×256×3. 5 epochs.

| Test Accuracy |
|--------------|
| **86.91%** |

---

### 4. InceptionV3 Transfer Learning (`InceptionV3_model/`)
Same architecture as Xception standalone. 5 epochs.

| Test Accuracy |
|--------------|
| **84.63%** |

---

### 5. CNN + Classical ML Ensemble (`CNN_WITH_SVM_RF_DT/`)
CNN features (Dense layer, 256 features, 128×128 input) fed into RF/DT/SVM.

| Classifier | Accuracy |
|------------|---------|
| Random Forest | 85.45% |
| Decision Tree | 80.99% |
| SVM | 66.60% |
| Ensemble (RF + DT + SVM) | **83.71%** |

Confusion matrix — CNN + Ensemble:

| | Glioma | Meningioma | No Tumor | Pituitary |
|--|--------|-----------|---------|----------|
| Glioma | 284 | 136 | 29 | 55 |
| Meningioma | 40 | 442 | 16 | 12 |
| No Tumor | 0 | 4 | 534 | 7 |
| Pituitary | 14 | 17 | 6 | 467 |

Classification report — CNN + Ensemble (accuracy 84%):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Glioma | 0.84 | 0.56 | 0.67 |
| Meningioma | 0.74 | 0.87 | 0.80 |
| No Tumor | 0.91 | 0.98 | 0.95 |
| Pituitary | 0.86 | 0.93 | 0.89 |
| **Overall** | **0.84** | **0.84** | **0.83** | |

---

### 6. Xception + Classical ML Ensemble (`Xception + Ensemble/`)
Xception fine-tuned (10 epochs, 128×128), Dense(256) as feature extractor → RF/DT/SVM.

| Classifier | Accuracy |
|------------|---------|
| Random Forest | 73.68% |
| Decision Tree | 68.20% |
| SVM | 56.71% |
| Soft Vote Ensemble | **70.28%** |

Confusion matrix — Xception + Ensemble:

| | Glioma | Meningioma | No Tumor | Pituitary |
|--|--------|-----------|---------|----------|
| Glioma | 226 | 127 | 42 | 109 |
| Meningioma | 69 | 368 | 25 | 48 |
| No Tumor | 16 | 19 | 492 | 18 |
| Pituitary | 58 | 56 | 26 | 364 |

Classification report — Xception + Ensemble (accuracy 70%):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Glioma | 0.75 | 0.53 | 0.62 |
| Meningioma | 0.70 | 0.81 | 0.75 |
| No Tumor | 0.92 | 0.99 | 0.95 |
| Pituitary | 0.81 | 0.87 | 0.84 |
| **Overall** | **0.80** | **0.80** | **0.79** | |

---

### 7. InceptionV3 + Classical ML Ensemble (`InceptionV3 + Ensemble/`)
Same pipeline as Xception+Ensemble, InceptionV3 backbone.

| Classifier | Accuracy |
|------------|---------|
| Random Forest | 74.64% |
| Decision Tree | 68.05% |
| SVM | 57.34% |
| Soft Vote Ensemble | **71.20%** |

Confusion matrix — InceptionV3 + Ensemble:

| | Glioma | Meningioma | No Tumor | Pituitary |
|--|--------|-----------|---------|----------|
| Glioma | 225 | 148 | 46 | 85 |
| Meningioma | 48 | 376 | 41 | 45 |
| No Tumor | 15 | 40 | 468 | 22 |
| Pituitary | 43 | 32 | 29 | 400 |

Classification report — InceptionV3 + Ensemble (accuracy 71%):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Glioma | 0.68 | 0.45 | 0.54 |
| Meningioma | 0.63 | 0.74 | 0.68 |
| No Tumor | 0.80 | 0.86 | 0.83 |
| Pituitary | 0.72 | 0.79 | 0.76 |
| **Overall** | **0.71** | **0.71** | **0.70** | |

---

### 8. Final Integrated Ensemble (`Final_model_Research/`)
Majority vote across **6 predictors**:
1. CNN direct prediction
2. InceptionV3 direct prediction
3. Xception direct prediction
4. CNN features → traditional ensemble (pkl)
5. InceptionV3 features → traditional ensemble (pkl)
6. Xception features → traditional ensemble (pkl)

Final confusion matrix (6,985 samples):

| | Glioma | Meningioma | No Tumor | Pituitary |
|--|--------|-----------|---------|----------|
| Glioma | 1688 | 34 | 6 | 5 |
| Meningioma | 42 | 1636 | 8 | 13 |
| No Tumor | 5 | 5 | 1837 | 0 |
| Pituitary | 4 | 4 | 0 | 1698 |

Final classification report:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Glioma | 0.97 | 0.97 | 0.97 | 1,733 |
| Meningioma | 0.97 | 0.96 | 0.97 | 1,699 |
| No Tumor | 0.99 | 0.99 | 0.99 | 1,847 |
| Pituitary | 0.99 | 1.00 | 0.99 | 1,706 |
| **Overall** | **0.98** | **0.98** | **0.98** | **6,985** |

**Final Ensemble Accuracy: 98.20%**

---

### Complete Accuracy Summary

| Model | Accuracy |
|-------|---------|
| CNN (standalone) | 94% |
| Xception (standalone) | 86.91% |
| InceptionV3 (standalone) | 84.63% |
| Traditional ML Ensemble (DT+RF+SVM on raw pixels) | 83% |
| CNN + Ensemble Models | 83.71% |
| InceptionV3 + Ensemble Models | 71.20% |
| Xception + Ensemble Models | 70.28% |
| **All 6 Models Combined (Final Ensemble)** | **98.20%** |

Traditional ML individual accuracy by feature source:

| Feature Source | SVM | Decision Tree | Random Forest |
|---------------|-----|--------------|--------------|
| Xception | 56.71% | 68.20% | 73.67% |
| InceptionV3 | 57.34% | 68.05% | 74.64% |
| CNN | 66.60% | 80.99% | 85.45% |
| Raw pixels (no deep features) | 88.26% | 80.02% | 79.52% |

---

## Saved Models (MODELS/)

The following files must exist in `MODELS/` for `Final_model.ipynb` to run:

| File | Description | In Repo? |
|------|-------------|---------|
| `MRI_model.sav` | Pickled CNN classification model | No — too large |
| `InceptionV3.h5` | InceptionV3 Keras model | No — too large |
| `Xception.h5` | Xception Keras model | No — too large |
| `cnn_ensemble.h5` | CNN feature extractor | Yes (2.1 MB) |
| `inc_ensemble.h5` | InceptionV3 feature extractor | No — too large |
| `xcp_ensemble.h5` | Xception feature extractor | No — too large |
| `cnn_ensemble_model.pkl` | CNN → classical ensemble | Yes (15 MB) |
| `inc_ensemble_model.pkl` | InceptionV3 → classical ensemble | No — too large |
| `xcp_ensemble_model.pkl` | Xception → classical ensemble | No — too large |
| `ensemble_model.pkl` | Final voting ensemble | No — too large |

---

## Environment Setup

```bash
# Python 3.8+
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow joblib tqdm streamlit
```

**GPU strongly recommended** — each training run takes 300–800 s/epoch on CPU.

Run notebooks in this order to reproduce from scratch:
1. `SVM_RF_DT_model/SVM_RF_DT_Model.ipynb`
2. `CNN/cnn_model.ipynb`
3. `Xception_model/Xception_model.ipynb`
4. `InceptionV3_model/InceptionV3_model.ipynb`
5. `Xception + Ensemble/Xception +Ensemble.ipynb`
6. `InceptionV3 + Ensemble/InceptionV3+Ensemble.ipynb`
7. `Final_model_Research/Final_model.ipynb`

---

## Uploading Heavy Files to GitHub

GitHub enforces a **100 MB hard limit** per file. The `.h5` model files (InceptionV3.h5, Xception.h5, etc.) and the MRI dataset exceed this limit.

### Option A — Git LFS (Recommended)
```bash
git lfs install
git lfs track "*.h5"
git lfs track "*.pkl"
git lfs track "*.sav"
git add .gitattributes
git add MODELS/
git commit -m "add models via Git LFS"
git push
```
GitHub gives 1 GB LFS storage free.

### Option B — Google Drive + gdown
Upload `.h5`/`.pkl` files to Google Drive, then add a download cell at the top of each notebook:
```python
!pip install gdown
import gdown
gdown.download("https://drive.google.com/uc?id=FILE_ID", "MODELS/InceptionV3.h5")
```

### Option C — HuggingFace Hub
```bash
pip install huggingface_hub
huggingface-cli login
# then upload via the Hub API
```

### Option D — .gitignore and document steps
```gitignore
*.h5
MRI_DATASET/
```
Commit only notebooks and small `.pkl` files; document exact training steps so anyone can reproduce.

---

## Methodology (from Research Report)

### Research Philosophy
Positivist + critical realism — empirical observation, measurable data, objective truths about the relative efficacy of various illness detection techniques, while acknowledging the complexity of medical imaging tasks.

### Research Approach
Deductive strategy (Research Onion model). Mixed-methods:
- **Quantitative:** accuracy, precision, recall, F1 score across all models
- **Qualitative:** focus groups / interviews with healthcare professionals

### Data Collection
- Primary: MRI images from Kaggle (JPG format)
- Secondary: literature review of existing brain tumor classification methods
- Tools: Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook

### Sampling
Stratified sampling — dataset divided into strata by disease class, samples chosen proportionally to preserve class distribution.

### Ethical Considerations
- No personal/sensitive data — Kaggle dataset is anonymised and publicly available
- No re-identification attempts
- Models supplement medical practitioners, not replace them
- Bias and fairness checks on demographic variety in dataset
- Data stored encrypted, access restricted, retained 5 years post-study then securely erased

---

## Literature Review Summary

| Author | Year | Method | Key Finding |
|--------|------|--------|------------|
| Patel | 2019 | CNN + LSTM for 3D CT | ICH identification at image level |
| Minz | 2017 | Adaboost + GLCM features | 89.90% accuracy for MR image classification |
| Zhou | 2018 | DenseNet + RNN | 92.13% with LSTM-DenseNet on 3D MRI |
| Cheng | 2015 | Offline DB + distance learning | 94.68% tumor classification |
| Abiwinanda | 2018 | Simple CNN | Glioma/meningioma/pituitary classification from 3,064 T-1 images |
| Sultan | 2019 | Deep neural network | Multi-class brain tumor classification |
| Talo et al. | 2019 | ResNet34 transfer learning | 100% accuracy on 613 MRIs (5-fold CV) |

**Gap identified:** Most methods focus on single-model solutions. No systematic comparison integrating ensemble learning, fine-tuned models, and classical ML on the same dataset.

---

## Conclusions (from Research Report)

### RO1 — Ensemble Approaches
Majority voting across 6 models (CNN, InceptionV3, Xception + their classical ML sub-ensembles) achieved **98.20%** accuracy — significantly outperforming any single model. The synergistic effect of combining diverse model predictions reduces individual model biases.

### RO2 — CNN Feature Extraction
CNN-extracted features (256 features via dense layer, 128×128 input) substantially improved SVM, DT, and RF classification accuracy compared to raw pixel features. Network depth, kernel size, and learning rate directly affect feature quality.

### RO3 — Pre-trained Models (Xception, InceptionV3)
Pre-trained models leverage transfer learning effectively. Standalone accuracies: Xception 86.91%, InceptionV3 84.63%. Their individual performance in ensemble sub-models was lower (70–71%), but they contributed to the 98.20% combined result.

### RO4 — Traditional ML + Ensemble
CNN-extracted features gave the best results for classical ML (SVM 66.6%, DT 80.99%, RF 85.45%). Ensemble combinations consistently outperformed individual classical models. Raw pixel SVM (88.26%) was the best standalone classical result.

---

## Recommendations (from Research Report)

1. **Deeper architectures** — experiment with more layers, kernel sizes, and learning rates for CNN feature extraction
2. **Advanced preprocessing** — noise reduction, intensity normalisation, image registration beyond basic resizing
3. **Expanded ensemble methods** — test stacking and boosting (not just majority voting)
4. **Data augmentation** — rotation, scaling, flipping to improve generalisation
5. **Multi-modal data** — combine MRI with CT scans and patient clinical data
6. **Real-time deployment** — optimise models for speed on clinical hardware
7. **Explainable AI (XAI)** — add attention mechanisms or LIME/SHAP for clinical interpretability
8. **Multi-disease detection** — extend framework to detect multiple diseases simultaneously

---

## Limitations (from Research Report)

- Dataset may not capture full diversity of real-world MRI scans (imaging equipment variability, patient demographics)
- Ensemble approaches require significantly more compute — not all clinical environments have GPU resources
- Pre-trained model dependency limits full customisation to brain tumor specifics
- Overfitting risk with complex CNN + ensemble models on high-dimensional MRI data
- Image preprocessing variations not fully explored
- Clinical interpretability and usability not assessed (focus was on accuracy metrics)

---

## Future Improvements

- Multi-disease detection models (simultaneous diagnosis of multiple conditions)
- Multi-task learning and attention mechanisms for complex pattern recognition
- Larger, more diverse training datasets covering multiple patient demographics
- Integration with clinical workflows for real-time diagnosis assistance
- Explainable AI techniques for medical practitioner trust and adoption
- Hardware-optimised models for low-latency deployment

---

## Known Code Issues

- `SVM_RF_DT_model/`: "Random Forest" cell instantiates `DecisionTreeClassifier` — DT and RF produce identical results (~80%)
- `CNN/cnn_model.ipynb`: Early cells have a `TypeError` (pth treated as list before fix) — later cells work correctly with `tf.keras.preprocessing.image_dataset_from_directory`
- All notebooks were developed on Windows (paths use `\\`). On Linux/Mac use `../MRI_DATASET/Training/`
- Xception/InceptionV3 standalone models show high validation loss despite reasonable accuracy — more fine-tuning layers or dropout recommended

---

---

## Notebook Flow Audit — Issues Found

Read all 9 notebooks. Several inconsistencies found that must be fixed before
the research can be considered reproducible. Severity: Critical / Medium / Minor.

### CRITICAL — Results Cannot Be Reproduced

| Issue | Notebook | Detail |
|-------|----------|--------|
| **Notebook is broken** | `CNN_WITH_SVM_RF_DT/CNN_WITH_SVM_RF_DT.ipynb` | Only 7.7 KB. Crashes with MemoryError before any model is trained. Results reported in the paper (RF 85.45%, DT 80.99%, SVM 66.60%) cannot be reproduced from this file. A complete rewrite is needed. |
| **Filename mismatch** | `CNN/cnn_model.ipynb` saves `cnn_model.sav`; `Final_model_Research/Final_model.ipynb` loads `MRI_model.sav` | The final ensemble will fail to load the CNN model unless the file is manually renamed or the path is fixed. |
| **Random Forest bug** | `SVM_RF_DT_model/SVM_RF_DT_Model.ipynb` | The "Random Forest" cell instantiates `DecisionTreeClassifier()` not `RandomForestClassifier()`. RF and DT results are identical because they are the same model. |
| **Double normalisation** | `CNN_WITH_SVM_RF_DT/CNN_WITH_SVM_RF_DT.ipynb` | Images loaded with `/255.0` then divided by `/255.0` again — pixel values become effectively zero before the MemoryError hits. |

### MEDIUM — Inconsistent Methodology Across Notebooks

| Issue | Affected Notebooks | Detail |
|-------|-------------------|--------|
| **Colour vs greyscale** | SVM_RF_DT uses `cv2.IMREAD_GRAYSCALE`; all DL notebooks use RGB 3-channel | Classical ML trains on greyscale (1 channel flattened); deep models train on RGB. If CNN features are fed to SVM/RF/DT, the feature distribution is inconsistent with the raw-pixel baseline. |
| **Normalisation formula** | SVM_RF_DT: `/255` · CNN/Xception/InceptionV3 standalone: `/255` via tf.keras · Ensemble sub-models: `/255 − 0.5` | The −0.5 shift means pixel range is [−0.5, 0.5] for ensemble models vs [0, 1] for standalone models. Final_model.ipynb mixes both without adjustment. |
| **Loss function / label encoding** | CNN, Xception, InceptionV3 standalone: `SparseCategoricalCrossentropy` (integer labels) · Xception+Ensemble, InceptionV3+Ensemble: `categorical_crossentropy` (one-hot via `LabelBinarizer`) | Inconsistent label encoding. Integer-label models and one-hot-label models cannot be directly compared or combined without conversion. |
| **Train/test split method** | SVM_RF_DT: explicit 80/20 `train_test_split` from Training folder only · All DL notebooks: 90/10 via `dataset.take/skip` + separate Testing folder | Two different evaluation protocols. SVM/RF/DT never uses the official Testing folder — it carves its own test set from Training. Results are not comparable to the DL models which use the official test split. |
| **Batch size** | Standalone DL: 32 · Ensemble sub-models: 5 | 6× difference. Smaller batch in ensemble training means noisier gradients and slower convergence. Not documented in the report. |
| **Epochs** | CNN: 20 · Xception/InceptionV3 standalone: 5 · Xception/InceptionV3 ensemble: 10 · AlexNet: 20 | No justification given for differences. Transfer learning models may be undertrained at 5 epochs (high validation loss observed in outputs confirms this). |

### MINOR — Incomplete Notebooks

| Issue | Notebook | Detail |
|-------|----------|--------|
| **No saved model** | `ALAEXNET/Alexnet.ipynb` | No `model.save()` call. AlexNet results cannot be loaded into the final ensemble. Test accuracy output cell is empty (model not evaluated). |
| **No test accuracy printed** | `ALAEXNET/Alexnet.ipynb` | `scores` cell has no output. AlexNet result is unknown and unreported in the research paper. |
| **Residual broken early cells** | `CNN/cnn_model.ipynb` | First 3 cells fail (`TypeError: pth treated as list`). Not a problem for reproduction since later cells use the correct `image_dataset_from_directory` approach, but confuses anyone reading the notebook. |

### Recommended Fix Order (before Phase 2 XAI work)

1. Fix `CNN_WITH_SVM_RF_DT.ipynb` — rewrite using the same pattern as Xception+Ensemble but with CNN backbone
2. Fix Random Forest bug — change `DecisionTreeClassifier()` to `RandomForestClassifier(n_estimators=100)`
3. Standardise normalisation — use `/255` (range 0–1) across all notebooks
4. Fix filename — rename saved CNN model to `MRI_model.sav` or update Final_model.ipynb to load `cnn_model.sav`
5. Standardise train/test — all notebooks should use the official `../MRI_DATASET/Testing/` folder for evaluation
6. Complete AlexNet — add `model.evaluate(test_ds)` and `model.save("AlexNet.h5")`
7. Delete broken early cells from CNN notebook

---

## XAI Extension — Phase 2 (Planned)

This section documents the planned Explainable AI upgrade to the existing research.
It directly addresses the limitation stated in the original report:
*"The study placed less emphasis on clinical usability and interpretability in favour of increasing accuracy metrics."*

### Literature Review — Honest Novelty Assessment

Before building Phase 2, a full literature search was conducted (May 2026).
The table below shows what has and has not been published.

| Proposed method | Prior art found? | Verdict |
|----------------|-----------------|---------|
| GRAD-CAM on a single model in medical imaging | Yes — standard practice | Zero novelty alone |
| Pixel-wise average of GRAD-CAM maps across models (chest X-ray) | Yes — Aasem & Iqbal (2024), Frontiers in Big Data | Incremental if applied to brain MRI |
| Confidence-weighted ensemble voting | Yes — heavily published (TSO-Optimised Voting 2024, Informatica; arXiv 2603.28357; MDPI Diagnostics 2025) | Zero novelty alone |
| IoU between model output and radiologist ground truth | Yes — Fusion-Based Brain Tumor arXiv 2508.06891 (IoU 0.78) | Zero novelty |
| **IoU between models' own attention maps (inter-model agreement)** | **Not found** | **Genuine novelty** |
| **Using inter-model IoU as a per-sample confidence/uncertainty signal** | **Not found** | **Genuine novelty** |
| **Disagreement-triggered clinical escalation rule** | **Not found for hybrid ensemble** | **Genuine novelty** |
| **Class-conditional analysis of attention disagreement** | **Not found** | **Genuine novelty** |

Key papers that must be cited and positioned against:
- Aasem & Iqbal (2024) "Ensemble-CAM for Thoracic Disease Localization" — PMC11096460
- Explainable Deep Ensemble Meta-Learning (2025) — PMC12427295 (EfficientNetB7+InceptionV3+Xception)
- Advanced Dynamic Ensemble with XAI (2025) — Nature Scientific Reports — PMC12334749
- TSO-Optimised Weighted Soft Voting (2024) — Informatica (99.92% on same dataset)

**Positioning statement for the paper:**
"Unlike existing ensemble-CAM methods that compare model attention against
radiologist-annotated bounding boxes, this work introduces Ensemble Attention
Agreement (EAA-IoU) — a per-sample inter-model saliency consensus score computed
from architecturally distinct deep models (CNN, Xception, InceptionV3). EAA-IoU
is used not as an accuracy metric but as a clinical uncertainty signal: low
inter-model agreement triggers escalation to radiologist review, independent of
the ensemble's final class prediction."

---

**New Research Question:**
Does inter-model GRAD-CAM agreement (EAA-IoU) correlate with misclassification
in a heterogeneous 6-model brain tumor ensemble, and can it serve as a
per-sample clinical uncertainty trigger?

**New Hypothesis:**
Images where CNN, Xception, and InceptionV3 produce low pairwise GRAD-CAM IoU
(models disagree on which region is the tumor) are significantly more likely to
be misclassified by the ensemble, enabling a disagreement-triggered escalation
rule that improves clinical safety without retraining any model.

---

### XAI Contribution 1 — Ensemble Attention Agreement (EAA-IoU) [NOVEL]

**What it is:**
Apply GRAD-CAM independently to CNN, Xception, and InceptionV3 on the same MRI.
Compute pairwise IoU between all three heatmaps (3 pairs).
Average into a single Ensemble Attention Agreement (EAA-IoU) score per image.
Use EAA-IoU as a clinical uncertainty signal — not as an accuracy metric.

**Why it is novel (vs existing literature):**
All existing ensemble-CAM / multi-model saliency papers (e.g. Aasem & Iqbal 2024)
measure IoU between model attention and a radiologist-drawn ground-truth bounding box.
This requires labelled segmentation data that most datasets (including this Kaggle set)
do not have. EAA-IoU instead measures whether the models agree with *each other* —
no ground truth annotations required. This framing as a self-supervised uncertainty
signal is not found in any published brain tumor paper.

**Pipeline:**

```
Input MRI image (256×256)
        │
        ├──► CNN          → GRAD-CAM heatmap A  (256×256, normalised 0–1)
        ├──► Xception     → GRAD-CAM heatmap B  (256×256, normalised 0–1)
        └──► InceptionV3  → GRAD-CAM heatmap C  (256×256, normalised 0–1)
                                    │
              Binarise each at threshold t=0.5
                                    │
         Pairwise IoU: IoU(A,B), IoU(A,C), IoU(B,C)
                                    │
         EAA-IoU = mean(IoU(A,B), IoU(A,C), IoU(B,C))
                                    │
         ┌─────────────────────────────────────────┐
         │ EAA-IoU ≥ 0.65 → HIGH CONFIDENCE       │
         │ EAA-IoU  0.40–0.65 → MODERATE           │
         │ EAA-IoU < 0.40 → FLAG for radiologist   │
         └─────────────────────────────────────────┘
                                    │
         Weighted-average pixel map → Consensus heatmap overlay
```

**Research questions this answers:**
1. Does low EAA-IoU correlate with misclassification? (testable on 2,063 test images)
2. Which tumor class has the lowest average EAA-IoU? (expected: glioma — matches confusion matrix)
3. What precision/recall tradeoff does an EAA-IoU escalation threshold give?
4. Can EAA-IoU flag the 1.8% of cases the ensemble misclassifies?

**Expected outputs:**
- EAA-IoU score and consensus heatmap for every test image
- Scatter plot: EAA-IoU vs prediction confidence — is there correlation?
- Box plot: EAA-IoU distribution per class (glioma expected lowest)
- ROC curve: EAA-IoU as a binary misclassification predictor
- Escalation threshold analysis: at EAA-IoU < X, what % of misclassifications are caught?

---

### XAI Contribution 2 — Confidence-Weighted Voting [KNOWN METHOD — implement honestly]

**Honest status:** Softmax-probability-weighted voting has been published many
times for brain tumor MRI (TSO-Optimised Voting, Informatica 2024; arXiv 2603.28357;
MDPI Diagnostics 2025). It is NOT a novel contribution on its own. It should be
implemented, evaluated, and cited properly as a known improvement to majority voting.

**Why include it anyway:**
- It gives a real confidence score per prediction (useful for the webapp)
- It is the baseline needed to test whether EAA-IoU adds value on top of it
- The combination of confidence-weighted vote + EAA-IoU uncertainty flag is the novel system

**Implementation:**

```python
# Replace:
final = max(set([pred_cnn, pred_inc, pred_xcp,
                 pred_cnn_ens, pred_inc_ens, pred_xcp_ens]),
            key=predictions.count)   # majority vote — 1 vote each

# With:
combined_probs = prob_cnn + prob_inc + prob_xcp + \
                 prob_cnn_ens + prob_inc_ens + prob_xcp_ens  # sum of 6 prob vectors
final_class = np.argmax(combined_probs)
vote_confidence = np.max(combined_probs) / combined_probs.sum()
```

**What to report:**
- Accuracy: majority vote (98.20%) vs confidence-weighted (expected 98.3–99.0%)
- Per-class improvement — especially glioma (hardest class)
- Accuracy is the baseline; EAA-IoU is the novel safety layer on top

---

### XAI New Files (to be created)

```
LMU_Research/
├── XAI_notebook/
│   ├── consensus_gradcam.ipynb     # GRAD-CAM on all 3 models + consensus map
│   ├── confidence_voting.ipynb     # Weighted vote vs majority vote comparison
│   └── xai_analysis.ipynb         # IoU scores, disagreement analysis, charts
└── webapp/
    ├── app.py                      # Streamlit web application
    ├── xai_engine.py               # GRAD-CAM + weighted voting logic
    ├── model_loader.py             # Loads all 6 models from MODELS/
    └── requirements.txt            # pip dependencies for the app
```

---

## Web Application — Clinical Demo System

A Streamlit web app that lets a user upload an MRI image and instantly see:
- The predicted tumor class and confidence score
- GRAD-CAM heatmaps from all three deep models
- The consensus activation map overlaid on the original MRI
- A per-model vote breakdown
- A clinical alert if ensemble confidence is low

### What the user sees (page layout)

```
┌─────────────────────────────────────────────────────────────┐
│  Brain Tumor MRI Classifier  —  XAI Ensemble System         │
├─────────────────────────────────────────────────────────────┤
│  [ Upload MRI Image ]                                       │
│                                                             │
│  ┌─────────────┐   ┌─────────────────────────────────────┐ │
│  │ Original MRI│   │  PREDICTION                         │ │
│  │   image     │   │  Class    : GLIOMA                  │ │
│  │             │   │  Confidence: 94.3%  ████████████░░  │ │
│  └─────────────┘   │  Status   : ✅ HIGH CONFIDENCE      │ │
│                    └─────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  GRAD-CAM HEATMAPS  — where each model looked              │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  │
│  │  CNN cam  │ │Xception   │ │InceptionV3│ │ CONSENSUS │  │
│  │  heatmap  │ │  heatmap  │ │  heatmap  │ │    MAP    │  │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘  │
│  Agreement Score (IoU): 0.81  — Models strongly agree      │
├─────────────────────────────────────────────────────────────┤
│  PER-MODEL VOTE BREAKDOWN                                   │
│  CNN              Glioma  97.1%  ████████████████████░     │
│  Xception         Glioma  91.4%  ██████████████████░░░     │
│  InceptionV3      Glioma  88.2%  █████████████████░░░░     │
│  CNN + Ensemble   Glioma  96.3%  ███████████████████░░     │
│  Inc + Ensemble   Glioma  90.1%  ██████████████████░░░     │
│  Xcp + Ensemble   Glioma  94.0%  ██████████████████░░░     │
├─────────────────────────────────────────────────────────────┤
│  CLASS PROBABILITY DISTRIBUTION                             │
│  Glioma      ████████████████████  94.3%                   │
│  Meningioma  ██░░░░░░░░░░░░░░░░░░   3.8%                   │
│  No Tumor    ░░░░░░░░░░░░░░░░░░░░   1.2%                   │
│  Pituitary   ░░░░░░░░░░░░░░░░░░░░   0.7%                   │
│                                                             │
│  ⚠  This tool is for research purposes only.               │
│     Always consult a qualified radiologist.                 │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Web UI | Streamlit | Page layout, file upload, charts |
| GRAD-CAM | tf-keras-vis | Generate heatmaps from Keras models |
| Heatmap overlay | OpenCV + Matplotlib | Colour map + overlay on original MRI |
| Model loading | TensorFlow / joblib | Load .h5 and .pkl model files |
| Confidence voting | NumPy | Weighted probability summation |
| Bar charts | Plotly | Interactive probability bars |

### Install dependencies

```bash
pip install streamlit tensorflow keras tf-keras-vis \
            opencv-python matplotlib numpy joblib \
            scikit-learn plotly pillow
```

### Run the app

```bash
cd LMU_Research/webapp
streamlit run app.py
```

App opens at `http://localhost:8501`

### Key functions in `xai_engine.py`

| Function | What it does |
|----------|-------------|
| `load_all_models()` | Loads all 6 models from MODELS/ once at startup |
| `preprocess_image(img, size)` | Resize + normalise to model input shape |
| `get_gradcam(model, img, class_idx)` | Returns 256×256 normalised heatmap |
| `build_consensus_map(maps, weights)` | Weighted pixel average of 3 heatmaps |
| `iou_score(map_a, map_b)` | Measures overlap between two heatmaps |
| `confidence_weighted_predict(img)` | Runs all 6 models, returns class + confidence |
| `overlay_heatmap(original, heatmap)` | Returns BGR image with red-blue cam overlay |

### Models required to run the webapp

All 6 models must be in `MODELS/` before starting the app:

```
MODELS/
├── MRI_model.sav          # CNN classifier
├── InceptionV3.h5         # InceptionV3 standalone
├── Xception.h5            # Xception standalone
├── cnn_ensemble.h5        # CNN feature extractor  ✅ in repo
├── inc_ensemble.h5        # InceptionV3 feature extractor
├── xcp_ensemble.h5        # Xception feature extractor
├── cnn_ensemble_model.pkl # CNN → classical ensemble  ✅ in repo
├── inc_ensemble_model.pkl # InceptionV3 → classical ensemble
├── xcp_ensemble_model.pkl # Xception → classical ensemble
└── ensemble_model.pkl     # Final voting ensemble
```

### Low-confidence alert logic

```python
CONFIDENCE_THRESHOLD = 0.70  # below this → show warning

if ensemble_confidence < CONFIDENCE_THRESHOLD:
    st.warning(
        "⚠ Low ensemble agreement detected. "
        "Models are uncertain about this image. "
        "Radiologist review strongly recommended."
    )
```

---

## Build Order for Phase 2

1. `XAI_notebook/consensus_gradcam.ipynb` — validate GRAD-CAM works on all 3 models
2. `XAI_notebook/confidence_voting.ipynb` — compare weighted vs majority vote on test set
3. `XAI_notebook/xai_analysis.ipynb` — IoU scores, disagreement charts, report figures
4. `webapp/model_loader.py` — model loading utility
5. `webapp/xai_engine.py` — all XAI logic
6. `webapp/app.py` — Streamlit UI
7. Test on 10–20 MRI samples from the test set before full demo

---

## References

Abiwinanda, N. et al. (2018) 'Brain tumor classification using convolutional neural network', *World Congress on Medical Physics and Biomedical Engineering*. Singapore.

Cheng, J. et al. (2015) 'Correction: Enhanced performance of brain tumor classification via tumor region augmentation and partition', *PLoS One*, 10(10), p. 0140381.

Cortes, C. (2019) 'Support-vector networks'.

Ismale, M. (2018) 'Brain tumor classification via statistical features and backpropagation neural network'.

M. Talo (2019) 'Application of deep transfer learning for automated brain abnormality classification using MR images'.

Minz, A. and Mahobiya, C. (2017) 'MR image classification using adaboost for brain tumor type', *IEEE 7th International Advance Computing Conference (IACC)*. Hyderabad.

Patel, A. et al. (2019) 'Image level training and prediction: intracranial hemorrhage identification in 3D non-contrast CT', *IEEE Access*, 7, pp. 92355–92364.

Sultan, H., Salem, N. and Al-Atabany, W. (2019) 'Multi-classification of brain tumor images using deep neural network', *IEEE Access*, 7, pp. 69215–69225.

Tripathi, P. and Bag, S. (2020) 'Non-invasively grading of brain tumor through noise robust textural and intensity based features', *Computers in Biology and Medicine*, 116, p. 103551.

Yousaf, F. et al. (2023) 'Multi-class disease detection using deep learning and human brain medical imaging', *Bioengineering*, 10(6), p. 704.

Zhou, Y. et al. (2018) 'Holistic brain tumor screening and classification based on DenseNet and recurrent neural network', *International MICCAI Brainlesion Workshop*. Springer.
