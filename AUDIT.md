# Audit of the previous HND report (Final_Research_Report.pdf, 2024-10-18)

This is Step 0 of the paper update. Each finding gives the evidence, the source file and how the new paper handles it.
Source files: `metrics/`, `confusion/`, `history/` and `logs/` (copied from the Hugging Face results folder
`results_2026-10-22/` of `shehank98/brain-tumor-mri-models`), the analysis outputs in `paper_assets/xai_results/`,
and the training code in `final_code_set_2026-10-22/`.

## A. Critical findings

### A1. The 98.20% "final ensemble" figure was measured on the training images
* Evidence: Table 25 and Table 26 of the old report have supports of 1,733 / 1,699 / 1,847 / 1,706 = **6,985**. These are the
  training-folder counts in Table 1 (1,733 / 1,699 / 1,847 / 1,705 = 6,984). The test folder had 2,063 images, so the
  98.20% score cannot come from held-out data.
* Re-run (2026): all six predictors were evaluated on the **1,600 held-out test images** of the current
  version of the dataset (400 per class, never used for training or model selection):
  * Majority vote: **92.50%** accuracy, macro-F1 0.9237 (`metrics/FINAL_majority_vote_classification_report.json`).
  * Confidence-weighted vote: **92.25%**, macro-F1 0.9206.
* Independent check (this audit): `paper_assets/scripts/09_rerun_final_ensemble.py` reloads every saved model from
  Hugging Face and re-predicts the test set. It reproduces 92.50% / 92.25%, and every per-image prediction matches
  `metrics/final_per_image_predictions.csv` (agreement 1.0 for all six members and both votes;
  `paper_assets/xai_results/final_ensemble_rerun.json`).
* In the paper: the 92.50% held-out figure is the headline. The 98.20% figure is reported once, clearly labelled
  "training images, HND 2024", in the comparison table and figure. The test result is 5.70 percentage points lower.

### A2. Hypothesis 1 ("fine-tuned models outperform the CNN") is rejected by the data
* CNN 90.44%, Xception 87.62%, InceptionV3 86.62% (`metrics/results_master.csv`).
* Exact McNemar tests on the same 1,600 images (`paper_assets/xai_results/mcnemar.csv`):
  * CNN vs Xception: p = 0.00049 (Holm-adjusted 0.0018).
  * CNN vs InceptionV3: p = 2.2e-6.
  * Xception vs InceptionV3: p = 0.10, which is not significant.
* Same at feature level: CNN features + soft vote 89.31%, Xception features 88.31%, InceptionV3 features 86.12%.
* In the paper: H1 is stated as **not supported**. H2 (the ensemble beats every single model) is **supported**: the
  majority vote beats every member, with Holm-adjusted McNemar p < 0.002 in all six comparisons.

### A3. "Fine-tuned" is the wrong word for the pretrained models
* Code: `_base()` in `final_code_set_2026-10-22/_generators/build_notebook.py` sets `b.trainable = False`.
  The ImageNet backbones are **frozen**. Only the new head is trained: GlobalAveragePooling + Dense(4) for the
  standalone models, and Flatten + Dense(256) + Dropout + Dense(4) for the 128-pixel feature extractors.
* In the paper: they are described as transfer learning with frozen backbones (feature extraction). Fine-tuning is
  moved to future work. The title keeps the registered wording, and the text explains the term.

### A4. Inception V3 vs Xception wording
* The old text (4.2.2) said "Inception V3 fared better than Xception, which came in at 86.91%", but 84.63% < 86.91%.
* Re-run: Xception 87.62% vs InceptionV3 86.62%. Xception is higher, but not significantly (McNemar p = 0.10).

## B. Claims not backed by work (removed or reworded)
| Old claim | Where | Action |
|---|---|---|
| KNN in the conceptual framework (Fig. 4, 2.2) | Ch. 1–2 | Removed. No KNN was trained. |
| U-Net, Naive Bayes | 1.1 | Removed. Not used. |
| "SPSS charts" (Annexure B title) | Annex B | Renamed. All analysis was done in Python. |
| "Correlation analysis" / "Regression analysis" (4.2, 4.3, 5.1) | Ch. 4–5 | Replaced. No correlation or regression model was fitted. The new analysis uses per-class precision/recall/F1, confusion matrices, ROC/PR AUC, exact McNemar tests with Holm correction, bootstrap 95% CIs and learning curves. |
| Focus groups / interviews with healthcare professionals; "qualitative views" in the abstract | Abstract, 3.2–3.4 | No responses exist. The study is now described as quantitative. The questionnaire (Annex C) is presented as a **proposed** clinician-validation instrument and as future work. |
| Cross-validation | 3.9 | Not performed. A fixed 90/10 train/validation split with seed 42 plus the official test folder is used. CV is listed as a limitation. |
| Noise reduction, intensity normalisation, image registration | 1.5, 3.9 | Not performed. Pre-processing is resize + RGB/255 only. |
| Averaging / stacking / boosting ensembles | 1.5, 2.2 | Only soft voting (classical stage) and majority / confidence-weighted voting (final stage) were used. |
| PyTorch | 3.6.3 | Not used. Only TensorFlow/Keras and scikit-learn. |
| Data augmentation "will be used" | 5.2 | Not applied in training. The paper states that no augmentation was applied. |

## C. Data, counts and typos
* Table 1: "PITIUTRY" → "Pituitary". The training total in Table 1 is 6,984 (pituitary 1,705), but Tables 25–26
  use 1,706 / 6,985. The old figures are reproduced only as historical values, with this note.
* Dataset version: the 2024 report used a 9,047-image copy. The re-run uses the current 7,200-image release
  (5,600 train / 1,600 test, 1,400 and 400 per class; `metrics/dataset_counts.json`). The old and new numbers are
  therefore not directly comparable, and the paper says so.
* **103 of the 400 meningioma test images are augmented copies** (file names `Te-aug-me_*`). Ensemble recall is
  94.17% on these and 95.96% on the 297 original images. The CNN alone scores 85.44% vs 94.61%.
  Reported as a data-quality limitation (`paper_assets/xai_results/results_stats.json → meningioma_aug`).
* Two pre-processing paths in the code: standalone metrics use the Keras loader (`image_dataset_from_directory`,
  TensorFlow resize). The final ensemble uses OpenCV resizing. The standalone members therefore score 90.50 / 86.38 /
  87.75% inside the ensemble pipeline, vs 90.44 / 86.62 / 87.62% in `results_master.csv`. Both are reported, and the
  difference is explained.
* "Personnel Reflection" → "Personal Reflection". "Edexcel UK" → Pearson.
* Literature numbers corrected:
  * Cheng et al. (2015) reported 91.28%, not 94.68%.
  * Cortes (2019) → Cortes and Vapnik (1995).
  * "Ismale" → Ismael and Abdel-Qader (2018).
  * Tripathi and Bag is a 2019 conference chapter (AISC), not a 2020 *Computers in Biology and Medicine* article.
  * Yousaf et al. (2023) is in *Biomedical Signal Processing and Control* 85, 104875.
  * All old references were incomplete ("s.l.: s.n.") and are now full Harvard entries verified on Crossref
    (`paper_assets/references/crossref_metadata.json`).
* Old-code issues already fixed in the 2026 re-run:
  * Random Forest cell instantiated a DecisionTreeClassifier (re-run RF 87.00% vs DT 77.88%).
  * Raw-pixel models were tested on a split carved from the training folder.
  * Mixed normalisation (/255 vs /255 − 0.5).
  * Batch sizes 5 vs 32.
  * Training-set evaluation of the final ensemble.

## D. Findings on the EAA-IoU contribution (existing custom metric in the repo)
* `metrics/eaa_iou_summary.json` (whole test set, HiResCAM maps):
  * Mean EAA-IoU is 0.1029 for correctly classified images vs 0.0471 for misclassified images
    (one-sided Mann–Whitney p = 4.3e-21).
  * AUROC 0.7525 for flagging errors.
* Caveats reported honestly:
  * The thresholds planned in the proposal (0.40 / 0.65) flag **100%** of images, because EAA-IoU values are small
    (mean 0.0985). Pairs that include the CNN have mean IoU of about 0.005. Data-driven thresholds of 0.05 and 0.10
    flag 28.62% and 50.81% of images and catch 61.29% and 82.26% of errors.
  * The ensemble's own confidence is a **stronger** error flag (AUROC 0.9127) than EAA-IoU.
* The novelty claim is therefore kept modest. EAA-IoU is presented as an annotation-free agreement signal that is
  significantly associated with error, not as a replacement for confidence.

## E. Items the student must verify manually
1. Module code, module leader and supervisor names on the title page and declaration. The template says
   Mr. Thepul Ginige; the interim report says Prof. Ruvan Abeysekara and Ms. Hiruni Wimalaweera.
2. The dataset licence on the Kaggle page. The Hugging Face mirror card says CC BY 4.0, and a secondary source says
   CC0. No DOI could be verified for the Kaggle dataset, so it is cited by URL.
3. GPU re-runs are **not** required for any number in the paper. All XAI results were produced on CPU from the
   saved models. Training-time numbers come from the 2026-10-26 Colab run logs.
4. If any clinician feedback is collected before submission, Annex C and §7 must be updated.
