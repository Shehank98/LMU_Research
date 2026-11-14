# Brain Tumour MRI Classification with Explainable AI: final paper package

## Files in `output/`
| File | What it is |
|---|---|
| `Brain_Tumor_XAI_Research_Paper.docx` | Full paper. New material is in **yellow highlight**, new headings end with **[NEW]**, and new figures and code listings have an orange left border. |
| `Brain_Tumor_XAI_Research_Paper_clean.docx` | Same paper with no highlighting and no [NEW] tags. Submit this one. |
| `Brain_Tumor_XAI_Research_Paper_clean.pdf` | PDF of the clean version, used for the visual layout check. |
| `check_report.txt` | Output of the automated cross-checks (numbers, figure/table references, citations, placeholders, word count). |

The table of contents and the lists of figures, tables and listings are real Word fields. Word refreshes them when the
file is opened, because the file asks it to. Accept the "update fields" prompt, or press Ctrl+A then F9.

## What was added
* **Re-run results (Ch. 4).** All 22 model configurations were re-evaluated on the **1,600 held-out test images**,
  using the logged results of the 2026-10-26 Colab run. The chapter adds:
  * Wilson and bootstrap 95% intervals.
  * Exact McNemar tests with Holm correction.
  * ROC/PR curves, learning curves and every confusion matrix.
  * Reproducibility checks and an old-vs-new comparison.
* **New Chapter 5 on explainable AI**, run on the real saved models from Hugging Face:
  * Methods: Grad-CAM, Grad-CAM++, HiResCAM, LIME, SHAP (CNN and Random Forest stage) and occlusion.
  * Examples: correct and misclassified images for every class.
  * Faithfulness: deletion/insertion AUC compared with a random baseline.
  * Weight-randomisation sanity check (Adebayo et al., 2018).
  * Cross-model agreement: EAA-IoU on all 1,600 images, plus Spearman correlation and top-20% IoU.
  * A check for attention outside the brain.
  * Clinical trust and limitations.
* **Other new sections:**
  * Objectives RO5–RO6, RQ3 and H3.
  * Scope, project management and feasibility.
  * Hyperparameter table, reproducibility and the statistical protocol.
  * Discussion chapter, covering Pearson criteria M2/D1/D2, with a literature comparison and a plain-language clinician summary.
  * Reflection chapter (P6/P7/D3).
  * 13 code listings.
  * Appendices: criteria mapping, change log, figures index and reproducibility artefacts.
* **References:** 62 Harvard entries. Every DOI was checked on Crossref (`paper_assets/references/crossref_metadata.json`).
  The remaining 9 entries (NeurIPS/BMVC/OSDI/JMLR/arXiv papers, the Kaggle dataset, the HF mirror and the HF model
  repository) were checked by web search or on Hugging Face.

## What was corrected
See `AUDIT.md` and Appendix F (change log). The main corrections:
* The 98.20% figure was computed on the **training** images. The held-out result is **92.50%** (majority vote).
* H1 (pretrained > CNN) is **rejected**: the CNN scored 90.44% vs Xception 87.62% and InceptionV3 86.62%.
* "Fine-tuned" models are really **frozen-backbone** transfer learning.
* The InceptionV3/Xception wording error is fixed.
* Unsupported claims were removed: KNN, U-Net, Naive Bayes, SPSS, correlation/regression "analysis", interviews and
  focus groups, cross-validation, and noise reduction/registration.
* "PITIUTRY" and the 6,984 vs 6,985 count are explained, and the "Personnel" typo is fixed.
* Every old reference was completed and corrected. For example, Cheng et al. reported 91.28%, not 94.68%.
* The clinician questionnaire is now presented as a **proposed** instrument (Appendix D), not as collected data.

## Please verify manually
1. **Title page and declaration.** Check the programme name ("BSc (Hons) Data Science (top-up)"), the module leader
   (the template says Mr. Thepul Ginige; the interim report says Prof. Ruvan Abeysekara), the supervisor and the
   submission date. Sign the declaration.
2. **Dataset licence** on the Kaggle page. The Hugging Face mirror says CC BY 4.0; a secondary source says CC0. No
   Kaggle DOI could be verified, so the dataset is cited by URL with "n.d.".
3. **Template structure.** The London Met template lists project management, feasibility, design, implementation and
   testing chapters. This paper follows the research-report structure you asked for, and the template items are folded in:
   * Project management and feasibility → 3.3–3.4.
   * Design and implementation → 3.7–3.11 and the listings.
   * Testing and verification → 4.9 and `check_paper.py`.
   Check that your module accepts this.
4. **No GPU re-run is needed for any number in the paper.** Training-time figures come from the logged Colab run; all
   XAI and verification work ran on CPU from the saved models. If you retrain anything, run
   `paper_assets/build/build_paper.py` and `check_paper.py` again.
5. **The web demonstrator's alert threshold** (`EAA_LOW = 0.40` in `webapp/backend/xai_engine.py`) flags almost every
   image (see 5.7 and 7.3). Recalibrate it before showing the app to anyone.
6. **The questionnaire (Appendix D)** needs ethics approval before use.

## How to rebuild
```bash
# results are already in metrics/, confusion/, history/ ; models are downloaded from shehank98/brain-tumor-mri-models
python paper_assets/scripts/01_verify_models.py          # needs the test images in $MRI_DATA and models in $MODEL_DIR
python paper_assets/scripts/02_xai_examples.py
python paper_assets/scripts/03_xai_quantitative.py
python paper_assets/scripts/05_figures_stats.py
python paper_assets/scripts/06_xai_example_figures.py
python paper_assets/scripts/07_shap_rf.py
python paper_assets/scripts/08_diagrams.py
python paper_assets/scripts/09_rerun_final_ensemble.py
python paper_assets/scripts/10_xai_quant_figures.py
python paper_assets/build/build_paper.py
python paper_assets/build/check_paper.py output/Brain_Tumor_XAI_Research_Paper_clean.docx
/usr/bin/python3 paper_assets/build/render_pdf.py output/Brain_Tumor_XAI_Research_Paper_clean.docx output/Brain_Tumor_XAI_Research_Paper_clean.pdf
```
