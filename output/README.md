# Report package

| File | What it is |
|---|---|
| `Brain_Tumor_XAI_Research_Paper.docx` | The report (Word). Contents, figure and table lists are Word fields: accept "update fields" when Word asks, or press Ctrl+A then F9. |
| `Brain_Tumor_XAI_Research_Paper.pdf` | PDF of the same report, for checking the layout. |
| `Paper_Charts_Editable.xlsx` | Every data chart of the report as a native Excel chart with its numbers next to it. Blue cells are inputs you can change; black cells are formulas. Sheet names start with the figure number used in the report. |
| `check_report.txt` | Output of the automatic checks: every number backed by a result file, every figure/table/listing referenced, every citation in the reference list, no placeholders, no em or en dashes. |

## Main results
| | Full test set (1,600) | Duplicate-free test subset (1,264) |
|---|---|---|
| Baseline six-model majority vote | 92.50% | 90.66% |
| Improved six-model majority vote (head cropping, augmentation, fine-tuning) | **95.62%** | **94.70%** |

* The improvement is statistically significant (exact McNemar test, Holm-adjusted p below 10^-8).
* The best single model is fine-tuned InceptionV3 (95.44%). In the improved set-up the ensemble is not significantly better than it.
* Glioma is the weak class (85% recall). Most remaining errors are gliomas predicted as meningioma, and for 2.56% of test images all six models are wrong.
* 98% was not reached on this test set. Part of the reason is that 21% of the official test images duplicate training images; figures above 98% reported on this dataset should be read with that in mind.

## Where things come from
* Baseline: `final_code_set_2026-10-22/` (Colab), results in `metrics/`, `confusion/`, `history/`.
* Improved: `src/` (run `src/run_all.sh`), results in `runs/v2/`.
* Analysis and figures: `paper_assets/scripts/` (set `MODEL_SET=improved` for the improved models).
* Report build and checks: `paper_assets/build/build_paper.py`, `paper_assets/build/check_paper.py`.
* Issues found in the code and how they were fixed: `CODE_REVIEW.md`.

## Web app
* The alert threshold is EAA-IoU below 0.10 and "strong agreement" is 0.20 or above (`webapp/backend/xai_engine.py`).
* To use the improved models, upload them once with `HF_TOKEN=<write token> python src/upload_to_hf.py`, then set `MODEL_SET=improved` on Railway.
* With the improved models, agreement is higher overall, so 0.10 flags only about 3% of images (see report Section 6.5). Consider 0.20 if you want more cases sent for review.

## Please check before submitting
1. Title page and declaration: programme name, module leader, supervisor, date, signature.
2. The dataset licence on the Kaggle page.
3. Your university's policy on declaring the use of AI tools. If a declaration is required, add it (for example in the acknowledgements).
4. `AUDIT.md` and `CLAUDE.md` in the repository root describe earlier work and how the project was set up. Decide whether they should stay in the repository you show the panel.
