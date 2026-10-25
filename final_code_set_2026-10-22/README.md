# Final code set — 2026-10-22

One Google Colab notebook, `Brain_Tumor_Train_And_Log.ipynb`, that:

1. retrains every model from scratch,
2. logs every result as data (CSV / JSON, never only images),
3. runs the explainability analysis (Grad-CAM family + EAA-IoU) on the whole test set,
4. uploads the models and results to Hugging Face, where the web app picks them up.

It is built so a crash or disconnect does not lose work, and it was tested end to end before
delivery, including a simulated crash in the middle of training.

---

## Part A. One-time setup (about 5 minutes)

**A1. Create a Hugging Face write token**
1. Go to huggingface.co, sign in as `shehank98`.
2. Click your profile picture, then **Settings > Access Tokens > Create new token**.
3. Token type: **Write**. Name it `colab-upload`. Create it and copy it.
   Keep it private: never paste it into a notebook cell, a report or a commit.

**A2. Give the token to Colab as a secret**
1. Open colab.research.google.com.
2. In the left sidebar click the **key icon (Secrets)**.
3. **Add new secret**: Name `HF_TOKEN`, Value = the token. Turn on **Notebook access**.
   (If you skip this, the upload cell asks you to paste the token instead.)

**A3. Check the dataset on Drive**
Your earlier runs used `MyDrive/MRI_DATASET`, and the notebook expects exactly that:
```
MyDrive/MRI_DATASET/
├── Training/  glioma/  meningioma/  notumor/  pituitary/
└── Testing/   glioma/  meningioma/  notumor/  pituitary/
```
If yours is somewhere else, change `DRIVE_DATA_DIR` in cell 2.

---

## Part B. Run the notebook in Colab

1. In Colab: **File > Upload notebook** and choose `Brain_Tumor_Train_And_Log.ipynb`.
2. **Runtime > Change runtime type > T4 GPU > Save.**
3. **Cell 1**: mounts Google Drive. Click *Connect to Google Drive* and allow access.
4. **Cell 2**: settings + dataset check. Every line must show an image count. The first time it copies
   the dataset from Drive to Colab's local disk (1 to 3 minutes). This is deliberate: reading images
   from Drive every epoch is very slow and is a common reason long runs stall.
5. **Cells 3 to 9**: set-up only, each finishes in seconds. Cell 3 must print a GPU line, not `NONE`.
6. **Stage cells**, run them one at a time, top to bottom, and wait for each to finish:

   | Order | Cell | What it does | Rough time on a T4 |
   |---|---|---|---|
   | 1 | `CNN_standalone` | Custom CNN, 20 epochs | 10–20 min |
   | 2 | `InceptionV3_standalone` | Transfer learning, 10 epochs | 10–20 min |
   | 3 | `Xception_standalone` | Transfer learning, 10 epochs | 15–25 min |
   | 4 | `CNN_feature_ensemble` | CNN features + RF / DT / SVM | 10–20 min |
   | 5 | `InceptionV3_feature_ensemble` | InceptionV3 features + RF / DT / SVM | 10–20 min |
   | 6 | `Xception_feature_ensemble` | Xception features + RF / DT / SVM | 10–20 min |
   | 7 | `Traditional_raw_pixels` | RF, DT, SVM, Logistic Regression on raw pixels | 30–90 min (SVM is slow) |
   | 8 | `FINAL_ensemble` | 6-model majority and confidence-weighted vote | about 5 min |
   | 9 | `EAA_IoU_analysis` | CAM maps + EAA-IoU for all 2,063 test images | 10–20 min |
   | 10 | `HF_upload` | Upload models and results to Hugging Face | 5–10 min |

   Times are estimates; free Colab can be slower when busy.

7. **Results summary cell**: prints the results table and downloads a zip of everything.

**If Colab disconnects or a cell fails**
- Reconnect, run **cells 1 to 9** again (seconds), then re-run the stage cell that stopped.
- Deep models continue from the **last finished epoch**; classical models already saved are skipped;
  EAA-IoU continues from the last image written; finished stages print `SKIP`.
- A failure in one stage is logged and isolated; it never deletes another stage's results.
- Free Colab sessions end after a few hours. That is fine: just reconnect and continue the next day.

**Where your results are (Google Drive)**
```
MyDrive/brain_tumor_results_2026-10-22/
├── metrics/results_master.csv                  one row per model: accuracy, precision, recall, F1, per-class F1, time
├── metrics/<model>_classification_report.json  per-class precision / recall / F1
├── metrics/final_per_image_predictions.csv     every test image: true class, each model's vote, ensemble result
├── metrics/eaa_iou_per_image.csv               every test image: pairwise IoU and EAA-IoU, correct or not
├── metrics/eaa_iou_summary.json                EAA correct vs wrong, per class, AUROC, Mann-Whitney p
├── metrics/eaa_iou_threshold_table.csv         % flagged and % errors caught for each threshold
├── confusion/<model>_confusion_matrix.csv
├── history/<model>_history.csv                 per-epoch train / validation accuracy and loss
├── models/                                     trained .h5 / .pkl files
├── logs/run_log.txt                            timestamped log of everything, including errors
└── progress.json                               which stages are done
```
The same files are uploaded to Hugging Face under `results_2026-10-22/`.

---

## Part C. Make the web app use the new models

The web app downloads its models from `shehank98/brain-tumor-mri-models` (`models/` folder) when it
starts. The notebook's upload cell writes exactly the files the app expects:
`cnn_model.h5, inceptionv3_model.h5, xception_model.h5, cnn_ext.h5, inceptionv3_ext.h5,
xception_ext.h5, cnn_ensemble_model.pkl, inceptionv3_ensemble_model.pkl, xception_ensemble_model.pkl, manifest.json`.

1. **Deploy the updated web app code.** These changes are on the branch
   `claude/interim-report-cnn-inception-hhotc0`. Railway must deploy from that branch (or merge it into
   the branch Railway uses).
2. **Railway variables** (Project > Variables). `HF_REPO_ID` now defaults to
   `shehank98/brain-tumor-mri-models` in the code, so nothing is required. Optional:
   `RESULTS_PREFIX = results_2026-10-22`.
3. **After every new upload from Colab**, either redeploy or open a terminal and run:
   ```
   curl -X POST https://<your-railway-domain>/api/admin/reload
   ```
4. **Check it worked**: open `https://<your-railway-domain>/api/models/status`.
   `loaded_count` should be `9`. The header of the site shows **9 of 9 model files ready**.
5. Open the site: **Diagnose** to upload a scan, **Results** for the live numbers from your run
   (badge: *Live results · results_2026-10-22*), **About** for the method.

---

## Part D. Using the results in the final report

| Report section | File |
|---|---|
| Main comparison table | `metrics/results_master.csv` |
| Per-class precision / recall / F1 | `metrics/<model>_classification_report.json` |
| Confusion matrices | `confusion/<model>_confusion_matrix.csv` |
| Training curves (plot them yourself) | `history/<model>_history.csv` |
| Ensemble analysis per image | `metrics/final_per_image_predictions.csv` |
| Explainability chapter (EAA-IoU) | `metrics/eaa_iou_summary.json`, `eaa_iou_threshold_table.csv`, `eaa_iou_per_image.csv` |

### Method note for the explainability chapter
The attention maps are computed with **HiResCAM** (Draelos and Carin, 2020), a member of the Grad-CAM
family, from the **pre-softmax class score**. Two reasons, both verified in testing:
- Standard Grad-CAM averages each channel's gradient over the image. That is exact for models with a
  Global Average Pooling head (InceptionV3, Xception), where HiResCAM and Grad-CAM give identical maps.
  The custom CNN, however, has a Flatten + Dense head; there the averaging cancels out. In a controlled
  test on synthetic scans with a lesion at a known position, Grad-CAM returned an **empty map in 4 of 4
  images**, while HiResCAM placed its peak within about 5 px of the lesion (radius 11 px).
- Using the softmax output instead of the class score makes confident models give vanishing
  gradients and empty maps.

To also report standard Grad-CAM for comparison, set `CAM_METHOD = 'gradcam'` in cell 2 and re-run
only the `EAA_IoU_analysis` stage (delete `metrics/eaa_iou_per_image.csv` and its entry in
`progress.json` first). The method used is recorded in `eaa_iou_summary.json`.

Reference: Draelos, R.L. and Carin, L. (2020) *Use HiResCAM instead of Grad-CAM for faithful
explanations of convolutional neural networks*. arXiv:2011.08891.
