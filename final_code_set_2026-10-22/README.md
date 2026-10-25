# Final code set — 2026-10-22

A single resumable Google Colab notebook that **re-trains every model from scratch and logs all results as data** (not images), so you can write the final report from reliable numbers.

**File:** `Brain_Tumor_Train_And_Log.ipynb`

## Why this version will not lose your work

| Problem you had | How this notebook solves it |
|---|---|
| Colab disconnects / crashes mid-training | Deep models **checkpoint after every epoch** to Google Drive. Re-run the cell and it **resumes from the last epoch** instead of starting over. |
| Local Colab disk is wiped after a session | Everything is written to **Google Drive**, so it survives. |
| One slow model breaks the whole run | Each model runs in **its own cell, isolated**. A failure is logged and the other models stay saved. |
| Results were never saved | Accuracy, precision, recall, F1, per-class scores, confusion matrices and per-epoch history are all saved as **CSV / JSON**. |
| Re-running wastes time | **Finished stages are skipped automatically.** |

## How to run

1. Upload `Brain_Tumor_Train_And_Log.ipynb` to Google Colab (or open it from Drive).
2. `Runtime > Change runtime type > GPU`.
3. In **cell 2**, set `DATA_DIR` to the folder that holds `Training/` and `Testing/` (four class folders each: `glioma, meningioma, notumor, pituitary`). If you do not have the dataset on Drive yet, use the **Optional — download the dataset** cell at the bottom.
4. Run cells top to bottom. Run the stage cells one at a time and wait for each to finish.
5. If anything breaks: reconnect, re-run cells 1–8, then re-run the stage cell that stopped. It resumes.

## Where your results end up (on Google Drive)

```
brain_tumor_results_2026-10-22/
├── metrics/
│   ├── results_master.csv              <-- one row per model: accuracy, P/R/F1, per-class F1, time
│   └── <model>_classification_report.json
├── confusion/<model>_confusion_matrix.csv
├── history/<model>_history.csv         <-- per-epoch train/val accuracy and loss
├── models/                             <-- trained .h5 / .pkl files
├── logs/run_log.txt                    <-- timestamped log of everything, including any errors
└── progress.json                       <-- which stages are done
```

`metrics/results_master.csv` is your results table for the report. The last cell also prints it and zips the whole folder for one-click download.

## Models trained and logged

Custom CNN, InceptionV3, Xception (standalone); CNN / InceptionV3 / Xception feature extractors feeding an RF + DT + SVM soft-vote ensemble; classical ML on raw pixels (RF, DT, SVM, Logistic Regression + ensemble); and the final 6-model ensemble (both majority vote and confidence-weighted vote).

## Note on the slow SVM

The classical SVM on raw pixels (65,536 features) is the slowest step. It cannot resume mid-fit, so it is run last and in its own stage; the moment it finishes it is saved, and any re-run skips it. Everything trained before it is already safe on Drive.
