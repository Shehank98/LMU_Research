# Individual model notebooks

One Colab notebook per model, so each can be run, checked and explained on its own.
They contain exactly the same code as `../Brain_Tumor_Train_And_Log.ipynb` (they are generated from it),
and they write to the same Google Drive folder, so you can mix them: run some here and some in the
all-in-one notebook. Finished work is never repeated.

## Run order

| # | Notebook | What it does | Needs first | Time on a T4 |
|---|---|---|---|---|
| 01 | `01_CNN.ipynb` | Custom 4-block CNN, trained from scratch (256 px) | – | 10–20 min |
| 02 | `02_InceptionV3.ipynb` | InceptionV3 transfer learning, frozen backbone + new head | – | 10–20 min |
| 03 | `03_Xception.ipynb` | Xception transfer learning, frozen backbone + new head | – | 15–25 min |
| 04 | `04_CNN_Feature_Ensemble.ipynb` | 128 px CNN; its 256-value `dense` layer feeds RF + DT + SVM (soft vote) | – | 10–20 min |
| 05 | `05_InceptionV3_Feature_Ensemble.ipynb` | Same idea with InceptionV3 features (`dense_2`) | – | 10–20 min |
| 06 | `06_Xception_Feature_Ensemble.ipynb` | Same idea with Xception features (`dense_3`) | – | 10–20 min |
| 07 | `07_Traditional_ML_Raw_Pixels.ipynb` | RF, DT, SVM, Logistic Regression on 65,536 raw pixels | – | 30–90 min |
| 08 | `08_Final_Ensemble.ipynb` | Combines the 6 predictors: majority vote and confidence-weighted vote | 01–06 | ~5 min |
| 09 | `09_EAA_IoU_Analysis.ipynb` | Attention maps for CNN / Xception / InceptionV3, EAA-IoU for every test image | 01–03, 08 | 10–20 min |
| 10 | `10_Upload_to_HuggingFace.ipynb` | Optional final sync of every model and result (useful if an earlier upload failed) | whatever is finished | 5–10 min |

## Every notebook uploads its own model to Hugging Face

The last cell of each notebook uploads what that notebook trained to
`shehank98/brain-tumor-mri-models`, together with every result file so far:

| Notebook | Uploaded to `models/` (used by the web app) | Uploaded to `models/classical/` |
|---|---|---|
| 01 | `cnn_model.h5` | – |
| 02 | `inceptionv3_model.h5` | – |
| 03 | `xception_model.h5` | – |
| 04 | `cnn_ext.h5`, `cnn_ensemble_model.pkl` | `cnn_feat_rf.pkl`, `cnn_feat_dt.pkl`, `cnn_feat_svm.pkl` |
| 05 | `inceptionv3_ext.h5`, `inceptionv3_ensemble_model.pkl` | `inceptionv3_feat_rf/dt/svm.pkl` |
| 06 | `xception_ext.h5`, `xception_ensemble_model.pkl` | `xception_feat_rf/dt/svm.pkl` |
| 07 | – | `raw_rf.pkl`, `raw_dt.pkl`, `raw_svm.pkl`, `raw_lr.pkl` |
| 08, 09 | no new model, only result files | – |

- It needs your Hugging Face **WRITE** token saved once as the Colab secret `HF_TOKEN` (key icon on the left).
- It only uploads after training has finished; a half-trained model is never uploaded.
- If the upload fails (for example a network error), just re-run that last cell. Training is not repeated.
- If the web app is already running, call `POST /api/admin/reload` (see `../README.md`) so it loads the new model.

## Order

01 to 07 are independent: run them in any order, on different days, or in two Colab accounts.
08 and 09 check their prerequisites first and stop with a clear message if something is missing.
08 refuses to build a partial ensemble (fewer than 6 predictors) unless you set
`ALLOW_PARTIAL_ENSEMBLE = True` in cell 2, so an incomplete ensemble can never slip into the report.

## Inside every notebook

| Cells | Content |
|---|---|
| Top text | What the notebook does, what it needs, what it saves, how long it takes |
| 1–7 | **Shared and identical in every notebook**: Google Drive, settings, logging, metric saving, crash-safe training, data loading, helpers |
| 8–9 | **This model only**: the model definition and its training / evaluation code |
| Run cell | `stage('...', run_...)`: runs it safely (resumes after a crash, skips if already done) |
| Results cell | Reads this model's results back from Drive: metrics, confusion matrix, last training epochs |
| Last cell | Uploads this notebook's model(s) and all result files to Hugging Face |

## How to run one

1. Colab: **File > Upload notebook**, pick the notebook.
2. **Runtime > Change runtime type > T4 GPU**.
3. **Runtime > Run all**.
4. If it disconnects: reconnect and **Run all** again. Training continues from the last finished epoch.

Do not run the **same** notebook in two Colab tabs at once; they would write to the same files.
For the one-time setup (Hugging Face token, dataset location) and how the web app picks up the
models, see `../README.md`.
