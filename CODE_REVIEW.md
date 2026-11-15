# Code review notes

This note lists the problems found in the project code and what was done about each one.

## Where the final code is

| Folder | What it contains |
|---|---|
| `src/` | Improved pipeline. One module per step: `config.py`, `data.py` (loading, head cropping, split), `models.py`, `train.py`, `classical.py`, `ensemble.py`, `metrics.py`; `run_all.sh` runs everything and skips finished steps. |
| `final_code_set_2026-10-22/` | Baseline pipeline (Colab notebook and one notebook per model). |
| `paper_assets/scripts/` | Analysis behind the report: model verification, explainability, statistics, figures, the editable Excel charts. Set `MODEL_SET=improved` to run them on the improved models. |
| `paper_assets/build/` | Builds the report from the result files and checks it (`build_paper.py`, `check_paper.py`). |
| `webapp/` | Demonstrator (FastAPI backend + React frontend; `MODEL_SET=improved` selects the improved models). |

Older folders (`CNN/`, `ALAEXNET/`, `original_notebooks/`, `standardised_notebooks/`, `training/`, `XAI_notebook/`) are earlier
versions kept for reference. They are not used for any number in the report.

## Issues found and how they were handled

| # | Where | Issue | Effect | Action |
|---|---|---|---|---|
| 1 | Baseline notebook | Standalone accuracy was measured with the Keras image loader, but the final ensemble re-loaded images with OpenCV. The two resize routines differ slightly. | Standalone accuracy differs by up to 0.24 points between the two paths. | Both values reported. `src/` uses one loader everywhere. |
| 2 | Baseline notebook | The validation split was random, not stratified. | Slightly uneven class mix in validation. | `src/data.py` uses a stratified split. |
| 3 | Baseline CNN | Flatten + Dense head ties the decision to fixed pixel positions. | Lets the CNN use the image border as a cue (report section 5.7). | Improved CNN uses batch normalisation and global average pooling. |
| 4 | Data | Image size and background differ by class because the classes come from different collections. | Models can partly classify from the background (78% from simple image statistics). | Images are cropped to the head (`crop_head` in `src/data.py`); augmentation added. |
| 5 | Data | 336 test images are near-identical copies of training images. | Accuracy on the full test set is optimistic. | Duplicates found with `paper_assets/scripts/13_duplicate_check.py`; results also reported on the 1,264 clean images. |
| 6 | Baseline classical stage | The SVM was trained on unscaled features. | SVMs are sensitive to feature scale. | `src/classical.py` standardises features before the SVM. |
| 7 | Baseline transfer models | Backbones were frozen, so the models were not fine-tuned. | Under-fitted pretrained models. | Improved run: head first, then the upper blocks fine-tuned with a lower learning rate. |
| 8 | Web app | The EAA-IoU alert threshold (0.40) and "strong agreement" level (0.65) were set before any results existed. | Every image was flagged. | Set to 0.10 and 0.20 from the observed distribution; the same constants are used in the API and the Streamlit pages. |
| 9 | Web app | For cropped models, heatmaps were drawn on the uncropped upload. | Heatmap and image would not line up. | Heatmaps are drawn on the cropped image when `MODEL_SET=improved`. |
| 10 | Baseline notebook (first run) | Two layers were given the same name (`dense`). | The first training attempt stopped with an error (see `logs/run_log.txt`). | Layer names made unique; the logged results come from the corrected run. |
| 11 | XAI analysis | The quantitative XAI script ran out of memory during the sanity check. | Summary file not written. | Split into `03_xai_quantitative.py` and `03b_xai_summary_sanity.py`. |
| 12 | All Python code | Inconsistent formatting. | Harder to read. | Formatted with `black` (line length 110). |

## How to reproduce
```bash
cd src && MRI_DATA=/path/to/MRI_DATASET ./run_all.sh          # improved models -> runs/v2/
cd ../paper_assets/scripts && MODEL_SET=improved python 02_xai_examples.py   # and the other analysis scripts
python ../build/build_paper.py && python ../build/check_paper.py
```
