# XAI Notebooks — Phase 2

Run these notebooks in order after all 8 training notebooks have completed.

| Notebook | Internal name | Purpose |
|----------|--------------|---------|
| `consensus_gradcam.ipynb` | `09_ConsensusGradCAM` | GRAD-CAM on CNN, InceptionV3, Xception → pairwise IoU → EAA-IoU CSV + charts |
| `confidence_voting.ipynb` | `10_ConfidenceVoting` | Compare confidence-weighted voting vs majority vote on full 2,063-image test set |
| `xai_analysis.ipynb`      | `11_XAIAnalysis`     | ROC/PR curves, class-conditional box plots, escalation threshold analysis |

## Run order

1. Notebooks 01–06 (training notebooks in `standardised_notebooks/`)
2. `consensus_gradcam.ipynb` — produces `eaa_iou_results.csv`
3. `confidence_voting.ipynb` — produces `voting_comparison_results.csv`
4. `xai_analysis.ipynb` — loads both CSVs and produces publication figures

## Colab Secrets (set before running)

| Secret name | Value |
|---|---|
| `KAGGLE_USERNAME` | `sk1285` |
| `KAGGLE_KEY` | `7261c6b4046a6bd5c9ba4d1a6f58c98f` |
| `HF_TOKEN` | *(your HuggingFace write token)* |

## Quick start (local)

```bash
pip install tensorflow keras opencv-python matplotlib numpy pandas \
            scikit-learn joblib huggingface_hub tqdm scipy seaborn
jupyter notebook
```

Open `consensus_gradcam.ipynb` first.

## Charts produced

### consensus_gradcam.ipynb
- `sample_gradcam_grid.jpg` — 4×5 grid: original + 3 model heatmaps + consensus per class
- `eaa_iou_distribution.jpg` — histogram and mean EAA-IoU per class
- `eaa_iou_low_vs_high.jpg` — example images at extreme IoU values

### confidence_voting.ipynb
- `voting_comparison_confusion_matrix.jpg` — side-by-side confusion matrices
- `voting_comparison_per_class_f1.jpg` — F1 improvement per class
- `voting_confidence_distribution.jpg` — confidence score distribution

### xai_analysis.ipynb
- `eaa_iou_roc_pr_curve.jpg` — ROC and PR curves for EAA-IoU as uncertainty predictor
- `eaa_iou_by_class_boxplot.jpg` — class-conditional IoU box plots
- `escalation_threshold_analysis.jpg` — sensitivity/specificity tradeoff at each threshold
- `eaa_iou_vs_confidence_scatter.jpg` — scatter plot of IoU vs vote confidence
