# XAI Notebooks — Phase 2

Run these notebooks in order after all models in MODELS/ are available.

| Notebook | Purpose |
|----------|---------|
| `consensus_gradcam.ipynb` | GRAD-CAM on CNN, Xception, InceptionV3 + consensus map + IoU scores |
| `confidence_voting.ipynb` | Compare confidence-weighted voting vs majority vote on full test set |
| `xai_analysis.ipynb`      | Disagreement analysis, per-class IoU charts, figures for report |

## Quick start

```bash
pip install tf-keras-vis opencv-python matplotlib numpy joblib tensorflow
jupyter notebook
```

Open `consensus_gradcam.ipynb` first.
