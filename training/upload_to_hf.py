"""
upload_to_hf.py — Upload all trained model files to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxx
    export HF_REPO_ID=your-username/brain-tumor-models
    python training/upload_to_hf.py

Or inline:
    HF_TOKEN=hf_xxx HF_REPO_ID=user/repo python training/upload_to_hf.py

The repo is created automatically if it doesn't exist.
All 9 model files are uploaded; existing files are overwritten.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import MODELS_DIR

HF_TOKEN   = os.environ.get('HF_TOKEN', '')
HF_REPO_ID = os.environ.get('HF_REPO_ID', '')

MODEL_FILES = [
    'cnn_standalone.h5',
    'InceptionV3.h5',
    'Xception.h5',
    'cnn_ensemble.h5',
    'inc_ensemble.h5',
    'xcp_ensemble.h5',
    'cnn_ensemble_model.pkl',
    'inc_ensemble_model.pkl',
    'xcp_ensemble_model.pkl',
]

MODEL_CARD = """\
---
tags:
- brain-tumor
- mri
- classification
- tensorflow
- ensemble
license: mit
---

# Brain Tumor MRI Classification — Ensemble Models

**Research project:** A Comparative Analysis of Ensemble Learning, Fine-Tuned Models and CNNs
**Student:** Pattiyage Shehan Kavishka (E187041) · ESOFT Metro Campus · HND Computing
**Final ensemble accuracy: 98.20%** on Kaggle Brain Tumor MRI Dataset (2,063 test images)

## Model files

| File | Description | Input size | Accuracy |
|------|-------------|-----------|---------|
| `cnn_standalone.h5` | Custom CNN classifier | 256×256 | 91.18% |
| `InceptionV3.h5` | InceptionV3 classifier | 256×256 | 84.63% |
| `Xception.h5` | Xception classifier | 256×256 | 86.91% |
| `cnn_ensemble.h5` | CNN feature extractor | 128×128 | — |
| `inc_ensemble.h5` | InceptionV3 feature extractor | 128×128 | — |
| `xcp_ensemble.h5` | Xception feature extractor | 128×128 | — |
| `cnn_ensemble_model.pkl` | RF+DT+SVM on CNN features | — | 83.71% |
| `inc_ensemble_model.pkl` | RF+DT+SVM on InceptionV3 features | — | 71.20% |
| `xcp_ensemble_model.pkl` | RF+DT+SVM on Xception features | — | 70.28% |

## Classes
`0=Glioma · 1=Meningioma · 2=No Tumor · 3=Pituitary` (alphabetical order)

## Methodology
- Normalisation: `/255.0` (range 0–1) for all models
- Evaluation: official Kaggle test split (never seen during training)
- Batch size: 32 for all models
- Random seed: 42
- Feature extractor validation: 10% stratified split from training data

## Usage
Set `HF_REPO_ID` and optionally `HF_TOKEN` env vars, then the webapp
auto-downloads models on first startup.
"""


def main():
    if not HF_TOKEN:
        print('ERROR: HF_TOKEN env var not set. Get your token from https://huggingface.co/settings/tokens')
        sys.exit(1)
    if not HF_REPO_ID:
        print('ERROR: HF_REPO_ID env var not set. Example: user/brain-tumor-models')
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print('ERROR: huggingface_hub not installed. Run: pip install huggingface_hub')
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, private=False)
        print(f'Repository: https://huggingface.co/{HF_REPO_ID}')
    except Exception as e:
        print(f'WARNING: Could not create repo: {e}')

    # Upload model card
    print('Writing model card…')
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo='README.md',
        repo_id=HF_REPO_ID,
        commit_message='Update model card',
    )

    # Upload each model file
    missing = []
    for filename in MODEL_FILES:
        local_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(local_path):
            print(f'  SKIP (not found): {filename}')
            missing.append(filename)
            continue
        size_mb = os.path.getsize(local_path) / 1e6
        print(f'  Uploading {filename} ({size_mb:.1f} MB)…', end='', flush=True)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            commit_message=f'Add {filename}',
        )
        print(' done')

    print(f'\nUpload complete — https://huggingface.co/{HF_REPO_ID}')
    if missing:
        print(f'Missing (not uploaded): {", ".join(missing)}')
        print('Run the relevant train_0X.py scripts first.')


if __name__ == '__main__':
    main()
