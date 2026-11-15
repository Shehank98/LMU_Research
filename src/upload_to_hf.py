"""Upload the improved models to Hugging Face so the web app can use them (MODEL_SET=improved).

usage:  HF_TOKEN=<write token> python upload_to_hf.py
"""

import os
from huggingface_hub import HfApi

import config as C

REPO = os.environ.get("HF_REPO_ID", "shehank98/brain-tumor-mri-models")


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    models_dir = os.path.join(C.OUT_DIR, "models")
    api.upload_folder(
        repo_id=REPO, folder_path=models_dir, path_in_repo="models/v2", allow_patterns=["*.keras", "*.pkl"]
    )
    for sub in ("metrics", "confusion", "history"):
        api.upload_folder(
            repo_id=REPO, folder_path=os.path.join(C.OUT_DIR, sub), path_in_repo=f"results_v2/{sub}"
        )
    print("uploaded to", REPO)


if __name__ == "__main__":
    main()
