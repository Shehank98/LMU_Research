#!/usr/bin/env python3
"""
Patch standardised notebooks and XAI notebooks to:
1. Add HF_REPO_ID to constants cells (notebooks 02, 04, 06, 08)
2. Add HF download-before-load to model loading cells (notebooks 02, 04, 06, 08)
3. Fix broken _dl() function in XAI notebooks 09, 10 (use shutil.copy from cache)
"""
import json, os, re

NB_DIR  = "/home/user/LMU_Research/standardised_notebooks"
XAI_DIR = "/home/user/LMU_Research/XAI_notebook"

HF_REPO = "shehank98/brain-tumor-mri-models"

# ---------------------------------------------------------------------------
# Helper: robust download snippet (replaces broken local_dir approach)
# ---------------------------------------------------------------------------
DOWNLOAD_FN = '''\
def _ensure_model(filename):
    """Download a model file from HuggingFace if not already in SAVED_MODELS_DIR."""
    import shutil
    dest = os.path.join(SAVED_MODELS_DIR, filename)
    if os.path.exists(dest):
        return dest
    print(f"  Downloading from HuggingFace: {filename} ...", end="", flush=True)
    try:
        from huggingface_hub import hf_hub_download, login as hf_login
        if HF_TOKEN:
            hf_login(token=HF_TOKEN, add_to_git_credential=False)
        # hf_hub_download returns a path in the HF cache; copy it to our dir
        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"models/{filename}",
            token=HF_TOKEN or None,
        )
        shutil.copy2(cached, dest)
        print(f" done ({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find or download '{filename}': {e}\\n"
            "Run the prerequisite notebook first and ensure it uploads to HuggingFace."
        )
'''

def load_nb(path):
    with open(path) as f:
        return json.load(f)

def save_nb(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Saved: {path}")

def get_src(cell):
    s = cell['source']
    return s if isinstance(s, str) else ''.join(s)

def set_src(cell, src):
    cell['source'] = src

# ---------------------------------------------------------------------------
# 1. Patch standardised notebooks 02, 04, 06, 08
# ---------------------------------------------------------------------------
PATCHES = {
    "02_CNN_Ensemble.ipynb": {
        "model_file": "cnn_model.h5",
        "load_pattern": "cnn_model_path = SAVED_MODELS_DIR + 'cnn_model.h5'",
    },
    "04_InceptionV3_Ensemble.ipynb": {
        "model_file": "inceptionv3_model.h5",
        "load_pattern": "inc_model_path = SAVED_MODELS_DIR + 'inceptionv3_model.h5'",
    },
    "06_Xception_Ensemble.ipynb": {
        "model_file": "xception_model.h5",
        "load_pattern": "xcp_model_path = SAVED_MODELS_DIR + 'xception_model.h5'",
    },
}

NB08_MODELS = [
    "cnn_model.h5", "inceptionv3_model.h5", "xception_model.h5",
    "cnn_ensemble_model.pkl", "inceptionv3_ensemble_model.pkl", "xception_ensemble_model.pkl",
]

for nb_name, info in PATCHES.items():
    path = os.path.join(NB_DIR, nb_name)
    nb   = load_nb(path)
    cells = nb['cells']

    # --- Step 1: add HF_REPO_ID to constants cell (code cell index 2) ---
    code_idx = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']
    const_cell_idx = code_idx[2]  # 3rd code cell = constants
    src = get_src(cells[const_cell_idx])
    if 'HF_REPO_ID' not in src:
        src = src.rstrip('\n') + f'\nHF_REPO_ID       = "{HF_REPO}"\n'
        set_src(cells[const_cell_idx], src)
        print(f"  {nb_name}: added HF_REPO_ID to constants cell")

    # --- Step 2: patch the model loading cell (code cell index 5) ---
    load_cell_idx = code_idx[5]   # 6th code cell = model loading
    src = get_src(cells[load_cell_idx])
    if '_ensure_model' not in src:
        # Prepend the download function + call before the load_model line
        download_call = f"_ensure_model('{info['model_file']}')\n\n"
        src = DOWNLOAD_FN + "\n" + download_call + src
        set_src(cells[load_cell_idx], src)
        print(f"  {nb_name}: added _ensure_model() to model loading cell")

    save_nb(nb, path)
    print(f"  Patched: {nb_name}")

# --- Patch notebook 08 separately (loads many models) ---
path08 = os.path.join(NB_DIR, "08_AllModelsCombined.ipynb")
nb08   = load_nb(path08)
cells08 = nb08['cells']
code_idx08 = [i for i, c in enumerate(cells08) if c['cell_type'] == 'code']

# Add HF_REPO_ID to constants cell
const_idx08 = code_idx08[2]
src = get_src(cells08[const_idx08])
if 'HF_REPO_ID' not in src:
    src = src.rstrip('\n') + f'\nHF_REPO_ID       = "{HF_REPO}"\n'
    set_src(cells08[const_idx08], src)
    print("  08: added HF_REPO_ID to constants cell")

# Patch model loading cell
load_idx08 = code_idx08[5]
src = get_src(cells08[load_idx08])
if '_ensure_model' not in src:
    calls = ''.join(f"_ensure_model('{f}')\n" for f in NB08_MODELS)
    src = DOWNLOAD_FN + "\n# Download all required models from HuggingFace\n" + calls + "\n" + src
    set_src(cells08[load_idx08], src)
    print("  08: added _ensure_model() for all 6 models")

save_nb(nb08, path08)
print("  Patched: 08_AllModelsCombined.ipynb")

# ---------------------------------------------------------------------------
# 2. Fix XAI notebooks 09 and 10: replace broken _dl() with correct version
# ---------------------------------------------------------------------------
CORRECT_DL = '''\
def _dl(filename):
    """Download a model file from HuggingFace to SAVED_MODELS_DIR."""
    import shutil
    dest = os.path.join(SAVED_MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  local: {filename}"); return dest
    print(f"  downloading: {filename} ...", end="", flush=True)
    try:
        from huggingface_hub import hf_hub_download, login as hf_login
        if HF_TOKEN: hf_login(token=HF_TOKEN, add_to_git_credential=False)
        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"models/{filename}",
            token=HF_TOKEN or None,
        )
        shutil.copy2(cached, dest)
        print(f" done ({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest
    except Exception as e:
        print(f"\\n  WARN: {filename}: {e}"); return None
'''

# Pattern that matches the broken _dl function (uses local_dir=)
BROKEN_DL_PAT = re.compile(
    r'def _dl\(filename\):.*?(?=\n(?:def |\w|\Z))',
    re.DOTALL
)

for xai_nb in ["consensus_gradcam.ipynb", "confidence_voting.ipynb"]:
    path = os.path.join(XAI_DIR, xai_nb)
    nb   = load_nb(path)
    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = get_src(cell)
        if 'def _dl(filename)' in src and 'local_dir' in src:
            # Replace the broken _dl function
            new_src = BROKEN_DL_PAT.sub(CORRECT_DL.rstrip('\n'), src, count=1)
            if new_src != src:
                set_src(cell, new_src)
                changed = True
                print(f"  {xai_nb}: fixed _dl() function")
                break
    if changed:
        save_nb(nb, path)
        print(f"  Patched: {xai_nb}")
    else:
        print(f"  {xai_nb}: no change needed (already correct or pattern not found)")

print("\nAll notebook patches applied.")
