"""Project settings shared by every script."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("MRI_DATA", os.path.join(ROOT, "MRI_DATASET"))
OUT_DIR = os.environ.get("RUN_DIR", os.path.join(ROOT, "runs", "v2"))

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LABELS = ["Glioma", "Meningioma", "No tumour", "Pituitary"]

IMG_SIZE = 224
SEED = 42
VAL_FRACTION = 0.1
BATCH = 32

CNN_EPOCHS = 45
CNN_PATIENCE = 8         # the CNN's validation loss is noisy, so it gets more patience
CNN_LR_PATIENCE = 4
HEAD_EPOCHS = 3  # transfer models: train the new head first
FINETUNE_EPOCHS = 12  # then unfreeze the top of the backbone
PATIENCE = 5

# first layer that is unfrozen during fine-tuning
UNFREEZE_FROM = {"xception": "block10_sepconv1_act", "inceptionv3": "mixed7"}
