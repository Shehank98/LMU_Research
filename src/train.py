"""Train one deep model and save its predictions and dense-layer features.

usage: python train.py cnn | inceptionv3 | xception
"""

import os, sys, random
import numpy as np
import pandas as pd
import tensorflow as tf

import config as C
from data import load_split, train_val_split
from models import build_cnn, build_transfer, unfreeze_top
from metrics import log_result, save_probs

random.seed(C.SEED)
np.random.seed(C.SEED)
tf.random.set_seed(C.SEED)


def as_dataset(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=C.SEED)
    return ds.batch(C.BATCH).map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b)).prefetch(tf.data.AUTOTUNE)


def fit(model, train, val, epochs, lr, history, tag, patience=C.PATIENCE, lr_patience=2, min_lr=1e-6):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=lr_patience, min_lr=min_lr),
    ]
    h = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks, verbose=2)
    df = pd.DataFrame(h.history).assign(stage=tag)
    return pd.concat([history, df], ignore_index=True)


def main(kind):
    X, y, _ = load_split("Training")
    X_tr, X_val, y_tr, y_val = train_val_split(X, y)
    X_te, y_te, _ = load_split("Testing")
    train, val = as_dataset(X_tr, y_tr, shuffle=True), as_dataset(X_val, y_val)

    history = pd.DataFrame()
    if kind == "cnn":
        model = build_cnn()
        history = fit(model, train, val, C.CNN_EPOCHS, 1e-3, history, "train", C.CNN_PATIENCE, C.CNN_LR_PATIENCE, 1e-5)
    else:
        model = build_transfer(kind)
        history = fit(model, train, val, C.HEAD_EPOCHS, 1e-3, history, "head")  # 1. new head only
        unfreeze_top(model, C.UNFREEZE_FROM[kind])
        history = fit(
            model, train, val, C.FINETUNE_EPOCHS, 1e-4, history, "finetune"
        )  # 2. fine-tune top blocks

    os.makedirs(os.path.join(C.OUT_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(C.OUT_DIR, "history"), exist_ok=True)
    model.save(os.path.join(C.OUT_DIR, "models", f"{kind}.keras"))
    history.index.name = "epoch"
    history.to_csv(os.path.join(C.OUT_DIR, "history", f"{kind}.csv"))

    p_val = model.predict(as_dataset(X_val, y_val), verbose=0)
    p_te = model.predict(as_dataset(X_te, y_te), verbose=0)
    log_result(kind, y_te, p_te.argmax(1), note=f"val acc {np.mean(p_val.argmax(1) == y_val):.4f}")

    extractor = tf.keras.Model(model.inputs, model.get_layer("features").output)
    feats = {
        name: extractor.predict(as_dataset(a, b), verbose=0)
        for name, a, b in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)]
    }
    save_probs(
        kind,
        val=p_val,
        test=p_te,
        y_val=y_val,
        y_test=y_te,
        f_train=feats["train"],
        f_val=feats["val"],
        f_test=feats["test"],
        y_train=y_tr,
    )


if __name__ == "__main__":
    main(sys.argv[1])
