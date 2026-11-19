"""Saving results as data files (CSV / JSON) so every number in the report can be traced."""

import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from settings import OUT_DIR, CLASSES


def _path(*parts):
    p = os.path.join(OUT_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def log_result(name, y_true, y_pred, note=""):
    """Write the confusion matrix and classification report, and add one row to results.csv."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).rename_axis("actual_vs_predicted").to_csv(
        _path("confusion", f"{name}.csv")
    )
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    json.dump(report, open(_path("metrics", f"{name}_report.json"), "w"), indent=2)
    f1 = f1_score(y_true, y_pred, average=None, labels=range(len(CLASSES)))
    row = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1.mean(),
        **{f"f1_{c}": v for c, v in zip(CLASSES, f1)},
        "n_test": len(y_true),
        "note": note,
    }
    table = _path("metrics", "results.csv")
    df = pd.read_csv(table) if os.path.exists(table) else pd.DataFrame()
    df = pd.concat([df[df.model != name] if len(df) else df, pd.DataFrame([row])], ignore_index=True)
    df.round(4).to_csv(table, index=False)
    print(f'{name}: accuracy {100 * row["accuracy"]:.2f}%  macro-F1 {row["f1_macro"]:.3f}', flush=True)
    return row


def save_probs(name, **arrays):
    np.savez_compressed(_path("probs", f"{name}.npz"), **arrays)


def load_probs(name):
    return np.load(_path("probs", f"{name}.npz"))
