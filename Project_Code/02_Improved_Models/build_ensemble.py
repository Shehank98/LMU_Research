"""Combine the six predictors by majority vote and by confidence-weighted (summed probability) vote."""

import numpy as np
import pandas as pd

import settings as C
from evaluation_metrics import log_result, load_probs, _path

MEMBERS = [
    "cnn",
    "inceptionv3",
    "xception",
    "cnn_feat_softvote",
    "inceptionv3_feat_softvote",
    "xception_feat_softvote",
]


def majority(probs):
    votes = probs.argmax(axis=2)  # (models, images)
    return np.array([np.bincount(v, minlength=len(C.CLASSES)).argmax() for v in votes.T])


def main():
    data = [load_probs(m) for m in MEMBERS]
    y_test, y_val = data[0]["y_test"], data[0]["y_val"]
    P_test = np.stack([d["test"] for d in data])
    P_val = np.stack([d["val"] for d in data])
    summed = P_test.sum(axis=0)
    log_result(
        "final_majority", y_test, majority(P_test), note=f"val acc {np.mean(majority(P_val) == y_val):.4f}"
    )
    log_result(
        "final_confidence_weighted",
        y_test,
        summed.argmax(1),
        note=f"val acc {np.mean(P_val.sum(0).argmax(1) == y_val):.4f}",
    )
    per_image = pd.DataFrame(
        {
            "idx": np.arange(len(y_test)),
            "true": y_test,
            "pred_majority": majority(P_test),
            "pred_confidence_weighted": summed.argmax(1),
            "ensemble_confidence": summed.max(1) / summed.sum(1),
        }
    )
    for m, p in zip(MEMBERS, P_test):
        per_image[f"pred_{m}"] = p.argmax(1)
        per_image[f"conf_{m}"] = p.max(1)
    per_image.to_csv(_path("metrics", "per_image_predictions.csv"), index=False)
    np.savez_compressed(_path("probs", "final_all.npz"), P=P_test, y=y_test, names=np.array(MEMBERS))


if __name__ == "__main__":
    main()
