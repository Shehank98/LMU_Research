"""Re-run the three saved standalone networks on the 1,600 test images.

Checks that the saved models reproduce the predictions logged by the training notebook, and saves the class
probabilities used for the ROC and precision-recall curves (Section 4.6).
Output: outputs/standalone_test_probs.npz, outputs/verification.json
"""

import os, json
import numpy as np
import pandas as pd
import tensorflow as tf

from settings import MODELS, OUT, PRED_CSV, load_numpy

X, y, files = load_numpy("Testing", 256)
log = pd.read_csv(PRED_CSV)
out = {"n_test": int(len(y)), "labels_match_log": bool((log["true"].values == y).all())}
probs = {}
for name, fn in [("CNN", "cnn_model.h5"), ("InceptionV3", "inceptionv3_model.h5"), ("Xception", "xception_model.h5")]:
    model = tf.keras.models.load_model(os.path.join(MODELS, fn), compile=False)
    p = model.predict(X, batch_size=32, verbose=0)
    probs[name] = p
    out[name] = {
        "agreement_with_logged_predictions": round(float((p.argmax(1) == log["pred_" + name].values).mean()), 4),
        "accuracy": round(float((p.argmax(1) == y).mean()), 4),
    }
    print(name, out[name], flush=True)
np.savez_compressed(os.path.join(OUT, "standalone_test_probs.npz"), y=y, files=np.array(files), **probs)
json.dump(out, open(os.path.join(OUT, "verification.json"), "w"), indent=2)
