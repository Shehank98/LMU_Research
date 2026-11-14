"""Re-run the three saved standalone models on the held-out test set and compare with the logged predictions."""
import os, json, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
import tensorflow as tf
X, y, files = load_numpy('Testing', 256)
log = pd.read_csv(os.path.join(ROOT, 'metrics', 'final_per_image_predictions.csv'))
out = {'n_test': int(len(y)), 'labels_match_log': bool((log['true'].values == y).all())}
probs = {}
for nm, fn in [('CNN', 'cnn_model.h5'), ('InceptionV3', 'inceptionv3_model.h5'), ('Xception', 'xception_model.h5')]:
    m = tf.keras.models.load_model(os.path.join(MODELS, fn), compile=False)
    p = m.predict(X, batch_size=32, verbose=0); probs[nm] = p
    agree = float((p.argmax(1) == log['pred_' + nm].values).mean())
    out[nm] = {'agreement_with_logged_predictions': round(agree, 4), 'recomputed_accuracy': round(float((p.argmax(1) == y).mean()), 4),
               'max_abs_conf_diff': round(float(np.abs(p.max(1) - log['conf_' + nm].values).max()), 5)}
    print(nm, out[nm], flush=True)
np.savez_compressed(os.path.join(RES, 'standalone_test_probs.npz'), y=y, files=np.array(files), **probs)
json.dump(out, open(os.path.join(RES, 'verification.json'), 'w'), indent=2); print(out)
