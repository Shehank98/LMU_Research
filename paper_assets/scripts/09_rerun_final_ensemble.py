"""Independent re-run of the full six-predictor ensemble on the 1,600 held-out test images, using the saved models only."""
import os, sys, json, numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(__file__)); from common import *
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
X128, y, _ = load_numpy('Testing', 128)
P = np.load(os.path.join(RES, 'standalone_test_probs.npz'))
probs = {k: P[k] for k in ['CNN', 'InceptionV3', 'Xception']}
def feats(model, layer, X, batch=64):
    if isinstance(model, tf.keras.Sequential):
        out = []
        for s in range(0, len(X), batch):
            h = tf.convert_to_tensor(X[s:s + batch])
            for l in model.layers:
                h = l(h, training=False)
                if l.name == layer: break
            out.append(h.numpy())
        return np.concatenate(out)
    return tf.keras.Model(model.inputs, model.get_layer(layer).output).predict(X, batch_size=batch, verbose=0)
for fn, layer, tag, nm in [('cnn_ext.h5', 'dense', 'cnn_feat', 'CNN_ensemble'), ('inceptionv3_ext.h5', 'dense_2', 'inceptionv3_feat', 'InceptionV3_ensemble'),
                           ('xception_ext.h5', 'dense_3', 'xception_feat', 'Xception_ensemble')]:
    F = feats(tf.keras.models.load_model(os.path.join(MODELS, fn), compile=False), layer, X128)
    clfs = [joblib.load(os.path.join(MODELS, f'{tag}_{k}.pkl')) for k in ['rf', 'dt', 'svm']]
    probs[nm] = sum(c.predict_proba(F) for c in clfs) / 3; print(nm, accuracy_score(y, probs[nm].argmax(1)), flush=True)
names = ['CNN', 'InceptionV3', 'Xception', 'CNN_ensemble', 'InceptionV3_ensemble', 'Xception_ensemble']
S = np.stack([probs[k] for k in names]); summed = S.sum(0)
conf = summed.argmax(1); votes = S.argmax(2)
maj = np.array([np.bincount(votes[:, i], minlength=4).argmax() for i in range(len(y))])
log = pd.read_csv(os.path.join(ROOT, 'metrics', 'final_per_image_predictions.csv'))
out = {'n_test': int(len(y)),
       'majority': {'accuracy_pct': round(100 * accuracy_score(y, maj), 2), 'macro_f1': round(f1_score(y, maj, average='macro'), 4), 'agreement_with_log': round(float((maj == log.pred_majority.values).mean()), 4)},
       'confidence_weighted': {'accuracy_pct': round(100 * accuracy_score(y, conf), 2), 'macro_f1': round(f1_score(y, conf, average='macro'), 4), 'agreement_with_log': round(float((conf == log.pred_confidence_weighted.values).mean()), 4)},
       'members_accuracy_pct': {k: round(100 * accuracy_score(y, probs[k].argmax(1)), 2) for k in names},
       'members_agreement_with_log': {k: round(float((probs[k].argmax(1) == log['pred_' + k].values).mean()), 4) for k in names}}
json.dump(out, open(os.path.join(RES, 'final_ensemble_rerun.json'), 'w'), indent=2); print(json.dumps(out, indent=1))
