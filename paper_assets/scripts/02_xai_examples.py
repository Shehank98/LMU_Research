"""Per-class correct and misclassified examples: Grad-CAM (3 models), Grad-CAM++, HiResCAM, LIME, SHAP, occlusion."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *; from xai_lib import *
import tensorflow as tf
tf.random.set_seed(42); np.random.seed(42)
pred = pd.read_csv(os.path.join(ROOT, 'metrics', 'final_per_image_predictions.csv'))
d = np.load(os.path.join(RES, 'standalone_test_probs.npz')); files = d['files']
X, y, _ = load_numpy('Testing', 256)
cm = pd.read_csv(os.path.join(ROOT, 'confusion', 'FINAL_majority_vote_confusion_matrix.csv'), index_col=0).values
sel = []
for c in range(4):
    ok = pred[(pred.true == c) & (pred.pred_majority == c)].sort_values('ensemble_confidence', ascending=False)
    sel.append(dict(kind='correct', idx=int(ok.idx.iloc[0]), true=c, pred=c))
for c in range(4):
    row = cm[c].copy(); row[c] = 0
    if row.sum() > 0:
        w = int(row.argmax()); t = c
    else:                                  # no errors for this class: show the most frequent miss INTO it (false negative)
        col = cm[:, c].copy(); col[c] = 0; t, w = int(col.argmax()), c
    bad = pred[(pred.true == t) & (pred.pred_majority == w)].sort_values('ensemble_confidence', ascending=False)
    sel.append(dict(kind='misclassified', idx=int(bad.idx.iloc[0]), true=t, pred=w, n_such_errors=int(cm[t, w])))
for s in sel: s['file'] = str(files[s['idx']]); s['ensemble_confidence'] = round(float(pred.ensemble_confidence[s['idx']]), 4)
idx = np.array([s['idx'] for s in sel]); cls = np.array([s['pred'] for s in sel]); Xs = X[idx]
maps = {}; t0 = time.time()
models = {n: load(n) for n in ['CNN', 'Xception', 'InceptionV3']}
for n, m in models.items():
    c = CamModel(m, TARGET[n])
    maps[f'gradcam_{n}'] = gradcam(c, Xs, cls); maps[f'gradcampp_{n}'] = gradcam_pp(c, Xs, cls); maps[f'hirescam_{n}'] = hirescam(c, Xs, cls)
    for k in range(len(Xs)): s = sel[k]; s.setdefault('model_pred', {})[n] = int(m.predict(Xs[k:k+1], verbose=0).argmax())
print('cams', time.time() - t0, flush=True)
cnn = models['CNN']
maps['occlusion_CNN'] = np.stack([occlusion(cnn, Xs[k], cls[k]) for k in range(len(Xs))]); print('occ', time.time() - t0, flush=True)
maps['occlusion_Xception'] = np.stack([occlusion(models['Xception'], Xs[k], cls[k]) for k in range(len(Xs))]); print('occX', time.time() - t0, flush=True)
maps['lime_CNN'] = np.stack([lime_map(cnn, Xs[k], int(cls[k])) for k in range(len(Xs))]); print('lime', time.time() - t0, flush=True)
maps['lime_Xception'] = np.stack([lime_map(models['Xception'], Xs[k], int(cls[k]), num_samples=400) for k in range(len(Xs))]); print('limeX', time.time() - t0, flush=True)
# SHAP (expected gradients) on the CNN, background = 50 random test images not in the example set
rng = np.random.default_rng(42); bg_idx = rng.choice(np.setdiff1d(np.arange(len(X)), idx), 50, replace=False)
shap_note = 'shap.GradientExplainer'
try:
    import shap
    ex = shap.GradientExplainer(cnn, X[bg_idx])
    sv = ex.shap_values(Xs, nsamples=200, rseed=42)
    sv = np.asarray(sv)
    if sv.ndim == 5 and sv.shape[-1] == 4: sv = np.stack([sv[k, ..., cls[k]] for k in range(len(Xs))])
    elif sv.ndim == 5: sv = np.stack([sv[cls[k], k] for k in range(len(Xs))])
except Exception as e:
    shap_note = 'own expected-gradients implementation (shap failed: %s)' % str(e)[:120]
    sv = []
    for k in range(len(Xs)):
        b = X[rng.choice(bg_idx, 200)]; a = rng.random((200, 1, 1, 1)).astype('float32')
        xi = tf.convert_to_tensor(b + a * (Xs[k][None] - b))
        with tf.GradientTape() as tp:
            tp.watch(xi); o = cnn(xi, training=False)[:, cls[k]]
        g = tp.gradient(o, xi).numpy(); sv.append(((Xs[k][None] - b) * g).mean(0))
    sv = np.stack(sv)
s_pos = np.maximum(np.asarray(sv).sum(-1), 0); s_pos = np.stack([cv2.GaussianBlur(m, (9, 9), 3) for m in s_pos])
maps['shap_CNN'] = s_pos / np.maximum(s_pos.reshape(len(s_pos), -1).max(1), 1e-12)[:, None, None]
print('shap', shap_note, time.time() - t0, flush=True)
np.savez_compressed(os.path.join(RES, 'example_maps.npz'), X=Xs, **maps)
json.dump({'examples': sel, 'shap_implementation': shap_note, 'shap_background': 50, 'lime_samples': {'CNN': 600, 'Xception': 400},
           'occlusion': {'patch': 32, 'stride': 16, 'fill': 0}}, open(os.path.join(RES, 'examples.json'), 'w'), indent=2)
print(json.dumps(sel, indent=1))
