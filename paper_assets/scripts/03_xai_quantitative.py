"""Quantitative XAI: deletion/insertion AUC, cross-model consistency, shortcut (outside-brain) energy, randomisation sanity check."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *; from xai_lib import *
import tensorflow as tf
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
tf.random.set_seed(42); np.random.seed(42); rng = np.random.default_rng(42)
pred = pd.read_csv(os.path.join(ROOT, 'metrics', 'final_per_image_predictions.csv'))
X, y, files = load_numpy('Testing', 256)
# stratified evaluation sample: 30 images per class, plus every ensemble error is oversampled to 40 in total
sample = np.concatenate([rng.choice(np.where(y == c)[0], 30, replace=False) for c in range(4)])
errs = pred.idx[pred.pred_majority != pred.true].values
sample = np.unique(np.concatenate([sample, rng.choice(np.setdiff1d(errs, sample), 40, replace=False)]))
cls = pred.pred_majority.values[sample]; Xs = X[sample]
out = {'n_sample': int(len(sample)), 'n_errors_in_sample': int((pred.pred_majority.values[sample] != y[sample]).sum())}
models = {n: load(n) for n in ['CNN', 'Xception', 'InceptionV3']}
maps = {}
for n, m in models.items():
    c = CamModel(m, TARGET[n])
    for meth, f in [('gradcam', gradcam), ('gradcampp', gradcam_pp), ('hirescam', hirescam)]:
        maps[(n, meth)] = np.concatenate([f(c, Xs[s:s + 16], cls[s:s + 16]) for s in range(0, len(Xs), 16)])
    maps[(n, 'random')] = np.stack([cv2.resize(rng.random((8, 8)).astype('float32'), (256, 256)) for _ in Xs])
np.savez_compressed(os.path.join(RES, 'sample_maps.npz'), sample=sample, **{f'{a}_{b}': v for (a, b), v in maps.items()})
t0 = time.time()
# 1. Deletion / insertion
di_rows = []; di_curves = {}
for n, m in models.items():
    for meth in ['gradcam', 'gradcampp', 'hirescam', 'random']:
        dels, ins, cd, ci = [], [], [], []
        for s in range(0, len(Xs), 16):
            curves, auc = deletion_insertion(m, Xs[s:s + 16], maps[(n, meth)][s:s + 16], cls[s:s + 16])
            dels.append(auc['deletion']); ins.append(auc['insertion']); cd.append(curves['deletion']); ci.append(curves['insertion'])
        dels, ins = np.concatenate(dels), np.concatenate(ins)
        di_curves[f'{n}_{meth}_deletion'] = np.concatenate(cd).mean(0); di_curves[f'{n}_{meth}_insertion'] = np.concatenate(ci).mean(0)
        di_rows.append(dict(model=n, method=meth, deletion_auc=round(float(dels.mean()), 4), deletion_sd=round(float(dels.std()), 4),
                            insertion_auc=round(float(ins.mean()), 4), insertion_sd=round(float(ins.std()), 4)))
        print(di_rows[-1], round(time.time() - t0), flush=True)
pd.DataFrame(di_rows).to_csv(os.path.join(RES, 'deletion_insertion.csv'), index=False)
np.savez(os.path.join(RES, 'deletion_insertion_curves.npz'), **di_curves)
# 2. Cross-model consistency (Grad-CAM): Spearman rank correlation and top-20% IoU, per true class and correct/incorrect
def top_iou(a, b, q=0.8):
    A, B = a >= np.quantile(a, q), b >= np.quantile(b, q); return (A & B).sum() / max((A | B).sum(), 1)
rows = []
pairs = [('CNN', 'Xception'), ('CNN', 'InceptionV3'), ('Xception', 'InceptionV3')]
for k in range(len(Xs)):
    r = dict(idx=int(sample[k]), true=int(y[sample[k]]), pred=int(cls[k]), correct=int(cls[k] == y[sample[k]]))
    for a, b in pairs:
        ma, mb = maps[(a, 'gradcam')][k][::4, ::4].ravel(), maps[(b, 'gradcam')][k][::4, ::4].ravel()
        rho = spearmanr(ma, mb).statistic; r[f'rho_{a}_{b}'] = float(np.nan_to_num(rho)); r[f'iou20_{a}_{b}'] = float(top_iou(ma, mb))
    rows.append(r)
cons = pd.DataFrame(rows); cons['mean_rho'] = cons.filter(like='rho_').mean(1); cons['mean_iou20'] = cons.filter(like='iou20_').mean(1)
cons.to_csv(os.path.join(RES, 'consistency_per_image.csv'), index=False)
out['consistency'] = {'pairwise_mean_rho': {c: round(float(cons[c].mean()), 4) for c in cons.columns if c.startswith('rho_')},
                      'pairwise_mean_iou20': {c: round(float(cons[c].mean()), 4) for c in cons.columns if c.startswith('iou20_')},
                      'by_true_class_mean_rho': {CLASSES[int(k)]: round(float(v), 4) for k, v in cons.groupby('true').mean_rho.mean().items()},
                      'by_true_class_mean_iou20': {CLASSES[int(k)]: round(float(v), 4) for k, v in cons.groupby('true').mean_iou20.mean().items()},
                      'correct_mean_rho': round(float(cons[cons.correct == 1].mean_rho.mean()), 4),
                      'incorrect_mean_rho': round(float(cons[cons.correct == 0].mean_rho.mean()), 4)}
# 3. Shortcut check: share of Grad-CAM energy outside the head and in the outer 12-pixel rim (skull / edge)
sc = []
for k in range(len(Xs)):
    bm = brain_mask(Xs[k]); er = cv2.erode(bm.astype('uint8'), np.ones((25, 25), 'uint8')).astype(bool); rim = bm & ~er
    for n in models:
        m = maps[(n, 'gradcam')][k]; tot = m.sum() + 1e-12
        sc.append(dict(model=n, correct=int(cls[k] == y[sample[k]]), outside=float(m[~bm].sum() / tot), rim=float(m[rim].sum() / tot),
                       rim_area=float(rim.mean()), outside_area=float((~bm).mean())))
sc = pd.DataFrame(sc); sc.to_csv(os.path.join(RES, 'shortcut_per_image.csv'), index=False)
out['shortcut'] = {n: {'mean_energy_outside_head_pct': round(100 * float(g.outside.mean()), 2), 'mean_area_outside_head_pct': round(100 * float(g.outside_area.mean()), 2),
                       'mean_energy_in_rim_pct': round(100 * float(g.rim.mean()), 2), 'mean_area_rim_pct': round(100 * float(g.rim_area.mean()), 2),
                       'images_with_over_50pct_outside_or_rim': int(((g.outside + g.rim) > 0.5).sum())} for n, g in sc.groupby('model')}
# 4. Model-parameter randomisation sanity check (Adebayo et al., 2018), cascading from the top layer
sub = np.arange(0, len(Xs), max(1, len(Xs) // 40))[:40]; Xr, cr = Xs[sub], cls[sub]
def cascade(name, groups):
    m = load(name); base = gradcam(CamModel(m, TARGET[name]), Xr, cr); res = []
    for g in groups:
        for ln in g:
            l = m.get_layer(ln); l.set_weights([rng.normal(0, np.std(w) + 1e-3, w.shape).astype(w.dtype) for w in l.get_weights()])
        r = gradcam(CamModel(m, TARGET[name]), Xr, cr)
        rho = [np.nan_to_num(spearmanr(base[k][::4, ::4].ravel(), r[k][::4, ::4].ravel()).statistic) for k in range(len(Xr))]
        ss = [ssim(base[k], r[k], data_range=1.0) for k in range(len(Xr))]
        res.append(dict(randomised_up_to=g[-1], spearman=round(float(np.mean(rho)), 4), ssim=round(float(np.mean(ss)), 4)))
        print(name, res[-1], flush=True)
    return res
cnn_layers = [l.name for l in load('CNN').layers if l.get_weights()][::-1]
xcp = load('Xception'); xcp_w = [l.name for l in xcp.layers if l.get_weights()][::-1]
xcp_groups = [['dense']] + [[l for l in xcp_w if l.startswith(b)] for b in ['block14', 'block13', 'block12', 'block11']]
out['sanity_check'] = {'n_images': int(len(Xr)), 'CNN': cascade('CNN', [[l] for l in cnn_layers]), 'Xception': cascade('Xception', xcp_groups)}
json.dump(out, open(os.path.join(RES, 'xai_quantitative.json'), 'w'), indent=2); print(json.dumps(out, indent=1))
