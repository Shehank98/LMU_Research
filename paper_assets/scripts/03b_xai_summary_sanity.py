"""Second stage of 03_xai_quantitative.py (split out after the first run ran out of memory during the sanity check):
summaries of cross-model consistency and shortcut checks, plus the cascading-randomisation sanity check on the same sample."""
import os, sys, json, gc, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *; from xai_lib import *
import tensorflow as tf
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
tf.random.set_seed(42); rng = np.random.default_rng(42)
pred = pd.read_csv(os.path.join(ROOT, 'metrics', 'final_per_image_predictions.csv'))
sample = np.load(os.path.join(RES, 'sample_maps.npz'))['sample']
cons = pd.read_csv(os.path.join(RES, 'consistency_per_image.csv')); sc = pd.read_csv(os.path.join(RES, 'shortcut_per_image.csv'))
out = {'n_sample': int(len(sample)), 'n_errors_in_sample': int((pred.pred_majority.values[sample] != pred.true.values[sample]).sum())}
out['consistency'] = {'pairwise_mean_rho': {c: round(float(cons[c].mean()), 4) for c in cons.columns if c.startswith('rho_')},
                      'pairwise_mean_iou20': {c: round(float(cons[c].mean()), 4) for c in cons.columns if c.startswith('iou20_')},
                      'by_true_class_mean_rho': {CLASSES[int(k)]: round(float(v), 4) for k, v in cons.groupby('true').mean_rho.mean().items()},
                      'by_true_class_mean_iou20': {CLASSES[int(k)]: round(float(v), 4) for k, v in cons.groupby('true').mean_iou20.mean().items()},
                      'correct_mean_rho': round(float(cons[cons.correct == 1].mean_rho.mean()), 4), 'incorrect_mean_rho': round(float(cons[cons.correct == 0].mean_rho.mean()), 4),
                      'correct_mean_iou20': round(float(cons[cons.correct == 1].mean_iou20.mean()), 4), 'incorrect_mean_iou20': round(float(cons[cons.correct == 0].mean_iou20.mean()), 4)}
from scipy.stats import mannwhitneyu
out['consistency']['mannwhitney_p_rho'] = float(mannwhitneyu(cons[cons.correct == 1].mean_rho, cons[cons.correct == 0].mean_rho, alternative='greater').pvalue)
out['shortcut'] = {n: {'mean_energy_outside_head_pct': round(100 * float(g.outside.mean()), 2), 'mean_area_outside_head_pct': round(100 * float(g.outside_area.mean()), 2),
                       'mean_energy_in_rim_pct': round(100 * float(g.rim.mean()), 2), 'mean_area_rim_pct': round(100 * float(g.rim_area.mean()), 2),
                       'images_with_over_50pct_outside_or_rim': int(((g.outside + g.rim) > 0.5).sum()),
                       'outside_or_rim_correct_pct': round(100 * float((g[g.correct == 1].outside + g[g.correct == 1].rim).mean()), 2),
                       'outside_or_rim_incorrect_pct': round(100 * float((g[g.correct == 0].outside + g[g.correct == 0].rim).mean()), 2)} for n, g in sc.groupby('model')}
X, y, _ = load_numpy('Testing', 256)
Xs, cls = X[sample], pred.pred_majority.values[sample]; del X; gc.collect()
sub = np.arange(0, len(Xs), max(1, len(Xs) // 40))[:40]; Xr, cr = Xs[sub], cls[sub]
def cascade(name, groups):
    m = load(name); base = gradcam(CamModel(m, TARGET[name]), Xr, cr); res = []
    for g in groups:
        for ln in g:
            l = m.get_layer(ln); l.set_weights([rng.normal(0, np.std(w) + 1e-3, w.shape).astype(w.dtype) for w in l.get_weights()])
        r = np.concatenate([gradcam(CamModel(m, TARGET[name]), Xr[s:s + 10], cr[s:s + 10]) for s in range(0, len(Xr), 10)])
        rho = [np.nan_to_num(spearmanr(base[k][::4, ::4].ravel(), r[k][::4, ::4].ravel()).statistic) for k in range(len(Xr))]
        ss = [ssim(base[k], r[k], data_range=1.0) for k in range(len(Xr))]
        res.append(dict(randomised_up_to=g[-1], spearman=round(float(np.mean(rho)), 4), ssim=round(float(np.mean(ss)), 4))); print(name, res[-1], flush=True)
    del m; tf.keras.backend.clear_session(); gc.collect(); return res
cnn_layers = [l.name for l in load('CNN').layers if l.get_weights()][::-1]
xcp_w = [l.name for l in load('Xception').layers if l.get_weights()][::-1]; tf.keras.backend.clear_session(); gc.collect()
xcp_groups = [['dense']] + [[l for l in xcp_w if l.startswith(b)] for b in ['block14', 'block13', 'block12', 'block11']]
out['sanity_check'] = {'n_images': int(len(Xr)), 'CNN': cascade('CNN', [[l] for l in cnn_layers]), 'Xception': cascade('Xception', xcp_groups)}
json.dump(out, open(os.path.join(RES, 'xai_quantitative.json'), 'w'), indent=2); print(json.dumps(out, indent=1))
