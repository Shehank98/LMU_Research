"""SHAP TreeExplainer on the Random Forest that classifies the 256 CNN dense-layer features (feature-level ensemble stage)."""
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); from common import *
import tensorflow as tf, joblib, shap
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False})
X, y, _ = load_numpy('Testing', 128)
ext = tf.keras.models.load_model(os.path.join(MODELS, 'cnn_ext.h5'), compile=False)
h = tf.convert_to_tensor(X)
feats = []
for s in range(0, len(X), 64):
    t = tf.convert_to_tensor(X[s:s + 64])
    for l in ext.layers:
        t = l(t, training=False)
        if l.name == 'dense': break
    feats.append(t.numpy())
F = np.concatenate(feats)
rf = joblib.load(os.path.join(MODELS, 'cnn_feat_rf.pkl'))
acc = float((rf.predict(F) == y).mean()); print('RF acc recomputed', acc)
rng = np.random.default_rng(42); idx = np.concatenate([rng.choice(np.where(y == c)[0], 50, replace=False) for c in range(4)])
ex = shap.TreeExplainer(rf); sv = np.asarray(ex.shap_values(F[idx]))           # (n, 256, 4) in recent shap
if sv.shape[0] == 4: sv = np.moveaxis(sv, 0, -1)
imp = np.abs(sv).mean(0)                                                          # (256 features, 4 classes)
top = np.argsort(-imp.sum(1))[:15]
active = float((F > 0).mean())
f, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
left = np.zeros(len(top))
for c in range(4):
    axs[0].barh([f'unit {k}' for k in top][::-1], imp[top, c][::-1], left=left[::-1], color=['#3B6FB6', '#E07B39', '#4E9F6E', '#B5487F'][c], label=LABELS[c]); left += imp[top, c]
axs[0].set_xlabel('mean |SHAP value| (stacked over classes)'); axs[0].legend(frameon=False, fontsize=7); axs[0].set_title('Top 15 CNN dense-layer features for the RF')
cum = np.cumsum(np.sort(imp.sum(1))[::-1]) / imp.sum()
axs[1].plot(np.arange(1, 257), cum, color='#3B6FB6'); axs[1].axhline(.8, ls=':', color='grey')
n80 = int(np.searchsorted(cum, .8) + 1); axs[1].axvline(n80, ls=':', color='grey'); axs[1].set_xlabel('Number of features (ranked)'); axs[1].set_ylabel('Cumulative share of |SHAP|')
axs[1].set_title(f'{n80} of 256 features carry 80% of attribution')
f.tight_layout(); f.savefig(os.path.join(FIG, 'fig_shap_rf.png')); plt.close(f)
out = {'rf_accuracy_recomputed_pct': round(100 * acc, 2), 'n_explained': int(len(idx)), 'features_for_80pct': n80,
       'dead_or_inactive_features': int(((F > 0).mean(0) == 0).sum()), 'mean_active_fraction': round(active, 4),
       'top5_units': [int(k) for k in top[:5]], 'top5_share_pct': round(100 * float(imp[top[:5]].sum() / imp.sum()), 2),
       'class_specific_top_unit': {LABELS[c]: int(np.argmax(imp[:, c])) for c in range(4)}}
pd.DataFrame(imp, columns=LABELS).rename_axis('unit').to_csv(os.path.join(RES, 'shap_rf_importance.csv'))
json.dump(out, open(os.path.join(RES, 'shap_rf.json'), 'w'), indent=1); print(out)
