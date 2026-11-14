"""Figures for the quantitative XAI results (deletion/insertion, sanity check, consistency, shortcut check)."""
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); from common import *; from xai_lib import *
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'gradcam': '#3B6FB6', 'gradcampp': '#E07B39', 'hirescam': '#4E9F6E', 'random': '#9A9A9A'}
NAME = {'gradcam': 'Grad-CAM', 'gradcampp': 'Grad-CAM++', 'hirescam': 'HiResCAM', 'random': 'Random'}
cur = np.load(os.path.join(RES, 'deletion_insertion_curves.npz')); di = pd.read_csv(os.path.join(RES, 'deletion_insertion.csv'))
f, axs = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey=True); x = np.linspace(0, 1, 21)
for j, m in enumerate(['CNN', 'Xception', 'InceptionV3']):
    for i, kind in enumerate(['deletion', 'insertion']):
        for meth in ['gradcam', 'gradcampp', 'hirescam', 'random']:
            a = di[(di.model == m) & (di.method == meth)][kind + '_auc'].values[0]
            axs[i, j].plot(x, cur[f'{m}_{meth}_{kind}'], color=COL[meth], lw=1.2, ls='--' if meth == 'random' else '-', label=f'{NAME[meth]} ({a:.3f})')
        axs[i, j].legend(frameon=False, fontsize=6.2, title=f'{kind.title()} AUC', title_fontsize=6.5)
        if j == 0: axs[i, j].set_ylabel('Probability of predicted class')
        axs[i, j].set_xlabel('Fraction of pixels ' + ('removed' if kind == 'deletion' else 'revealed'))
    axs[0, j].set_title(m)
f.tight_layout(); f.savefig(os.path.join(FIG, 'fig_deletion_insertion.png')); plt.close(f)
q = json.load(open(os.path.join(RES, 'xai_quantitative.json')))
# sanity check: similarity curves + visual cascade for one image
f = plt.figure(figsize=(7.2, 3.9)); gs = f.add_gridspec(2, 7, height_ratios=[1.05, 1])
ax = f.add_subplot(gs[0, :3])
for m, c in [('CNN', '#3B6FB6'), ('Xception', '#B5487F')]:
    s = q['sanity_check'][m]; ax.plot(range(1, len(s) + 1), [d['spearman'] for d in s], 'o-', color=c, label=f'{m} Spearman', ms=3)
    ax.plot(range(1, len(s) + 1), [d['ssim'] for d in s], 's:', color=c, label=f'{m} SSIM', ms=3)
ax.axhline(0, color='grey', lw=.6); ax.set_xlabel('Layer groups randomised (from the top)'); ax.set_ylabel('Similarity to original map'); ax.legend(frameon=False, fontsize=6.3)
ax.set_title('Similarity of Grad-CAM after randomisation')
E = np.load(os.path.join(RES, 'example_maps.npz')); x0 = E['X'][0]; ex = json.load(open(os.path.join(RES, 'examples.json')))['examples'][0]
m = load('CNN'); rng = np.random.default_rng(42); layers_ = [l.name for l in m.layers if l.get_weights()][::-1]
maps = [gradcam(CamModel(m, TARGET['CNN']), x0[None], [ex['pred']])[0]]; titles = ['Original']
for ln in layers_:
    l = m.get_layer(ln); l.set_weights([rng.normal(0, np.std(w) + 1e-3, w.shape).astype(w.dtype) for w in l.get_weights()])
    maps.append(gradcam(CamModel(m, TARGET['CNN']), x0[None], [ex['pred']])[0]); titles.append('to ' + ln)
for k in range(7):
    a = f.add_subplot(gs[1, k]); a.imshow(x0); a.imshow(maps[k], cmap='jet', alpha=.45, vmin=0, vmax=1); a.set_title(titles[k], fontsize=6); a.axis('off')
f.tight_layout(); f.savefig(os.path.join(FIG, 'fig_sanity_check.png')); plt.close(f)
# consistency and shortcut
cons = pd.read_csv(os.path.join(RES, 'consistency_per_image.csv')); sc = pd.read_csv(os.path.join(RES, 'shortcut_per_image.csv'))
f, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
pairs = [c for c in cons.columns if c.startswith('rho_')]
g = cons.groupby('true')[pairs].mean(); w = .25
for i, p in enumerate(pairs): axs[0].bar(np.arange(4) + (i - 1) * w, g[p], w, label=p[4:].replace('_', ' vs '), color=['#3B6FB6', '#E07B39', '#4E9F6E'][i])
axs[0].set_xticks(range(4), ['G', 'M', 'N', 'P']); axs[0].set_ylabel('Mean Spearman ρ between maps'); axs[0].legend(frameon=False, fontsize=5.8); axs[0].set_title('Agreement by true class')
axs[1].boxplot([cons[cons.correct == 1].mean_rho, cons[cons.correct == 0].mean_rho], tick_labels=['Correct', 'Misclassified'], showfliers=False, patch_artist=True, boxprops=dict(facecolor='#DCE6F2'))
axs[1].set_title('Mean pairwise ρ (sample)')
s = sc.groupby('model')[['outside', 'rim', 'outside_area', 'rim_area']].mean().loc[['CNN', 'Xception', 'InceptionV3']]
xx = np.arange(3); axs[2].bar(xx - .2, 100 * (s.outside + s.rim), .4, color='#B5487F', label='Share of Grad-CAM mass')
axs[2].bar(xx + .2, 100 * (s.outside_area + s.rim_area), .4, color='#C9C9C9', label='Share of image area')
axs[2].set_xticks(xx, s.index); axs[2].set_ylabel('% outside head or in skull rim'); axs[2].legend(frameon=False, fontsize=6); axs[2].set_title('Attention outside the brain')
f.tight_layout(); f.savefig(os.path.join(FIG, 'fig_consistency_shortcut.png')); plt.close(f)
print('ok')
