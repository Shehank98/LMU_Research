import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); from common import *
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
E = np.load(os.path.join(RES, 'example_maps.npz')); ex = json.load(open(os.path.join(RES, 'examples.json')))['examples']
X = E['X']
def panel(rows, cols, name):
    f, axs = plt.subplots(len(rows), len(cols) + 1, figsize=(1.0 * (len(cols) + 1), 1.08 * len(rows)))
    for r, k in enumerate(rows):
        e = ex[k]; axs[r, 0].imshow(X[k]); t = f"True: {LABELS[e['true']]}\nPred: {LABELS[e['pred']]}"
        axs[r, 0].set_ylabel(t, fontsize=6.5, rotation=0, ha='right', va='center', labelpad=4)
        for c, (key, title) in enumerate(cols, 1):
            axs[r, c].imshow(X[k]); axs[r, c].imshow(E[key][k], cmap='jet', alpha=.45, vmin=0, vmax=1)
            if r == 0: axs[r, c].set_title(title, fontsize=6.5)
        if r == 0: axs[r, 0].set_title('MRI', fontsize=6.5)
    for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
    plt.subplots_adjust(wspace=.04, hspace=.06); f.savefig(os.path.join(FIG, name)); plt.close(f)
cols = [('gradcam_CNN', 'Grad-CAM\nCNN'), ('gradcam_Xception', 'Grad-CAM\nXception'), ('gradcam_InceptionV3', 'Grad-CAM\nInceptionV3'),
        ('gradcampp_CNN', 'Grad-CAM++\nCNN'), ('lime_CNN', 'LIME\nCNN'), ('shap_CNN', 'SHAP\nCNN'), ('occlusion_CNN', 'Occlusion\nCNN')]
panel([0, 1, 2, 3], cols, 'fig_xai_correct.png'); panel([4, 5, 6, 7], cols, 'fig_xai_misclassified.png')
cols2 = [('gradcam_Xception', 'Grad-CAM'), ('gradcampp_Xception', 'Grad-CAM++'), ('hirescam_Xception', 'HiResCAM'), ('lime_Xception', 'LIME'), ('occlusion_Xception', 'Occlusion')]
panel([0, 4, 2, 6], cols2, 'fig_xai_methods_xception.png')
cols3 = [('gradcam_CNN', 'Grad-CAM'), ('gradcampp_CNN', 'Grad-CAM++'), ('hirescam_CNN', 'HiResCAM'), ('lime_CNN', 'LIME'), ('shap_CNN', 'SHAP'), ('occlusion_CNN', 'Occlusion')]
panel([0, 4, 2, 6], cols3, 'fig_xai_methods_cnn.png')
# simple agreement numbers between methods on these 8 examples (Spearman), saved for the text
from scipy.stats import spearmanr
keys = [k for k in E.files if k != 'X']; agree = {}
for a in ['gradcam_CNN', 'gradcampp_CNN', 'hirescam_CNN', 'lime_CNN', 'shap_CNN', 'occlusion_CNN']:
    for b in ['occlusion_CNN']:
        if a != b: agree[f'{a}~{b}'] = round(float(np.mean([np.nan_to_num(spearmanr(E[a][k][::4, ::4].ravel(), E[b][k][::4, ::4].ravel()).statistic) for k in range(8)])), 3)
for a in ['gradcam_Xception', 'gradcampp_Xception', 'hirescam_Xception', 'lime_Xception']:
    agree[f'{a}~occlusion_Xception'] = round(float(np.mean([np.nan_to_num(spearmanr(E[a][k][::4, ::4].ravel(), E['occlusion_Xception'][k][::4, ::4].ravel()).statistic) for k in range(8)])), 3)
json.dump(agree, open(os.path.join(RES, 'method_agreement_examples.json'), 'w'), indent=1); print(agree)
