"""Architecture and pipeline diagrams drawn with matplotlib (no external images)."""
import os, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(__file__)); from common import FIG
plt.rcParams.update({'font.family': 'DejaVu Serif', 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
def box(ax, x, y, w, h, t, fc='#DCE6F2', ec='#3B6FB6', fs=7.5, bold=False, rot=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs * 0.8, fontweight='bold' if bold else 'normal', rotation=rot)
def arr(ax, x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8, lw=.9, color='#444444'))
def canvas(w, h, W, H):
    f, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off'); return f, ax
# 1. custom CNN
f, ax = canvas(7.2, 1.9, 15, 3.6)
L = [('Input\n256×256×3', '#F2F2F2'), ('Conv 3×3\n32, ReLU', None), ('MaxPool\n2×2', '#EFE3D3'), ('Conv 3×3\n64, ReLU', None), ('MaxPool\n2×2', '#EFE3D3'),
     ('Conv 3×3\n64, ReLU', None), ('MaxPool\n2×2', '#EFE3D3'), ('Conv 3×3\n64, ReLU', None), ('MaxPool\n2×2', '#EFE3D3'), ('Flatten', '#F2F2F2'),
     ('Dense 256\n(features)', '#D9EAD3'), ('Dropout\n0.5', '#F2F2F2'), ('Dense 4\nsoftmax', '#F4D9E6')]
x = .1
for i, (t, c) in enumerate(L):
    box(ax, x, 0.9, 0.95, 2.5, t.replace('\n', ' ').replace('(feature layer)', '(features)'), fc=c or '#DCE6F2', fs=7.2, rot=90)
    if i: arr(ax, x - .15, 1.85, x, 1.85)
    x += 1.15
ax.text(9.2, .55, 'Grad-CAM target layer = last convolution (conv2d_7)', fontsize=6.5, color='#B5487F', ha='center')
arr(ax, 8.6, .75, 8.15, 1.0); f.savefig(os.path.join(FIG, 'fig_arch_cnn.png')); plt.close(f)
# 2. six-predictor hybrid ensemble
f, ax = canvas(7.2, 4.0, 16, 9)
box(ax, .2, 3.9, 2.0, 1.2, 'MRI image\n(test set)', fc='#F2F2F2', bold=True)
rows = [('256 px', 'Custom CNN\n(trained from scratch)', None), ('256 px', 'InceptionV3\n(frozen backbone + head)', None), ('256 px', 'Xception\n(frozen backbone + head)', None),
        ('128 px', 'CNN extractor\nDense-256 features', 'RF + DT + SVM\nsoft vote'), ('128 px', 'InceptionV3 extractor\nDense-256 features', 'RF + DT + SVM\nsoft vote'),
        ('128 px', 'Xception extractor\nDense-256 features', 'RF + DT + SVM\nsoft vote')]
for i, (sz, m, clf) in enumerate(rows):
    yy = 8.0 - i * 1.45
    arr(ax, 2.2, 4.5, 3.3, yy + .5); ax.text(2.6, yy + .95, sz, fontsize=5.5, color='#555555')
    box(ax, 3.3, yy, 3.8, 1.0, m, fc='#DCE6F2' if i < 3 else '#D9EAD3', fs=6)
    if clf:
        box(ax, 7.6, yy, 2.4, 1.0, clf, fc='#FFF2CC', ec='#B7950B', fs=6); arr(ax, 7.1, yy + .5, 7.6, yy + .5); arr(ax, 10.0, yy + .5, 10.9, 4.5)
        ax.text(10.05, yy + .62, f'p{i + 1}', fontsize=6, color='#B5487F')
    else:
        arr(ax, 7.1, yy + .5, 10.9, 4.5); ax.text(7.2, yy + .62, f'p{i + 1}', fontsize=6, color='#B5487F')
box(ax, 10.9, 3.6, 2.6, 1.8, 'Combine six\nprobability\nvectors p1…p6', fc='#F4D9E6', ec='#B5487F', fs=6.2)
box(ax, 13.9, 4.8, 2.0, 1.2, 'Majority vote\n(one vote\nper model)', fc='#F4D9E6', ec='#B5487F', fs=5.6)
box(ax, 13.9, 3.0, 2.0, 1.2, 'Confidence-\nweighted vote\n(argmax of Σp)', fc='#F4D9E6', ec='#B5487F', fs=5.6)
arr(ax, 13.5, 4.8, 13.9, 5.3); arr(ax, 13.5, 4.2, 13.9, 3.6)
f.savefig(os.path.join(FIG, 'fig_arch_ensemble.png')); plt.close(f)
# 3. XAI pipeline
f, ax = canvas(7.2, 3.6, 16, 8)
box(ax, .2, 3.4, 2.0, 1.2, 'MRI image +\npredicted class', fc='#F2F2F2', bold=True)
for i, m in enumerate(['CNN', 'Xception', 'InceptionV3']):
    yy = 6.0 - i * 2.2; arr(ax, 2.2, 4.0, 3.0, yy + .5); box(ax, 3.0, yy, 2.2, 1.0, m, fs=7)
    arr(ax, 5.2, yy + .5, 6.0, yy + .5); box(ax, 6.0, yy - .2, 3.0, 1.4, 'Grad-CAM, Grad-CAM++,\nHiResCAM, occlusion,\nLIME' + (', SHAP' if i == 0 else ''), fc='#D9EAD3', fs=6.2)
    arr(ax, 9.0, yy + .5, 10.0, 4.6 if i == 0 else 4.0 if i == 1 else 3.4)
box(ax, 10.0, 5.8, 2.7, 1.5, 'Qualitative review\nper class, correct vs\nmisclassified', fc='#FFF2CC', ec='#B7950B', fs=6.5)
box(ax, 10.0, 3.2, 2.7, 1.9, 'Faithfulness\ndeletion / insertion AUC\nvs random baseline', fc='#FFF2CC', ec='#B7950B', fs=6.5)
box(ax, 10.0, .6, 2.7, 1.9, 'Sanity check\ncascading weight\nrandomisation', fc='#FFF2CC', ec='#B7950B', fs=6.5)
box(ax, 13.3, 5.8, 2.6, 1.5, 'Cross-model\nconsistency\n(EAA-IoU, Spearman)', fc='#F4D9E6', ec='#B5487F', fs=6.5)
box(ax, 13.3, 3.2, 2.6, 1.9, 'Shortcut check\nattribution outside\nhead and in skull rim', fc='#F4D9E6', ec='#B5487F', fs=6.5)
box(ax, 13.3, .6, 2.6, 1.9, 'SHAP on the RF\nthat reads CNN\nfeatures', fc='#F4D9E6', ec='#B5487F', fs=6.5)
for yy in (6.5, 4.1, 1.5): arr(ax, 12.7, yy, 13.3, yy)
f.savefig(os.path.join(FIG, 'fig_xai_pipeline.png')); plt.close(f)
# 4. conceptual framework
f, ax = canvas(7.2, 2.6, 16, 5.6)
stages = [('Data\nKaggle MRI,\n4 classes, 5,600\ntrain / 1,600 test', '#F2F2F2'), ('Pre-processing\nresize 256 /\n128 px, RGB ÷\n255, seed 42', '#F2F2F2'),
          ('Feature\nlearning\nCNN, InceptionV3,\nXception', '#DCE6F2'), ('Classification\nsoftmax heads\nand RF / DT /\nSVM', '#D9EAD3'),
          ('Ensemble\nmajority and\nconfidence-\nweighted vote', '#F4D9E6'), ('Evaluation\naccuracy, F1,\nROC, McNemar,\nbootstrap CI', '#FFF2CC')]
for i, (t, c) in enumerate(stages):
    box(ax, .2 + i * 2.65, 3.0, 2.3, 2.0, t, fc=c, fs=6.4)
    if i: arr(ax, .2 + i * 2.65 - .35, 4.0, .2 + i * 2.65, 4.0)
box(ax, 4.0, .2, 11.8, 1.9, 'Explainability layer (new): Grad-CAM, Grad-CAM++, HiResCAM, LIME, SHAP, occlusion\n→ faithfulness, sanity check, cross-model agreement (EAA-IoU), shortcut check\n→ clinician-facing summary and web demonstrator',
    fc='#FFF9C4', ec='#B7950B', fs=6.6)
for x0 in (6.65, 9.3, 11.95): arr(ax, x0, 3.0, x0, 1.9)
f.savefig(os.path.join(FIG, 'fig_conceptual_framework.png')); plt.close(f)
print('ok')
