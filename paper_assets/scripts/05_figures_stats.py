"""Results figures (Chapter 3-4) and statistics. Every number comes from metrics/, confusion/, history/ or the verified re-run."""
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.dirname(__file__)); from common import *
from scipy.stats import binomtest, mannwhitneyu
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.titlesize': 10})
C = ['#3B6FB6', '#E07B39', '#4E9F6E', '#B5487F']          # one colour per class, colour-blind safe
M, CF, H = [os.path.join(ROOT, k) for k in ('metrics', 'confusion', 'history')]
master = pd.read_csv(os.path.join(M, 'results_master.csv')); stats = {}
NICE = {'CNN_standalone': 'CNN', 'InceptionV3_standalone': 'InceptionV3', 'Xception_standalone': 'Xception',
        'cnn_feat_rf': 'CNN feat. + RF', 'cnn_feat_dt': 'CNN feat. + DT', 'cnn_feat_svm': 'CNN feat. + SVM', 'cnn_feat_softvote_ensemble': 'CNN feat. + soft vote',
        'inceptionv3_feat_rf': 'InceptionV3 feat. + RF', 'inceptionv3_feat_dt': 'InceptionV3 feat. + DT', 'inceptionv3_feat_svm': 'InceptionV3 feat. + SVM',
        'inceptionv3_feat_softvote_ensemble': 'InceptionV3 feat. + soft vote', 'xception_feat_rf': 'Xception feat. + RF', 'xception_feat_dt': 'Xception feat. + DT',
        'xception_feat_svm': 'Xception feat. + SVM', 'xception_feat_softvote_ensemble': 'Xception feat. + soft vote', 'raw_rf': 'Raw pixels + RF',
        'raw_dt': 'Raw pixels + DT', 'raw_svm': 'Raw pixels + SVM', 'raw_lr': 'Raw pixels + LR', 'raw_softvote_ensemble': 'Raw pixels + soft vote',
        'FINAL_majority_vote': 'Final 6-model majority vote', 'FINAL_confidence_weighted': 'Final 6-model confidence-weighted vote'}
def save(f, name): f.savefig(os.path.join(FIG, name)); plt.close(f)
def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return c - h, c + h
# ---- dataset
cnt = json.load(open(os.path.join(M, 'dataset_counts.json')))
f, ax = plt.subplots(figsize=(6, 2.6)); w = 0.38; x = np.arange(4)
for k, (sp, col) in enumerate([('Training', '#3B6FB6'), ('Testing', '#E07B39')]):
    v = [cnt[sp][c] for c in CLASSES]; b = ax.bar(x + (k - .5) * w, v, w, label=sp, color=col)
    ax.bar_label(b, fontsize=8)
ax.set_xticks(x, LABELS); ax.set_ylabel('Images'); ax.legend(frameon=False); ax.set_ylim(0, 1650); save(f, 'fig_dataset_distribution.png')
X, y, files = load_numpy('Testing', 256); rng = np.random.default_rng(7)
f, axs = plt.subplots(2, 4, figsize=(6.4, 3.4))
for c in range(4):
    for r in range(2):
        i = rng.choice(np.where(y == c)[0]); axs[r, c].imshow(X[i]); axs[r, c].axis('off')
    axs[0, c].set_title(LABELS[c])
save(f, 'fig_sample_mri.png')
# ---- training curves
for group, names in [('standalone', [('cnn_model', 'CNN (256 px)'), ('inceptionv3_model', 'InceptionV3 (256 px)'), ('xception_model', 'Xception (256 px)')]),
                     ('extractors', [('cnn_ext', 'CNN extractor (128 px)'), ('inceptionv3_ext', 'InceptionV3 extractor (128 px)'), ('xception_ext', 'Xception extractor (128 px)')])]:
    f, axs = plt.subplots(2, 3, figsize=(7, 4.2), sharex='col')
    for j, (fn, t) in enumerate(names):
        h = pd.read_csv(os.path.join(H, fn + '_history.csv')); e = h.epoch + 1
        axs[0, j].plot(e, h.accuracy, color='#3B6FB6', label='train'); axs[0, j].plot(e, h.val_accuracy, color='#E07B39', label='validation')
        axs[1, j].plot(e, h.loss, color='#3B6FB6'); axs[1, j].plot(e, h.val_loss, color='#E07B39')
        axs[0, j].set_title(t); axs[1, j].set_xlabel('Epoch'); stats.setdefault('history', {})[fn] = {
            'epochs_logged': int(len(h)), 'final_train_acc': round(float(h.accuracy.iloc[-1]), 4), 'final_val_acc': round(float(h.val_accuracy.iloc[-1]), 4),
            'best_val_acc': round(float(h.val_accuracy.max()), 4), 'best_val_epoch': int(h.epoch[h.val_accuracy.idxmax()] + 1),
            'min_val_loss_epoch': int(h.epoch[h.val_loss.idxmin()] + 1), 'final_val_loss': round(float(h.val_loss.iloc[-1]), 4), 'min_val_loss': round(float(h.val_loss.min()), 4)}
    axs[0, 0].set_ylabel('Accuracy'); axs[1, 0].set_ylabel('Loss'); axs[0, 0].legend(frameon=False, fontsize=8)
    save(f, f'fig_training_{group}.png')
# ---- confusion matrices (row-normalised)
def cmplot(keys, name, ncol):
    nrow = int(np.ceil(len(keys) / ncol)); f, axs = plt.subplots(nrow, ncol, figsize=(1.75 * ncol, 1.8 * nrow)); axs = np.atleast_1d(axs).ravel()
    for a, k in zip(axs, keys):
        m = pd.read_csv(os.path.join(CF, k + '_confusion_matrix.csv'), index_col=0).values; n = m / m.sum(1, keepdims=True)
        a.imshow(n, cmap='Blues', vmin=0, vmax=1)
        for i in range(4):
            for j in range(4): a.text(j, i, f'{n[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='white' if n[i, j] > .6 else 'black')
        a.set_xticks(range(4), ['G', 'M', 'N', 'P'], fontsize=7); a.set_yticks(range(4), ['G', 'M', 'N', 'P'], fontsize=7)
        a.set_title(NICE[k].replace('Final 6-model confidence-weighted vote','Final: conf.-weighted').replace('Final 6-model majority vote','Final: majority vote').replace('InceptionV3 feat.','IncV3 feat.').replace('Xception feat.','Xcp feat.'), fontsize=7.5); a.set_xlabel('Predicted', fontsize=7); a.set_ylabel('True', fontsize=7)
    for a in axs[len(keys):]: a.axis('off')
    f.tight_layout(); save(f, name)
cmplot(['CNN_standalone', 'InceptionV3_standalone', 'Xception_standalone', 'FINAL_majority_vote', 'FINAL_confidence_weighted'], 'fig_cm_deep_final.png', 5)
cmplot([k for k in NICE if '_feat_' in k], 'fig_cm_feature_ensembles.png', 4)
cmplot([k for k in NICE if k.startswith('raw_')], 'fig_cm_raw.png', 5)
# ---- accuracy chart with Wilson 95% CI
order = master.sort_values('accuracy'); n = int(master.n_test.iloc[0])
lo, hi = wilson(order.accuracy.values, n)
fam = lambda k: '#B5487F' if k.startswith('FINAL') else '#3B6FB6' if k.endswith('standalone') else '#4E9F6E' if '_feat_' in k else '#9A9A9A'
f, ax = plt.subplots(figsize=(6.4, 5.2))
ax.barh([NICE[k] for k in order.model], 100 * order.accuracy, color=[fam(k) for k in order.model],
        xerr=[100 * (order.accuracy - lo), 100 * (hi - order.accuracy)], error_kw=dict(lw=.8, capsize=2))
for i, v in enumerate(order.accuracy): ax.text(100 * hi[i] + .4, i, f'{100 * v:.2f}', va='center', fontsize=7.5)
ax.set_xlim(70, 100); ax.set_xlabel('Test accuracy (%) with Wilson 95% CI, n = 1,600')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=l) for c, l in [('#B5487F', 'Final ensemble'), ('#3B6FB6', 'Deep standalone'), ('#4E9F6E', 'Deep features + classical'), ('#9A9A9A', 'Raw pixels + classical')]],
          frameon=False, loc='lower right', fontsize=7.5); save(f, 'fig_accuracy_all.png')
stats['wilson_ci'] = {k: [round(100 * float(a), 2), round(100 * float(b), 2)] for k, a, b in zip(order.model, lo, hi)}
# ---- per-class F1 heatmap
fm = master.set_index('model')[['f1_glioma', 'f1_meningioma', 'f1_notumor', 'f1_pituitary']].loc[list(NICE)]
f, ax = plt.subplots(figsize=(5.2, 6.2)); im = ax.imshow(fm.values, cmap='viridis', vmin=.65, vmax=1, aspect='auto')
for i in range(len(fm)):
    for j in range(4): ax.text(j, i, f'{fm.values[i, j]:.3f}', ha='center', va='center', fontsize=7, color='white' if fm.values[i, j] < .85 else 'black')
ax.set_xticks(range(4), LABELS); ax.set_yticks(range(len(fm)), [NICE[k] for k in fm.index], fontsize=7.5); plt.colorbar(im, ax=ax, shrink=.6, label='F1-score')
save(f, 'fig_f1_heatmap.png')
# ---- ROC / PR for the standalone deep models (recomputed probabilities, verified identical to the logged predictions)
P = np.load(os.path.join(RES, 'standalone_test_probs.npz')); yt = P['y']; stats['auc'] = {}
f, axs = plt.subplots(2, 3, figsize=(7, 4.6))
for j, nm in enumerate(['CNN', 'InceptionV3', 'Xception']):
    p = P[nm]; stats['auc'][nm] = {}
    for c in range(4):
        fpr, tpr, _ = roc_curve(yt == c, p[:, c]); pr, rc, _ = precision_recall_curve(yt == c, p[:, c])
        a, ap = auc(fpr, tpr), average_precision_score(yt == c, p[:, c]); stats['auc'][nm][CLASSES[c]] = {'roc_auc': round(a, 4), 'pr_auc': round(ap, 4)}
        axs[0, j].plot(fpr, tpr, color=C[c], lw=1.2, label=f'{LABELS[c]} {a:.3f}'); axs[1, j].plot(rc, pr, color=C[c], lw=1.2, label=f'{LABELS[c]} {ap:.3f}')
    stats['auc'][nm]['macro_roc_auc'] = round(float(roc_auc_score(yt, p, multi_class='ovr', average='macro')), 4)
    axs[0, j].plot([0, 1], [0, 1], ':', color='grey', lw=.8); axs[0, j].set_title(nm); axs[0, j].legend(frameon=False, fontsize=6.5, title='ROC AUC', title_fontsize=6.5)
    axs[1, j].legend(frameon=False, fontsize=6.5, title='Average precision', title_fontsize=6.5, loc='lower left')
    axs[0, j].set_xlabel('False positive rate'); axs[1, j].set_xlabel('Recall')
axs[0, 0].set_ylabel('True positive rate'); axs[1, 0].set_ylabel('Precision'); f.tight_layout(); save(f, 'fig_roc_pr.png')
# ---- McNemar (exact) and bootstrap CIs for the models with per-image predictions
d = pd.read_csv(os.path.join(M, 'final_per_image_predictions.csv')); y0 = d.true.values
pc = {'CNN': d.pred_CNN, 'InceptionV3': d.pred_InceptionV3, 'Xception': d.pred_Xception, 'CNN_ensemble': d.pred_CNN_ensemble,
      'InceptionV3_ensemble': d.pred_InceptionV3_ensemble, 'Xception_ensemble': d.pred_Xception_ensemble,
      'FINAL_majority': d.pred_majority, 'FINAL_confidence_weighted': d.pred_confidence_weighted}
corr = {k: (v.values == y0) for k, v in pc.items()}
rng = np.random.default_rng(42); B = 2000; boot = rng.integers(0, len(y0), (B, len(y0)))
stats['bootstrap_ci'] = {k: {'acc': round(100 * float(v.mean()), 2), 'ci95': [round(100 * float(np.percentile(v[boot].mean(1), q)), 2) for q in (2.5, 97.5)]} for k, v in corr.items()}
pairs = [('FINAL_majority', k) for k in ['CNN', 'InceptionV3', 'Xception', 'CNN_ensemble', 'InceptionV3_ensemble', 'Xception_ensemble', 'FINAL_confidence_weighted']] + \
        [('CNN', 'Xception'), ('CNN', 'InceptionV3'), ('Xception', 'InceptionV3')]
rows = []
for a, b in pairs:
    n01 = int((corr[a] & ~corr[b]).sum()); n10 = int((~corr[a] & corr[b]).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    rows.append(dict(a=a, b=b, a_right_b_wrong=n01, a_wrong_b_right=n10, p_exact=p))
mc = pd.DataFrame(rows).sort_values('p_exact'); m_ = len(mc); mc['p_holm'] = np.minimum(1, np.maximum.accumulate((m_ - np.arange(m_)) * mc.p_exact.values))
mc = mc.sort_index(); stats['mcnemar'] = mc.round(6).to_dict('records'); mc.to_csv(os.path.join(RES, 'mcnemar.csv'), index=False)
# within-ensemble pipeline accuracies (cv2 resize) vs results_master (Keras loader)
stats['member_acc_in_ensemble_pipeline'] = {k: round(100 * float(v.mean()), 2) for k, v in corr.items()}
# agreement / diversity of the six members
votes = np.stack([pc[k].values for k in ['CNN', 'InceptionV3', 'Xception', 'CNN_ensemble', 'InceptionV3_ensemble', 'Xception_ensemble']])
nagree = np.array([np.bincount(votes[:, i], minlength=4).max() for i in range(len(y0))])
stats['vote_agreement'] = {f'{k}_of_6': {'images': int((nagree == k).sum()), 'majority_acc_pct': round(100 * float((d.pred_majority.values == y0)[nagree == k].mean()), 2) if (nagree == k).any() else None} for k in range(2, 7)}
stats['oracle_any_member_correct_pct'] = round(100 * float((votes == y0).any(0).mean()), 2)
stats['all_members_wrong'] = int((~(votes == y0).any(0)).sum())
# augmented meningioma test images
aug = np.array(['aug' in f for f in P['files']]); men = yt == 1
stats['meningioma_aug'] = {'n_aug': int((aug & men).sum()), 'n_orig': int((~aug & men).sum()),
    'ensemble_recall_aug_pct': round(100 * float(corr['FINAL_majority'][aug & men].mean()), 2), 'ensemble_recall_orig_pct': round(100 * float(corr['FINAL_majority'][~aug & men].mean()), 2),
    'cnn_recall_aug_pct': round(100 * float(corr['CNN'][aug & men].mean()), 2), 'cnn_recall_orig_pct': round(100 * float(corr['CNN'][~aug & men].mean()), 2)}
# ---- EAA-IoU figures (logged run, whole test set)
e = pd.read_csv(os.path.join(M, 'eaa_iou_per_image.csv')).drop_duplicates('idx')
f, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
data = [e[(e.true == c)].eaa_iou for c in range(4)]
axs[0].boxplot(data, tick_labels=['G', 'M', 'N', 'P'], showfliers=False, patch_artist=True, boxprops=dict(facecolor='#DCE6F2'))
axs[0].set_title('EAA-IoU by true class'); axs[0].set_ylabel('EAA-IoU')
axs[1].boxplot([e[e.correct == 1].eaa_iou, e[e.correct == 0].eaa_iou], tick_labels=['Correct', 'Misclassified'], showfliers=False, patch_artist=True, boxprops=dict(facecolor='#DCE6F2'))
axs[1].set_title('Correct vs misclassified')
fpr, tpr, _ = roc_curve(1 - e.correct, -e.eaa_iou); axs[2].plot(fpr, tpr, color='#B5487F'); axs[2].plot([0, 1], [0, 1], ':', color='grey')
axs[2].set_title(f'Low EAA-IoU as error flag (AUROC {auc(fpr, tpr):.3f})'); axs[2].set_xlabel('False positive rate'); axs[2].set_ylabel('True positive rate')
f.tight_layout(); save(f, 'fig_eaa_iou.png')
stats['eaa_recomputed'] = {'auroc': round(float(auc(fpr, tpr)), 4), 'n': int(len(e)), 'median_correct': round(float(e[e.correct == 1].eaa_iou.median()), 4),
                           'median_misclassified': round(float(e[e.correct == 0].eaa_iou.median()), 4),
                           'pair_means': {c: round(float(e[c].mean()), 4) for c in ['iou_cnn_xcp', 'iou_cnn_inc', 'iou_xcp_inc']},
                           'share_zero_eaa_pct': round(100 * float((e.eaa_iou == 0).mean()), 2)}
for q in [0.05, 0.1]:
    flag = e.eaa_iou < q; stats['eaa_recomputed'][f'threshold_{q}'] = {'flagged_pct': round(100 * float(flag.mean()), 2),
        'errors_caught_pct': round(100 * float((flag & (e.correct == 0)).sum() / (e.correct == 0).sum()), 2),
        'error_rate_flagged_pct': round(100 * float(1 - e[flag].correct.mean()), 2), 'error_rate_unflagged_pct': round(100 * float(1 - e[~flag].correct.mean()), 2)}
# confidence as error flag for comparison
stats['confidence_auroc'] = round(float(roc_auc_score(d.pred_confidence_weighted.values != y0, -d.ensemble_confidence.values)), 4)
# ---- old (HND, 2024) vs re-run (2026) comparison chart
old = {'CNN': 94.0, 'Xception': 86.91, 'InceptionV3': 84.63, 'CNN feat. ens.': 83.71, 'Xception feat. ens.': 70.28, 'InceptionV3 feat. ens.': 71.20, 'Raw-pixel ens.': 83.0, 'Final ensemble': 98.20}
new = {'CNN': 90.44, 'Xception': 87.62, 'InceptionV3': 86.62, 'CNN feat. ens.': 89.31, 'Xception feat. ens.': 88.31, 'InceptionV3 feat. ens.': 86.12, 'Raw-pixel ens.': 84.06, 'Final ensemble': 92.50}
for k, mk in [('CNN', 'CNN_standalone'), ('Xception', 'Xception_standalone'), ('InceptionV3', 'InceptionV3_standalone'), ('CNN feat. ens.', 'cnn_feat_softvote_ensemble'),
              ('Xception feat. ens.', 'xception_feat_softvote_ensemble'), ('InceptionV3 feat. ens.', 'inceptionv3_feat_softvote_ensemble'), ('Raw-pixel ens.', 'raw_softvote_ensemble'), ('Final ensemble', 'FINAL_majority_vote')]:
    assert abs(new[k] - 100 * master.set_index('model').accuracy[mk]) < 1e-6, k
f, ax = plt.subplots(figsize=(6.4, 2.9)); x = np.arange(len(old))
b1 = ax.bar(x - .2, list(old.values()), .4, color='#C9C9C9', label='HND report (2024)'); b2 = ax.bar(x + .2, list(new.values()), .4, color='#3B6FB6', label='Standardised re-run (2026), held-out test')
ax.bar_label(b1, fmt='%.1f', fontsize=6.5); ax.bar_label(b2, fmt='%.1f', fontsize=6.5)
ax.set_xticks(x, list(old.keys()), rotation=25, ha='right', fontsize=7.5); ax.set_ylim(60, 102); ax.set_ylabel('Accuracy (%)'); ax.legend(frameon=False, fontsize=7.5, loc='lower left')
ax.annotate('98.20% was measured on the\n6,985 training images', xy=(7 - .2, 98.2), xytext=(4.2, 99), fontsize=6.5, arrowprops=dict(arrowstyle='->', lw=.6))
save(f, 'fig_old_vs_new.png')
json.dump(stats, open(os.path.join(RES, 'results_stats.json'), 'w'), indent=2); print(json.dumps({k: stats[k] for k in ['mcnemar', 'bootstrap_ci', 'meningioma_aug', 'eaa_recomputed', 'confidence_auroc', 'vote_agreement', 'oracle_any_member_correct_pct', 'all_members_wrong', 'member_acc_in_ensemble_pipeline']}, indent=1))
