from engine import *

def row(label, m):
    return [label, f'<<acc.{m}>>', f'<<prec.{m}>>', f'<<rec.{m}>>', f'<<f1.{m}>>', f'<<st.wilson_ci.{m}>>']
HDR = ['Model', 'Accuracy (%)', 'Macro precision', 'Macro recall', 'Macro F1', 'Wilson 95% CI (%)']
W = [4.6, 2.0, 2.0, 2.0, 1.8, 3.0]

def ch4():
    mc_rows = []
    for a, b, la, lb in [('FINAL_majority', 'CNN', 'Majority vote', 'CNN'), ('FINAL_majority', 'Xception', 'Majority vote', 'Xception'),
                         ('FINAL_majority', 'InceptionV3', 'Majority vote', 'InceptionV3'), ('FINAL_majority', 'CNN_ensemble', 'Majority vote', 'CNN features + soft vote'),
                         ('FINAL_majority', 'Xception_ensemble', 'Majority vote', 'Xception features + soft vote'), ('FINAL_majority', 'InceptionV3_ensemble', 'Majority vote', 'InceptionV3 features + soft vote'),
                         ('FINAL_majority', 'FINAL_confidence_weighted', 'Majority vote', 'Confidence-weighted vote'), ('CNN', 'Xception', 'CNN', 'Xception'),
                         ('CNN', 'InceptionV3', 'CNN', 'InceptionV3'), ('Xception', 'InceptionV3', 'Xception', 'InceptionV3')]:
        k = f'mc.{a}.{b}'; mc_rows.append([la, lb, f'<<{k}.n01>>', f'<<{k}.n10>>', f'<<{k}.p>>', f'<<{k}.ph>>'])
    boot = [[n, f'<<st.bootstrap_ci.{k}.acc>>', f'<<st.bootstrap_ci.{k}.ci95>>'] for n, k in [
        ('CNN', 'CNN'), ('InceptionV3', 'InceptionV3'), ('Xception', 'Xception'), ('CNN features + soft vote', 'CNN_ensemble'),
        ('InceptionV3 features + soft vote', 'InceptionV3_ensemble'), ('Xception features + soft vote', 'Xception_ensemble'),
        ('Final majority vote', 'FINAL_majority'), ('Final confidence-weighted vote', 'FINAL_confidence_weighted')]]
    fin = [[c, f'<<precision.FINAL_majority_vote.{k}>>', f'<<recall.FINAL_majority_vote.{k}>>', f'<<f1c.FINAL_majority_vote.{k}>>', '400']
           for c, k in [('Glioma', 'glioma'), ('Meningioma', 'meningioma'), ('No tumour', 'notumor'), ('Pituitary', 'pituitary')]]
    auc_rows = [[m] + [f'<<st.auc.{m}.{c}.roc_auc>> / <<st.auc.{m}.{c}.pr_auc>>' for c in ['glioma', 'meningioma', 'notumor', 'pituitary']] + [f'<<st.auc.{m}.macro_roc_auc>>']
                for m in ['CNN', 'InceptionV3', 'Xception']]
    old_new = [['Custom CNN', '94 (test)', '<<acc.CNN_standalone>>'], ['Xception', '86.91 (test)', '<<acc.Xception_standalone>>'],
               ['InceptionV3', '84.63 (test)', '<<acc.InceptionV3_standalone>>'], ['CNN features + RF / DT / SVM', '85.45 / 80.99 / 66.60', '<<acc.cnn_feat_rf>> / <<acc.cnn_feat_dt>> / <<acc.cnn_feat_svm>>'],
               ['Xception features + RF / DT / SVM', '73.67 / 68.20 / 56.71', '<<acc.xception_feat_rf>> / <<acc.xception_feat_dt>> / <<acc.xception_feat_svm>>'],
               ['InceptionV3 features + RF / DT / SVM', '74.64 / 68.05 / 57.34', '<<acc.inceptionv3_feat_rf>> / <<acc.inceptionv3_feat_dt>> / <<acc.inceptionv3_feat_svm>>'],
               ['Raw pixels + SVM / DT / RF', '88.26 / 80.02 / 79.52 (split of training folder)', '<<acc.raw_svm>> / <<acc.raw_dt>> / <<acc.raw_rf>>'],
               ['Final six-model majority vote', '98.20 (on 6,985 training images)', '<<acc.FINAL_majority_vote>> (1,600 held-out images)']]
    return [
        H(1, 'Chapter 4 Results'),
        P('This chapter reports the classification results of the standardised re-run on the 1,600 held-out test images. It moves from '
          'training behaviour to the individual models, the ensembles, statistical comparisons, per-class behaviour and reproducibility '
          'checks, and ends by comparing the new results with the HND report and testing hypotheses H1 and H2. All values come from the '
          'logged result files (Appendix H); figures and tables in this chapter are new because every number was regenerated.', new=True),
        H(2, '4.1 Training behaviour', new=True),
        P('{fig:train_sa} and {fig:train_ext} show the training and validation curves. The custom CNN fits the training data almost perfectly '
          '(final training accuracy <<st.history.cnn_model.final_train_acc>>, validation <<st.history.cnn_model.final_val_acc>>). Its validation '
          'loss is lowest at epoch <<st.history.cnn_model.min_val_loss_epoch>> (<<st.history.cnn_model.min_val_loss>>) and then drifts upward to '
          '<<st.history.cnn_model.final_val_loss>>, a sign of mild over-fitting that early stopping would have reduced. The CNN history file '
          'holds <<st.history.cnn_model.epochs_logged>> rows because one epoch was lost when training resumed after a disconnection. The frozen '
          'InceptionV3 and Xception classifiers learn more slowly and were still improving at epoch 10 (final validation accuracy '
          '<<st.history.inceptionv3_model.final_val_acc>> and <<st.history.xception_model.final_val_acc>>), which suggests they were '
          'under-trained rather than over-fitted. Validation accuracy is several points higher than test accuracy for every model (for the CNN, '
          '<<st.history.cnn_model.final_val_acc>> against 0.9044). The validation images come from the same folder as the training images, so '
          'this gap is an early warning that performance on new data will be lower.', new=True),
        FIG('train_sa', 'paper_assets/figures/fig_training_standalone.png', 'Training and validation accuracy and loss of the standalone deep models (history/*.csv)', new=True, width=6.2),
        FIG('train_ext', 'paper_assets/figures/fig_training_extractors.png', 'Training and validation curves of the 128-pixel feature extractors', new=True, width=6.2),
        H(2, '4.2 Standalone deep models', new=True),
        P('{tab:deep} gives the test results. The custom CNN is the most accurate single model (<<acc.CNN_standalone>>%), ahead of Xception '
          '(<<acc.Xception_standalone>>%) and InceptionV3 (<<acc.InceptionV3_standalone>>%). The difference between the CNN and each pretrained '
          'network is statistically significant, but the difference between the two pretrained networks is not (Section 4.6).', new=True),
        TAB('deep', 'Test results of the standalone deep models (n = 1,600)', HDR,
            [row('Custom CNN (256 px)', 'CNN_standalone'), row('InceptionV3, frozen backbone (256 px)', 'InceptionV3_standalone'), row('Xception, frozen backbone (256 px)', 'Xception_standalone')],
            new=True, widths=W, font=9),
        H(2, '4.3 Classical classifiers on raw pixels', new=True),
        P('On 65,536 raw greyscale pixels ({tab:raw}), the Random Forest is the best classical model (<<acc.raw_rf>>%), followed by the SVM '
          '(<<acc.raw_svm>>%), Logistic Regression (<<acc.raw_lr>>%) and the Decision Tree (<<acc.raw_dt>>%). Their soft vote reaches '
          '<<acc.raw_softvote_ensemble>>%, below the Random Forest alone. A soft vote does not help when one member (here the Decision Tree) '
          'is much weaker and over-confident. These models are also the most expensive: the raw-pixel SVM took 5812.1 seconds to train.', new=True),
        TAB('raw', 'Test results of classical classifiers on raw pixels', HDR,
            [row('Random Forest', 'raw_rf'), row('Support Vector Machine', 'raw_svm'), row('Logistic Regression', 'raw_lr'), row('Decision Tree', 'raw_dt'),
             row('Soft vote (RF + DT + SVM)', 'raw_softvote_ensemble')], new=True, widths=W, font=9),
        H(2, '4.4 Deep features with classical classifiers', new=True),
        P('Replacing raw pixels with 256 deep features improves every classical classifier ({tab:feat}). With CNN features, the Random Forest '
          'reaches <<acc.cnn_feat_rf>>% and the SVM <<acc.cnn_feat_svm>>%. Xception features come next (Random Forest <<acc.xception_feat_rf>>%), '
          'then InceptionV3 features (Random Forest <<acc.inceptionv3_feat_rf>>%). The SVM benefits most: from <<acc.raw_svm>>% on raw pixels to '
          '<<acc.cnn_feat_svm>>% on CNN features, while its training time fell from 5812.1 to 2.7 seconds. Within each feature source, the soft '
          'vote is close to, but not above, the best member.', new=True),
        TAB('feat', 'Test results of classical classifiers on deep features (128 px extractors)', HDR,
            [row('CNN features + RF', 'cnn_feat_rf'), row('CNN features + SVM', 'cnn_feat_svm'), row('CNN features + DT', 'cnn_feat_dt'), row('CNN features + soft vote', 'cnn_feat_softvote_ensemble'),
             row('Xception features + RF', 'xception_feat_rf'), row('Xception features + SVM', 'xception_feat_svm'), row('Xception features + DT', 'xception_feat_dt'), row('Xception features + soft vote', 'xception_feat_softvote_ensemble'),
             row('InceptionV3 features + RF', 'inceptionv3_feat_rf'), row('InceptionV3 features + SVM', 'inceptionv3_feat_svm'), row('InceptionV3 features + DT', 'inceptionv3_feat_dt'), row('InceptionV3 features + soft vote', 'inceptionv3_feat_softvote_ensemble')],
            new=True, widths=W, font=9),
        H(2, '4.5 The six-model ensemble', new=True),
        P('The six-predictor majority vote achieves **<<acc.FINAL_majority_vote>>%** accuracy on the held-out test set (macro F1 '
          '<<f1.FINAL_majority_vote>>), and the confidence-weighted vote achieves <<acc.FINAL_confidence_weighted>>% (macro F1 '
          '<<f1.FINAL_confidence_weighted>>). This is the headline result of the study. It replaces the 98.20% reported in the HND study, which '
          'was measured on training images. {tab:final} gives the per-class results of the majority vote, and {fig:cm_deep} shows the '
          'row-normalised confusion matrices of the deep models and both ensembles.', new=True),
        TAB('final', 'Per-class test results of the six-model majority vote', ['Class', 'Precision (%)', 'Recall (%)', 'F1-score', 'Support'], fin, new=True, widths=[4, 3, 3, 3, 2.5], font=9),
        FIG('cm_deep', 'paper_assets/figures/fig_cm_deep_final.png', 'Row-normalised confusion matrices of the standalone deep models and the two final ensembles (G = glioma, M = meningioma, N = no tumour, P = pituitary)', new=True, width=6.3),
        P('No-tumour images are always recognised (recall <<recall.FINAL_majority_vote.notumor>>%) and pituitary tumours nearly always '
          '(<<recall.FINAL_majority_vote.pituitary>>%). Glioma is the weak point: recall is only <<recall.FINAL_majority_vote.glioma>>%, with '
          '<<cm.FINAL_majority_vote.glioma.meningioma>> of 400 gliomas predicted as meningioma and <<cm.FINAL_majority_vote.glioma.notumor>> '
          'predicted as no tumour. The second error matters clinically, because it is a missed tumour. The ensemble mostly helps by correcting '
          'errors that only a minority of members make. All six members agree on <<st.vote_agreement.6_of_6.images>> images, where the vote is '
          '<<st.vote_agreement.6_of_6.majority_acc_pct>>% accurate. On the <<st.vote_agreement.3_of_6.images>> images where the best class '
          'receives only three votes, accuracy falls to <<st.vote_agreement.3_of_6.majority_acc_pct>>%. At least one member is correct for '
          '<<st.oracle_any_member_correct_pct>>% of images, but on <<st.all_members_wrong>> images all six are wrong, so the ensemble cannot recover them.', new=True),
        P('Majority voting and confidence-weighted voting differ on only 22 images (13 favour majority voting and 9 favour confidence '
          'weighting). The difference is not significant (exact McNemar p = <<mc.FINAL_majority.FINAL_confidence_weighted.p>>). Confidence '
          'weighting therefore does not improve accuracy here, but it does provide a useful per-image confidence score (Section 5.7).', new=True),
        H(2, '4.6 Statistical comparison of models', new=True),
        P('{tab:mcnemar} lists the exact McNemar tests. The columns n01 and n10 count the images that only model A, or only model B, classifies '
          'correctly. The majority vote is significantly better than every one of its six members after Holm correction. The CNN is '
          'significantly better than both Xception and InceptionV3, while Xception and InceptionV3 are not significantly different. '
          '{tab:boot} gives bootstrap 95% intervals, and {fig:acc_all} places all 22 configurations on one chart with Wilson intervals.', new=True),
        TAB('mcnemar', 'Exact McNemar tests on the 1,600 test images (Holm-adjusted p over ten comparisons)', ['Model A', 'Model B', 'n01 (A right, B wrong)', 'n10 (A wrong, B right)', 'p (exact)', 'p (Holm)'],
            mc_rows, new=True, widths=[3.0, 4.0, 2.2, 2.2, 2.2, 2.0], font=8.5),
        TAB('boot', 'Bootstrap 95% confidence intervals for test accuracy (2,000 resamples, ensemble pipeline)', ['Predictor', 'Accuracy (%)', '95% CI (%)'], boot, new=True, widths=[6, 3, 4], font=9,
            note='Note: inside the ensemble pipeline images are resized with OpenCV, so the three standalone members score slightly differently (90.50, 86.38 and 87.75%) from the Keras-loader figures in {tab:deep} (Section 4.9).'),
        FIG('acc_all', 'paper_assets/figures/fig_accuracy_all.png', 'Test accuracy of all 22 configurations with Wilson 95% confidence intervals', new=True, width=6.0),
        H(2, '4.7 Per-class performance', new=True),
        P('{fig:f1} shows per-class F1 for every configuration. The same pattern holds across all 22: no-tumour and pituitary are easy, '
          'meningioma is intermediate, and glioma is hardest. Glioma F1 ranges from <<f1c.raw_lr.glioma>> (raw-pixel Logistic Regression) to '
          '<<f1c.FINAL_majority_vote.glioma>> (majority vote). Among single models, Xception has the lowest glioma recall '
          '(<<recall.Xception_standalone.glioma>>%), mainly because it labels <<cm.Xception_standalone.glioma.meningioma>> gliomas as '
          'meningioma. The CNN misses fewer gliomas into meningioma (<<cm.CNN_standalone.glioma.meningioma>>) but more into no tumour '
          '(<<cm.CNN_standalone.glioma.notumor>>). This complementarity is part of why the vote helps. Confusion matrices of all classical '
          'configurations are in {fig:cm_feat} and {fig:cm_raw}.', new=True),
        FIG('f1', 'paper_assets/figures/fig_f1_heatmap.png', 'Per-class F1-score of every configuration', new=True, width=5.0),
        FIG('cm_feat', 'paper_assets/figures/fig_cm_feature_ensembles.png', 'Row-normalised confusion matrices of the classical classifiers on deep features', new=True, width=6.2),
        FIG('cm_raw', 'paper_assets/figures/fig_cm_raw.png', 'Row-normalised confusion matrices of the classical classifiers on raw pixels', new=True, width=6.3),
        H(2, '4.8 ROC and precision–recall analysis', new=True),
        P('The ROC and precision–recall curves of the three deep classifiers are in {fig:roc} and the areas in {tab:auc}. Ranking quality is '
          'high for all three (macro ROC AUC between <<st.auc.Xception.macro_roc_auc>> and <<st.auc.InceptionV3.macro_roc_auc>>). Interestingly, '
          'InceptionV3 has the best macro ROC AUC even though it has the lowest accuracy. Its probabilities rank cases well, but its default '
          'decision rule (argmax) places the boundaries less well. Glioma has the lowest areas for every model; the CNN’s glioma ROC AUC '
          '(<<st.auc.CNN.glioma.roc_auc>>) is the lowest of all, which matches its glioma-to-no-tumour errors.', new=True),
        FIG('roc', 'paper_assets/figures/fig_roc_pr.png', 'One-vs-rest ROC (top) and precision–recall (bottom) curves of the three deep classifiers', new=True, width=6.3),
        TAB('auc', 'ROC AUC / average precision per class and macro ROC AUC', ['Model', 'Glioma', 'Meningioma', 'No tumour', 'Pituitary', 'Macro ROC AUC'], auc_rows, new=True, widths=[2.6, 2.6, 2.6, 2.6, 2.6, 2.2], font=8.5),
        H(2, '4.9 Reproducibility checks', new=True),
        P('Three checks were run. First, the three saved standalone models were reloaded and re-run on the test images. Their predictions agreed '
          'with the logged per-image predictions for 100% of images (verification.json). Second, the whole six-predictor ensemble was '
          'rebuilt from the saved extractors and classical classifiers, and reproduced <<final_ensemble_rerun.majority.accuracy_pct>>% (majority) '
          'and <<final_ensemble_rerun.confidence_weighted.accuracy_pct>>% (confidence-weighted), again with 100% per-image agreement. Third, the '
          'Random Forest on CNN features was re-run and gave <<shap_rf.rf_accuracy_recomputed_pct>>%, as logged. One small inconsistency was '
          'found: standalone metrics were computed with the Keras image loader, while the ensemble re-loads images with OpenCV. The two '
          'resizing routines differ slightly, which moves the standalone accuracies by up to 0.24 percentage points (for example, the CNN scores '
          '90.44% with the Keras loader and 90.50% in the ensemble pipeline). Both figures are reported, and the conclusions do not depend on which is used.', new=True),
        H(2, '4.10 Comparison with the HND (2024) results', new=True),
        P('{tab:oldnew} and {fig:oldnew} compare the HND figures with the standardised re-run. The comparison is indicative only: the HND study '
          'used an older 9,047-image copy of the data, different preprocessing per notebook, a classical baseline tested on a split of the '
          'training folder, and a final ensemble evaluated on the training images. The feature-based classical models improve sharply in the '
          're-run, mainly because the HND notebooks used batch size 5, mixed normalisation and, for the CNN pipeline, a crashing notebook. The '
          'final ensemble falls from 98.20% (training images) to <<acc.FINAL_majority_vote>>% (held-out images). This 5.70-point gap is the '
          'clearest evidence that evaluating on training data inflated the original claim.', new=True),
        TAB('oldnew', 'HND (2024) figures compared with the standardised re-run (accuracy, %)', ['Model', 'HND report (2024)', 'Re-run (2026), held-out test'], old_new, new=True, widths=[5.5, 5.0, 5.0], font=9),
        FIG('oldnew', 'paper_assets/figures/fig_old_vs_new.png', 'HND (2024) accuracies against the standardised re-run on held-out data', new=True, width=6.0),
        H(2, '4.11 Hypotheses H1 and H2', new=True),
        P('**H1 is not supported.** The custom CNN (<<acc.CNN_standalone>>%) is more accurate than Xception (<<acc.Xception_standalone>>%; '
          'McNemar p = <<mc.CNN.Xception.p>>) and InceptionV3 (<<acc.InceptionV3_standalone>>%; p = <<mc.CNN.InceptionV3.p>>). The same order '
          'holds when the networks are used as feature extractors for classical classifiers. With frozen backbones and ten epochs, ImageNet '
          'features did not beat a small network trained directly on MRI. **H2 is supported.** The majority vote is significantly more accurate '
          'than every member (all Holm-adjusted p ≤ <<mc.FINAL_majority.CNN.ph>>), although its gain over the best member, the CNN, is modest '
          '(2.06 points against the Keras-loader CNN figure).', new=True),
    ]
