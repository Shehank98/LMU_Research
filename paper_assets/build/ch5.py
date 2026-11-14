from engine import *

def ch5():
    di_rows = []
    for m in ['CNN', 'Xception', 'InceptionV3']:
        for meth, nm in [('gradcam', 'Grad-CAM'), ('gradcampp', 'Grad-CAM++'), ('hirescam', 'HiResCAM'), ('random', 'Random baseline')]:
            di_rows.append([m, nm, f'<<di.{m}.{meth}.deletion_auc>> (<<di.{m}.{meth}.deletion_sd>>)', f'<<di.{m}.{meth}.insertion_auc>> (<<di.{m}.{meth}.insertion_sd>>)'])
    san_rows = []
    for mdl, n in [('CNN', 6), ('Xception', 5)]:
        for i in range(n):
            san_rows.append([mdl, f'<<san.{mdl}.{i}.layer>>', f'<<san.{mdl}.{i}.rho>>', f'<<san.{mdl}.{i}.ssim>>'])
    sc_rows = [[m, '<<xai_quantitative.shortcut.%s.mean_energy_outside_head_pct>>' % m, '<<xai_quantitative.shortcut.%s.mean_area_outside_head_pct>>' % m,
                '<<xai_quantitative.shortcut.%s.mean_energy_in_rim_pct>>' % m, '<<xai_quantitative.shortcut.%s.mean_area_rim_pct>>' % m,
                '<<xai_quantitative.shortcut.%s.images_with_over_50pct_outside_or_rim>>' % m] for m in ['CNN', 'Xception', 'InceptionV3']]
    cons_rows = [['CNN vs Xception', '<<xai_quantitative.consistency.pairwise_mean_rho.rho_CNN_Xception>>', '<<xai_quantitative.consistency.pairwise_mean_iou20.iou20_CNN_Xception>>', '<<st.eaa_recomputed.pair_means.iou_cnn_xcp>>'],
                 ['CNN vs InceptionV3', '<<xai_quantitative.consistency.pairwise_mean_rho.rho_CNN_InceptionV3>>', '<<xai_quantitative.consistency.pairwise_mean_iou20.iou20_CNN_InceptionV3>>', '<<st.eaa_recomputed.pair_means.iou_cnn_inc>>'],
                 ['Xception vs InceptionV3', '<<xai_quantitative.consistency.pairwise_mean_rho.rho_Xception_InceptionV3>>', '<<xai_quantitative.consistency.pairwise_mean_iou20.iou20_Xception_InceptionV3>>', '<<st.eaa_recomputed.pair_means.iou_xcp_inc>>']]
    cls_rows = [[c, '<<recall.FINAL_majority_vote.%s>>' % k, '<<eaa.class.%s>>' % k, '<<xai_quantitative.consistency.by_true_class_mean_rho.%s>>' % k]
                for c, k in [('Glioma', 'glioma'), ('Meningioma', 'meningioma'), ('No tumour', 'notumor'), ('Pituitary', 'pituitary')]]
    return [
        H(1, 'Chapter 5 Explainable AI Analysis', new=True),
        P('This chapter is new. It asks what evidence the trained models use, whether the explanations can be trusted, and whether they help '
          'explain the errors found in Chapter 4, especially for glioma. It first shows explanations for correct and misclassified cases, then '
          'tests the explanations quantitatively, and finally discusses what they mean for clinical trust. All analyses use the saved models '
          'from the training run and the held-out test images (Section 3.11).', new=True),
        H(2, '5.1 Explanations for correctly classified images', new=True),
        P('{fig:xai_ok} shows the most confidently correct test image of each class, with Grad-CAM for the three networks and five further '
          'methods for the CNN. The pretrained networks produce broad, smooth maps, because their last feature maps are only 6 × 6 (InceptionV3) '
          'or 8 × 8 (Xception) cells at 256-pixel input. For the meningioma example, InceptionV3’s map is centred on the round, well-defined '
          'lesion, and Xception’s covers the same hemisphere. For the pituitary example, Xception concentrates on the midline region around the '
          'sella. For the no-tumour example, both spread attention over the whole brain. This is plausible, because the absence of a tumour can '
          'only be judged by looking everywhere.', new=True),
        P('The CNN’s explanations look very different. Its Grad-CAM and Grad-CAM++ maps are sharp but scattered. Much of their mass lies on '
          'the skull edge and scalp rather than on the lesion. SHAP highlights fine points near the lesion and along the skull outline, and '
          'occlusion finds that hiding a single patch rarely changes the CNN’s confident prediction. One exception is the glioma example, where '
          'the only influential patch lies on the lesion. LIME often selects large super-pixels that include background outside the head. '
          'Taken together, these maps suggest that the CNN, the most accurate single model, partly relies on global image appearance '
          '(contrast, skull shape and image framing) as well as on the lesion itself. This is a warning sign of possible shortcut learning, '
          'which Section 5.6 examines.', new=True),
        FIG('xai_ok', 'paper_assets/figures/fig_xai_correct.png', 'Explanations for the most confident correct prediction in each class (heatmaps for the predicted class; red = high importance)', new=True, width=6.3),
        H(2, '5.2 Explanations for misclassified images', new=True),
        P('{fig:xai_err} shows confidently misclassified images that represent the most common error of each class. In each case, all three '
          'networks made the same wrong prediction. Three patterns emerge.', new=True),
        B(['**Glioma predicted as meningioma** (the most common error, <<cm.FINAL_majority_vote.glioma.meningioma>> cases). Xception, '
           'InceptionV3 and occlusion all focus on the large, bright, sharply bounded lesion. The models are therefore looking at the tumour, '
           'but its appearance resembles their notion of a meningioma. The error lies in interpreting the lesion, not in finding it, which '
           'matches the known visual overlap between the two classes on single slices.',
           '**Glioma predicted as no tumour** (<<cm.FINAL_majority_vote.glioma.notumor>> cases, the most dangerous error). Xception’s and '
           'InceptionV3’s Grad-CAM maps sit mainly on the opposite, normal-looking hemisphere, and occlusion finds no single region that '
           'drives the decision. The subtle, poorly marginated lesion is largely ignored. The models are judging "normal" from the healthy '
           'tissue, which shows why heatmaps should be checked against the visible abnormality.',
           '**Meningioma predicted as pituitary and pituitary predicted as meningioma.** In the first case InceptionV3 attends to the lower '
           'image and neck region, outside the brain, and the CNN’s occlusion map is strongest along the bottom edge of the image. In the '
           'second, Xception focuses on a horizontal band through the corpus callosum rather than the sella. Both errors involve attention '
           'away from the lesion, and partly outside the brain.'], new=True),
        FIG('xai_err', 'paper_assets/figures/fig_xai_misclassified.png', 'Explanations for confidently misclassified images (the explained class is the wrong predicted class)', new=True, width=6.3),
        H(2, '5.3 Agreement between explanation methods', new=True),
        P('{fig:xai_xcp} and {fig:xai_cnn} compare methods on the same images. For Xception, Grad-CAM and HiResCAM are identical, as theory '
          'predicts for a network ending in global average pooling (Draelos and Carin, 2020). Grad-CAM++ is very similar, and the occlusion '
          'maps, which use no gradients at all, broadly agree with them (mean Spearman correlation with occlusion '
          '<<method_agreement_examples.gradcam_Xception~occlusion_Xception>> over the eight examples). For the CNN, the methods disagree far '
          'more: correlation with occlusion is only <<method_agreement_examples.gradcam_CNN~occlusion_CNN>> for Grad-CAM, '
          '<<method_agreement_examples.hirescam_CNN~occlusion_CNN>> for HiResCAM and <<method_agreement_examples.shap_CNN~occlusion_CNN>> for '
          'SHAP. For the CNN, then, the choice of explanation method changes the story told to the user, and a single heatmap should not be '
          'trusted on its own.', new=True),
        FIG('xai_xcp', 'paper_assets/figures/fig_xai_methods_xception.png', 'Five explanation methods for Xception on four of the example images', new=True, width=5.8),
        FIG('xai_cnn', 'paper_assets/figures/fig_xai_methods_cnn.png', 'Six explanation methods for the custom CNN on the same four images', new=True, width=6.0),
        H(2, '5.4 Faithfulness: deletion and insertion', new=True),
        P('{tab:di} and {fig:di} report deletion and insertion AUCs on <<xai_quantitative.n_sample>> test images, '
          '<<xai_quantitative.n_errors_in_sample>> of which were misclassified by the ensemble. A faithful map should give a lower deletion AUC '
          '(the score collapses quickly when the highlighted pixels are removed) and a higher insertion AUC than a random map. Every gradient-based method passes this test for every network: deletion AUCs are lower and insertion AUCs higher than the random baseline. The margin is largest for Xception (deletion <<di.Xception.gradcam.deletion_auc>> vs <<di.Xception.random.deletion_auc>>; insertion <<di.Xception.gradcam.insertion_auc>> vs <<di.Xception.random.insertion_auc>>) and smallest for the CNN with Grad-CAM (deletion <<di.CNN.gradcam.deletion_auc>> vs <<di.CNN.random.deletion_auc>>; insertion <<di.CNN.gradcam.insertion_auc>> vs <<di.CNN.random.insertion_auc>>). For the CNN, HiResCAM is clearly the most faithful map (deletion <<di.CNN.hirescam.deletion_auc>>, insertion <<di.CNN.hirescam.insertion_auc>>). This is what theory predicts for a Flatten + Dense head, where averaging the gradients in Grad-CAM discards spatial information (Draelos and Carin, 2020). For the pretrained networks, Grad-CAM and HiResCAM coincide, and Grad-CAM++ is slightly less faithful than Grad-CAM. The standard deviations are large, so faithfulness varies considerably from image to image. Because the random baseline also scores well for the CNN on insertion (<<di.CNN.random.insertion_auc>>), the CNN can often recover its prediction from a blurred image plus a few sharp regions. This is another sign that it relies on coarse, global cues.', new=True),
        TAB('di', 'Deletion and insertion AUC (mean and standard deviation over the evaluation sample; lower deletion and higher insertion are better)',
            ['Model', 'Method', 'Deletion AUC (SD)', 'Insertion AUC (SD)'], di_rows, new=True, widths=[3.0, 3.6, 4.0, 4.0], font=9),
        FIG('di', 'paper_assets/figures/fig_deletion_insertion.png', 'Mean deletion (top) and insertion (bottom) curves; AUCs in brackets (for Xception and InceptionV3 the Grad-CAM line lies under the identical HiResCAM line)', new=True, width=6.3),
        H(2, '5.5 Sanity check: model-parameter randomisation', new=True),
        P('If an explanation reflects what a model has learned, it should change when the learned weights are destroyed (Adebayo et al., 2018). '
          '{tab:san} and {fig:san} show the similarity between the original Grad-CAM maps and the maps after the layers were re-initialised '
          'cumulatively from the output downwards, measured on <<xai_quantitative.sanity_check.n_images>> images. Both networks pass. Randomising only the output layer already lowers the rank correlation with the original maps to <<san.CNN.first.rho>> for the CNN and <<san.Xception.first.rho>> for Xception. Randomising the convolutional layers brings it close to zero (CNN <<san.CNN.last.rho>>). For Xception, randomising block 14 makes the class score insensitive to the feature maps, so the Grad-CAM maps become empty (all zero after the ReLU) and correlation is undefined, reported as 0. SSIM does not fall to zero, because the maps share large near-zero backgrounds. The CNN’s correlation is not perfectly monotonic (it rises to <<san.CNN.1.rho>> after the second dense layer is also randomised), which reflects the instability of random maps rather than recovery of the original. The maps therefore depend on the learned weights. They are not simple edge detectors, the failure mode Adebayo et al. (2018) found for some methods.', new=True),
        TAB('san', 'Cascading randomisation: similarity of Grad-CAM maps to the original (mean over images)', ['Model', 'Randomised down to layer', 'Spearman ρ', 'SSIM'],
            san_rows, new=True, widths=[2.6, 6.0, 3.0, 3.0], font=9),
        FIG('san', 'paper_assets/figures/fig_sanity_check.png', 'Sanity check: similarity curves (top) and the CNN’s Grad-CAM for the glioma example as layers are randomised (bottom)', new=True, width=6.3),
        H(2, '5.6 Shortcut-learning check', new=True),
        P('{tab:sc} measures how much Grad-CAM mass falls outside the head or in the outer skull rim, compared with how much of the image those '
          'regions occupy. On average, <<sc.area>>% of each image lies outside the head or within the outer rim. A map that ignored location would put about the same share of its mass there. The CNN puts <<sc.CNN.mass>>% of its Grad-CAM mass in these regions, which is no better than chance, and on <<xai_quantitative.shortcut.CNN.images_with_over_50pct_outside_or_rim>> of the <<xai_quantitative.n_sample>> sampled images more than half of its attention lies outside the brain or on the skull rim. InceptionV3 is similar (<<sc.InceptionV3.mass>>%). Xception concentrates more on the brain itself (<<sc.Xception.mass>>%). Attention outside the brain is not higher for misclassified images (CNN: <<xai_quantitative.shortcut.CNN.outside_or_rim_correct_pct>>% for correct and <<xai_quantitative.shortcut.CNN.outside_or_rim_incorrect_pct>>% for incorrect predictions). This suggests a general reliance on global appearance (skull shape, contrast and framing, which differ between the source collections) rather than a defect specific to errors. It does not prove shortcut learning, but it is consistent with it (Geirhos et al., 2020), and it is the most important caution raised by the explanations.', new=True),
        TAB('sc', 'Grad-CAM mass outside the head and in the 12-pixel outer rim, against the share of image area (mean %, evaluation sample)',
            ['Model', 'Mass outside head', 'Area outside head', 'Mass in rim', 'Area of rim', 'Images > 50% outside or rim'], sc_rows, new=True, widths=[2.6, 2.4, 2.4, 2.2, 2.2, 3.2], font=9),
        H(2, '5.7 Cross-model agreement and EAA-IoU', new=True),
        P('The EAA-IoU score was computed for all 1,600 test images during the training run (eaa_iou_summary.json). Agreement is low overall: '
          'mean EAA-IoU is <<eaa.mean_eaa_all>>. Images the ensemble classified correctly have higher agreement (<<eaa.mean_eaa_correct>>) '
          'than misclassified ones (<<eaa.mean_eaa_misclassified>>; one-sided Mann–Whitney p = <<eaa.p>>), and low EAA-IoU identifies '
          'misclassified images with an AUROC of <<eaa.auroc_low_eaa_predicts_error>> ({fig:eaa}). This supports the hypothesis that attention '
          'disagreement is associated with error.', new=True),
        P('Three caveats limit its value. First, the score is driven almost entirely by the two pretrained networks. The mean IoU of the '
          'Xception–InceptionV3 pair is <<st.eaa_recomputed.pair_means.iou_xcp_inc>>, whereas any pair that includes the CNN averages about '
          '0.005, because the CNN’s maps are sharp and scattered while the others are smooth ({tab:cons}). Second, the thresholds proposed '
          'before the experiments (0.40 for an alert, 0.65 for high confidence) are unusable: every image falls below 0.40. Data-driven '
          'thresholds work better. At 0.10, <<st.eaa_recomputed.threshold_0.1.flagged_pct>>% of images are flagged and '
          '<<st.eaa_recomputed.threshold_0.1.errors_caught_pct>>% of errors are caught; at 0.05, '
          '<<st.eaa_recomputed.threshold_0.05.flagged_pct>>% are flagged and <<st.eaa_recomputed.threshold_0.05.errors_caught_pct>>% of errors '
          'are caught ({tab:thr} in Appendix C). Third, the ensemble’s own confidence (the winning share of the summed probabilities) identifies '
          'errors much better, with an AUROC of <<st.confidence_auroc>>. EAA-IoU is therefore a genuine but secondary uncertainty signal. It '
          'may add information when shown alongside confidence, but it should not replace it.', new=True),
        FIG('eaa', 'paper_assets/figures/fig_eaa_iou.png', 'EAA-IoU on all 1,600 test images: by true class, correct vs misclassified, and as an error detector', new=True, width=6.3),
        TAB('cons', 'Pairwise agreement between Grad-CAM maps (evaluation sample: Spearman ρ and top-20% IoU; whole test set: EAA pair IoU at threshold 0.5)',
            ['Pair', 'Spearman ρ', 'Top-20% IoU', 'IoU at 0.5 (all 1,600)'], cons_rows, new=True, widths=[4.4, 3.2, 3.2, 4.4], font=9),
        FIG('cons', 'paper_assets/figures/fig_consistency_shortcut.png', 'Agreement between models by class and by correctness, and the share of attention outside the brain', new=True, width=6.3),
        H(2, '5.8 Why is glioma recall lowest?', new=True),
        P('{tab:glioma} brings the evidence together by class. Glioma has the lowest recall, but it does not have the lowest attention agreement. Mean EAA-IoU is lowest for meningioma (<<eaa.class.meningioma>>) and pituitary (<<eaa.class.pituitary>>), and the rank correlation between maps is lowest for pituitary (<<xai_quantitative.consistency.by_true_class_mean_rho.pituitary>>). Only no tumour stands out, with clearly higher agreement on both measures. The expectation from the proposal, that glioma would show the most disagreement, is therefore not supported. Agreement does track correctness within the sample, with a mean rank correlation of <<xai_quantitative.consistency.correct_mean_rho>> for correct and <<xai_quantitative.consistency.incorrect_mean_rho>> for misclassified images (Mann–Whitney p = <<cons.p>>), matching the EAA-IoU result on the full test set. The examples in Section 5.2 suggest a better explanation for the glioma problem. In the most common error the models agree and do look at the lesion, but read a well-defined, bright lesion as meningioma. In the most dangerous error they miss a subtle, infiltrative lesion and base a "no tumour" decision on the normal hemisphere. Neither failure can be detected by disagreement, which explains why EAA-IoU catches some errors but not all.', new=True),
        TAB('glioma', 'Class-level summary: ensemble recall and explanation agreement', ['Class', 'Ensemble recall (%)', 'Mean EAA-IoU (all images)', 'Mean Spearman ρ (sample)'],
            cls_rows, new=True, widths=[3.4, 3.6, 4.0, 4.0], font=9),
        H(2, '5.9 Explaining the feature-level ensemble with SHAP', new=True),
        P('The classical stage can be explained exactly. TreeSHAP was applied to the Random Forest that classifies the CNN’s 256 dense-layer '
          'features ({fig:shap_rf}), using 200 test images (50 per class). Re-running the forest on the test set reproduced its logged '
          'accuracy of <<shap_rf.rf_accuracy_recomputed_pct>>%. The attributions are concentrated: <<shap_rf.features_for_80pct>> of the 256 '
          'features carry 80% of the total attribution, and <<shap_rf.dead_or_inactive_features>> features are never active on the test set. '
          'The most important features are partly class-specific (unit <<shap_rf.class_specific_top_unit.No tumour>> for no tumour, unit '
          '<<shap_rf.class_specific_top_unit.Pituitary>> for pituitary, units <<shap_rf.class_specific_top_unit.Glioma>> and '
          '<<shap_rf.class_specific_top_unit.Meningioma>> for glioma and meningioma). Glioma and meningioma share several of the same top '
          'features, which is consistent with their confusion. Because the features are learned, SHAP at this level shows how the decision '
          'is distributed but not what the features mean visually. Linking features back to image regions would require the image-level '
          'methods above.', new=True),
        FIG('shap_rf', 'paper_assets/figures/fig_shap_rf.png', 'TreeSHAP for the Random Forest on CNN features: most important units by class (left) and concentration of attribution (right)', new=True, width=6.2),
        H(2, '5.10 Clinical trust and the limits of saliency maps', new=True),
        P('The analysis supports a cautious position. On the positive side, the explanations are not arbitrary. They beat random maps on '
          'faithfulness tests and they depend on the trained weights, so they genuinely reflect model behaviour. They also explain the main '
          'glioma error as a misreading of a visible lesion rather than a failure to find it. On the negative side, the three networks rarely '
          'agree on where to look. The most accurate model places much of its attention on the skull and image border. Some errors coincide '
          'with attention outside the brain. Explanation methods also disagree with each other, especially for the CNN. This agrees with '
          'benchmarks showing that saliency maps are unreliable localisers in medical images (Arun et al., 2021; Saporta et al., 2022) and with '
          'the warning that explanations can create false confidence (Ghassemi, Oakden-Rayner and Beam, 2021).', new=True),
        P('Three implications follow for a clinical demonstrator. Heatmaps should be shown as "where the model looked", never as a tumour '
          'outline. Several methods or models should be shown together, so that disagreement is visible. And an explanation that falls '
          'outside the brain, or away from an obvious lesion, should itself be treated as a reason for closer review. Inherently interpretable '
          'or attention-constrained models (Rudin, 2019), and evaluation against radiologist annotations, would be needed before explanations '
          'could support clinical decisions.', new=True),
        H(2, '5.11 Hypothesis H3', new=True),
        P('H3 is **partly supported**. The faithfulness part holds: for all three networks, deletion AUC is lower and insertion AUC higher than for random maps. The model-dependence part holds: maps change substantially once the trained weights are randomised. The agreement part holds statistically but is weak in practice: lower agreement is significantly associated with misclassification (EAA-IoU AUROC <<eaa.auroc_low_eaa_predicts_error>>; Spearman-based p = <<cons.p>>), but it is a weaker warning sign than ensemble confidence (AUROC <<st.confidence_auroc>>), it does not single out glioma, and it is dominated by the two pretrained networks. Beyond H3, the explanations revealed that the most accurate model places about half of its attention outside the brain, which is a substantive finding about model behaviour that accuracy alone did not show.', new=True),
    ]
