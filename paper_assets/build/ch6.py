from engine import *

PRED = [("CNN", "CNN", "cnn"), ("InceptionV3", "InceptionV3", "inceptionv3"), ("Xception", "Xception", "xception"),
        ("CNN features + soft vote", "CNN_features", "cnn_feat_softvote"),
        ("InceptionV3 features + soft vote", "InceptionV3_features", "inceptionv3_feat_softvote"),
        ("Xception features + soft vote", "Xception_features", "xception_feat_softvote"),
        ("Six-model majority vote", "final_majority", "final_majority"),
        ("Six-model confidence-weighted vote", "final_confidence_weighted", "final_confidence_weighted")]


def ch6():
    acc_rows = [[n, f"<<clean.baseline.{k}.full.acc>>", f"<<clean.improved.{k}.full.acc>>", f"<<clean.baseline.{k}.clean.acc>>",
                 f"<<clean.improved.{k}.clean.acc>> (<<clean.improved.{k}.clean.ci95>>)"] for n, k, _ in PRED]
    cls_rows = [[c, f"<<iprecision.final_majority.{k}>>", f"<<irecall.final_majority.{k}>>", f"<<if1c.final_majority.{k}>>",
                 f"<<clean.improved.final_majority.clean.recall.{k}>>"]
                for c, k in [("Glioma", "glioma"), ("Meningioma", "meningioma"), ("No tumour", "notumor"), ("Pituitary", "pituitary")]]
    mc_rows = [[t, f"<<mci.{t.replace(' ', '_').replace('(', '').replace(')', '')}.n01>>", f"<<mci.{t.replace(' ', '_').replace('(', '').replace(')', '')}.n10>>",
                f"<<mci.{t.replace(' ', '_').replace('(', '').replace(')', '')}.p>>", f"<<mci.{t.replace(' ', '_').replace('(', '').replace(')', '')}.ph>>"]
               for t in ["improved vs baseline majority (full)", "improved vs baseline majority (clean)"]
               + [f"improved majority vs improved {k} (full)" for k in ["CNN", "InceptionV3", "Xception", "CNN_features", "InceptionV3_features", "Xception_features"]]]
    di_rows = [[m, nm, f"<<di.{m}.{meth}.deletion_auc>> / <<di.{m}.{meth}.insertion_auc>>", f"<<imp.di.{m}.{meth}.deletion_auc>> / <<imp.di.{m}.{meth}.insertion_auc>>"]
               for m in ["CNN", "Xception", "InceptionV3"] for meth, nm in [("gradcam", "Grad-CAM"), ("random", "Random baseline")]]
    sc_rows = [[m, f"<<sc.{m}.mass>>", f"<<imp.sc.{m}.mass>>", f"<<sinv.{m}.outside_only_recall_by_class.notumor>>",
                f"<<imp.shortcut_investigation.{m}.outside_only_recall_by_class.notumor>>"] for m in ["CNN", "Xception", "InceptionV3"]]
    return [
        H(1, "Chapter 6 Improved Configuration"),
        P("This chapter evaluates the improved configuration described in Section 3.6: images cropped to the head, augmented training "
          "images, fine-tuned pretrained backbones and a custom CNN with batch normalisation and global average pooling. It reports accuracy "
          "on the full test set and on the duplicate-free subset, compares the result with the baseline statistically, and repeats the "
          "explainability checks of Chapter 5 to see whether the problems found there were reduced."),
        H(2, "6.1 Training behaviour"),
        P("{fig:train_imp} shows the training curves. Both pretrained networks improve quickly once the upper blocks are unfrozen (dotted "
          "line), and their validation accuracy reaches <<ihist.inceptionv3.best_val_acc>> (InceptionV3) and <<ihist.xception.best_val_acc>> "
          "(Xception). The first run of the improved CNN under-fitted: its learning rate was reduced after only two epochs without "
          "improvement, the noisy validation loss triggered this repeatedly, and training stopped with a training accuracy of 0.87. It was "
          "therefore re-run with more patience (learning rate reduced after four epochs, early stopping after eight, at most 45 epochs). The "
          "decision was based on the training curves only; the test set was not used. The re-run reached a best validation accuracy of "
          "<<ihist.cnn.best_val_acc>>."),
        FIG("train_imp", "paper_assets/figures/fig_training_improved.png", "Training and validation curves of the improved models (dotted line: start of fine-tuning)", width=6.2),
        H(2, "6.2 Accuracy"),
        P("{tab:imp_acc} and {fig:bvi} compare the two configurations. <<IMP_ACC_TEXT>>"),
        TAB("imp_acc", "Accuracy (%) of the baseline and improved configurations on the full test set and on the duplicate-free subset (95% bootstrap interval for the improved clean result)",
            ["Predictor", "Baseline, full", "Improved, full", "Baseline, clean", "Improved, clean"], acc_rows, widths=[4.6, 2.2, 2.2, 2.4, 4.0], font=8.5),
        FIG("bvi", "paper_assets/figures/fig_baseline_vs_improved.png", "Accuracy of every predictor in the baseline and improved configurations", width=6.3),
        H(2, "6.3 The improved ensemble"),
        P("{tab:imp_cls} gives the per-class results of the improved majority vote and {fig:cm_imp} the confusion matrices. <<IMP_CLASS_TEXT>>"),
        TAB("imp_cls", "Per-class results of the improved six-model majority vote", ["Class", "Precision (%)", "Recall (%)", "F1-score", "Recall on clean subset (%)"],
            cls_rows, widths=[3.2, 2.8, 2.6, 2.4, 4.0], font=9),
        FIG("cm_imp", "paper_assets/figures/fig_cm_improved.png", "Row-normalised confusion matrices of the improved models and ensembles (G = glioma, M = meningioma, N = no tumour, P = pituitary)", width=6.3),
        H(2, "6.4 Statistical comparison"),
        P("{tab:imp_mc} lists exact McNemar tests. <<IMP_MC_TEXT>>"),
        TAB("imp_mc", "Exact McNemar tests for the improved configuration (Holm-adjusted over the eight comparisons)",
            ["Comparison", "n01 (first right only)", "n10 (second right only)", "p (exact)", "p (Holm)"], mc_rows, widths=[6.2, 2.2, 2.2, 2.2, 2.2], font=8),
        H(2, "6.5 Explanations after the changes"),
        P("The explainability checks of Chapter 5 were repeated on the improved models with the same sample design. {fig:xai_imp} shows the "
          "examples, {tab:imp_di} the faithfulness results and {tab:imp_sc} the attention outside the brain. <<IMP_XAI_TEXT>>"),
        FIG("xai_imp", "paper_assets/figures_improved/fig_xai_correct.png", "Explanations of the improved models for the most confident correct prediction in each class", width=6.3),
        FIG("xai_imp_err", "paper_assets/figures_improved/fig_xai_misclassified.png", "Explanations of the improved models for confidently misclassified images", width=6.3),
        TAB("imp_di", "Deletion / insertion AUC of Grad-CAM and of a random map, baseline against improved models",
            ["Model", "Map", "Baseline (deletion / insertion)", "Improved (deletion / insertion)"], di_rows, widths=[2.8, 3.2, 4.6, 4.6], font=9),
        TAB("imp_sc", "Attention outside the head or in the skull rim (% of Grad-CAM mass) and no-tumour recall from the background alone (%)",
            ["Model", "Mass outside, baseline", "Mass outside, improved", "No-tumour recall from background, baseline", "Same, improved"], sc_rows,
            widths=[2.6, 2.8, 2.8, 4.0, 3.0], font=8.5),
        P("<<IMP_EAA_TEXT>>"),
        H(2, "6.6 Hypotheses H1 and H4"),
        P("<<IMP_HYP_TEXT>>"),
    ]
