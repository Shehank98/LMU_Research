from engine import *


def ch7():
    lit = [["Minz and Mahobiya (2017)", "GLCM texture + AdaBoost", "Brain MRI (tumour types)", "89.90"],
           ["Cheng et al. (2015)", "Region augmentation + ring partition, bag-of-words", "figshare CE-T1, 3 classes", "91.28"],
           ["Zhou et al. (2019)", "DenseNet features + recurrent network", "3D MRI slices", "92.13"],
           ["Deepak and Ameer (2019)", "GoogLeNet features + SVM / KNN", "figshare, 3 classes, patient-level 5-fold CV", "98 (mean)"],
           ["Rehman et al. (2020)", "Fine-tuned VGG16", "figshare, 3 classes", "98.69"],
           ["Talo et al. (2019)", "ResNet34 transfer learning", "613 images, normal vs abnormal, 5-fold CV", "100"],
           ["This study, baseline", "Six-predictor ensemble, frozen backbones", "Nickparvar (n.d.), 4 classes, full / duplicate-free test",
            "<<clean.baseline.final_majority.full.acc>> / <<clean.baseline.final_majority.clean.acc>>"],
           ["This study, improved", "Six-predictor ensemble, cropping and fine-tuning", "Same",
            "<<clean.improved.final_majority.full.acc>> / <<clean.improved.final_majority.clean.acc>>"]]
    return [
        H(1, "Chapter 7 Discussion"),
        P("This chapter interprets the results, compares them with published work, evaluates the strengths and weaknesses of the data "
          "collection, analysis and methodology, and ends with a plain-language summary for clinical stakeholders."),
        H(2, "7.1 Principal findings"),
        P("<<DISC_MAIN_TEXT>>"),
        H(2, "7.2 Comparison with the literature"),
        P("{tab:lit} places the results beside figures reported in the literature. Most published studies used other datasets (often the "
          "three-class figshare set), sometimes with cross-validation and fine-tuning, and few report whether their test images also appear in "
          "the training data. The figures are therefore not directly comparable; the table shows the range rather than a ranking. The improved "
          "configuration is in the same range as the fine-tuned transfer-learning studies (Rehman et al., 2020; Deepak and Ameer, 2019), which "
          "is consistent with fine-tuning being the largest single change. Kang, Ullah and Gwak (2021) found that deep features markedly "
          "improve classical classifiers, and the baseline supports this: the SVM gained 7.44 points when moved from raw pixels "
          "(<<acc.raw_svm>>%) to CNN features (<<acc.cnn_feat_svm>>%). The explainability findings agree with medical-imaging benchmarks that "
          "found saliency maps to be unreliable localisers (Arun et al., 2021; Saporta et al., 2022), and the background cue is a concrete "
          "example of the shortcut learning described by Geirhos et al. (2020)."),
        TAB("lit", "Reported accuracy in related studies (as reported by the authors; datasets and protocols differ)", ["Study", "Method", "Data and validation", "Accuracy (%)"],
            lit, widths=[3.4, 4.4, 5.0, 2.4], font=8.5),
        H(2, "7.3 Merits, limitations and pitfalls of the data collection and analysis"),
        P("Using a public, labelled dataset made the study feasible, cheap and free of patient-privacy risk, and it allows others to reproduce "
          "the work. The same choice brings pitfalls, and this study measured two of them directly. About a fifth of the official test images "
          "duplicate training images, which inflates accuracy, and the classes come from different collections, so image size and background "
          "carry class information. Neither problem is visible from accuracy alone. In addition, about a quarter of the meningioma test images "
          "are augmented copies, labels come from the original collections and cannot be checked here, and the slices carry no patient "
          "identifiers, so a patient-level split is impossible."),
        P("The analysis has clear strengths. One held-out test set is used for every model; results are logged as data; uncertainty is "
          "quantified with Wilson and bootstrap intervals; models are compared with paired McNemar tests with Holm correction; and the "
          "explanations are tested with deletion and insertion curves, randomisation and agreement measures rather than only shown. Its "
          "limitations are equally clear: one split and one training run per model, so run-to-run variation is unmeasured; few "
          "hyperparameters tuned; and explanation metrics computed on a 160-image sample without tumour masks."),
        H(2, "7.4 Critical evaluation of the methodology"),
        P("The positivist, deductive design suited questions about measurable differences between models, and stating hypotheses in advance "
          "made it possible to reject H1 for the baseline rather than reinterpret it. Treating explanations as evidence to be tested, in line "
          "with the critical-realist stance, turned out to be the most productive methodological choice: it exposed a dataset cue and led "
          "directly to the improved configuration. The weakest part of the design is that the second configuration changed several things at "
          "once (cropping, augmentation, fine-tuning and the CNN head), so the gain cannot be attributed to any single change. An ablation that "
          "adds one change at a time would answer that, at the cost of several more training runs."),
        P("Several alternatives were considered. k-fold cross-validation would give more stable estimates and allow tests across folds "
          "(Dietterich, 1998) but would have multiplied the training time on the available hardware. A patient-level split would be more "
          "rigorous but is impossible without patient identifiers. Stacking with a meta-learner (Kakon et al., 2025) could learn better weights "
          "than a fixed vote, but needs a separate validation set to avoid leakage. Inherently interpretable models (Rudin, 2019) or radiologist "
          "masks would allow stronger claims about explanations. The qualitative strand planned in the proposal was not carried out, which is "
          "the largest methodological gap and is addressed in the recommendations."),
        H(2, "7.5 Threats to validity"),
        B(["Construct validity: accuracy on balanced classes does not capture clinical cost; missing a glioma (a false \"no tumour\") is worse than confusing two tumour types, so per-class recall is reported alongside accuracy.",
           "Internal validity: duplicated and augmented test images, possible patient overlap between folders and no repeated runs. The duplicate-free subset addresses the first.",
           "External validity: one public dataset of 2D slices; performance on scans from other hospitals and scanners is unknown (Zech et al., 2018).",
           "Conclusion validity: paired tests were Holm-corrected and intervals reported; small differences are not treated as real."]),
        H(2, "7.6 Summary for clinical stakeholders"),
        P("The following summary is written for radiologists, neurosurgeons and hospital managers."),
        B(["What it does. The system reads one MRI slice and suggests one of four labels: glioma, meningioma, pituitary tumour or no tumour. It also shows heatmaps of the image regions that most influenced the suggestion.",
           "How well it works. On 1,600 images it had never seen, the improved system was right <<clean.improved.final_majority.full.acc>>% of the time, and <<clean.improved.final_majority.clean.acc>>% on the images that have no copy in its training data. It recognised every no-tumour image and about 99% of meningiomas and pituitary tumours. It was weakest on gliomas: about 15 in 100 were mislabelled, usually as meningioma, and 19 of 400 were wrongly called "no tumour".",
           "When to trust it less. Be cautious when its confidence is low or when the six internal models disagree; most of its mistakes happen in these cases. The demonstrator raises an alert when the models' heatmaps overlap very little.",
           "What the heatmaps mean. A heatmap shows where a model looked, not where the tumour is. A heatmap that does not overlap the visible lesion, or that sits on the skull or background, is a warning sign.",
           "What it is not. It is a research prototype. It has not been tested on images from your hospital, does not use clinical history and cannot grade tumours. It must not be used for diagnosis without radiologist review and proper clinical validation."]),
        P("A prototype web demonstrator (webapp/) presents these elements for an uploaded image: the predicted class, the votes of the six "
          "predictors, a heatmap per network and an agreement-based alert, calibrated from the test results (alert below an EAA-IoU of 0.10, "
          "strong agreement from 0.20)."),
    ]


def ch8():
    ro = [["RO1 Custom CNN", "Met", "Baseline CNN <<acc.CNN_standalone>>%, but part of this came from the background; on cropped images the CNN reached <<clean.improved.CNN.full.acc>>%."],
          ["RO2 Pretrained networks", "Met", "Frozen: <<acc.Xception_standalone>>% (Xception), <<acc.InceptionV3_standalone>>% (InceptionV3). Fine-tuned: <<clean.improved.Xception.full.acc>>% and <<clean.improved.InceptionV3.full.acc>>%."],
          ["RO3 Classical classifiers", "Met", "Deep features improve every classical classifier (baseline SVM <<acc.raw_svm>>% on pixels, <<acc.cnn_feat_svm>>% on CNN features)."],
          ["RO4 Ensemble", "Met", "Majority vote <<clean.baseline.final_majority.full.acc>>% (baseline) and <<clean.improved.final_majority.full.acc>>% (improved); <<clean.improved.final_majority.clean.acc>>% on the duplicate-free subset."],
          ["RO5 Explanations", "Met", "Six methods applied to correct and misclassified cases of every class, for both configurations."],
          ["RO6 Evaluating and using explanations", "Met", "Faithfulness, sanity, agreement and background checks; the background cue led to the improved configuration."],
          ["Clinician feedback (proposal)", "Not met", "No responses collected; questionnaire kept as a proposed instrument (Appendix D)."]]
    return [
        H(1, "Chapter 8 Conclusions and Recommendations"),
        P("This chapter states how far the objectives were met, answers the research questions, and makes recommendations."),
        H(2, "8.1 Extent to which the objectives were met"),
        P("{tab:ro} summarises, objective by objective, how far the outcomes meet the objectives set in Section 1.4."),
        TAB("ro", "Achievement of the research objectives", ["Objective", "Status", "Evidence"], ro, widths=[4.0, 2.2, 9.2], font=9),
        H(2, "8.2 Answers to the research questions"),
        P("RQ1. It depends on fine-tuning. Used as frozen feature extractors, InceptionV3 and Xception were significantly less accurate than the custom CNN. Fine-tuned on cropped images they became the strongest models (<<clean.improved.InceptionV3.full.acc>>% and <<clean.improved.Xception.full.acc>>%), while the CNN, deprived of the background cue, fell to <<clean.improved.CNN.full.acc>>%."),
        P("RQ2. In the baseline, yes: the six-model majority vote (<<clean.baseline.final_majority.full.acc>>%) was significantly better than every member. In the improved configuration the ensemble (<<clean.improved.final_majority.full.acc>>%) is the most accurate predictor but is not significantly better than fine-tuned InceptionV3 alone. Confidence weighting did not improve accuracy in either configuration, but it provides a per-image confidence score that flags errors well."),
        P("RQ3. Partly. The explanations are more faithful than random maps and depend on the model weights, so they reflect real model "
          "behaviour. They differ markedly between architectures, and agreement between them is a weaker error signal than ensemble "
          "confidence. Their most useful result was diagnostic: they revealed that the baseline relied on the image background."),
        P("<<RQ4_TEXT>>"),
        H(2, "8.3 Recommendations"),
        NL(["Report accuracy on held-out data and, for public datasets, check the test set for duplicates of training images and report a duplicate-free figure as well.",
            "Crop images to the anatomy and inspect explanations before trusting accuracy: in this study a simple background check explained a large part of the baseline's performance.",
            "Use ensemble confidence as the main trigger for human review, with attention agreement as a secondary cue; the web demonstrator's thresholds (0.10 and 0.20) were set from the observed distribution and should be re-checked on new data.",
            "Run an ablation that adds cropping, augmentation and fine-tuning one at a time, with repeated runs, to attribute the improvement.",
            "Validate on an external dataset from another source and run the Appendix D questionnaire with radiologists once ethical approval is in place.",
            "Obtain even a small set of tumour masks so that explanations can be checked against anatomy."]),
        H(2, "8.4 Limitations"),
        P("The main limitations are: one public dataset of 2D slices without patient identifiers; duplicated and augmented test images; one "
          "training run per model without cross-validation; several changes introduced together in the improved configuration; explanation "
          "metrics on a 160-image sample; no pixel-level ground truth; and no clinician evaluation. Ensembles also cost more to run, because "
          "six predictors are evaluated per image."),
        H(2, "8.5 Future work"),
        P("Future work should (i) compare post-hoc saliency with inherently interpretable or attention-based models (Rudin, 2019); (ii) explore "
          "stacking and calibrated probability averaging; (iii) move from 2D slices to 3D volumes and multi-sequence MRI; (iv) add tumour "
          "localisation as a second task; and (v) study with clinicians whether explanations and confidence improve decisions or only increase "
          "trust (Ghassemi, Oakden-Rayner and Beam, 2021)."),
    ]


def ch9():
    return [
        H(1, "Chapter 9 Reflection"),
        P("This chapter reflects on how effective the research methods were, which alternatives would have served better, and what I learned."),
        H(2, "9.1 Effectiveness of the research methods"),
        P("The quantitative experimental method answered RQ1 and RQ2 decisively because every model was evaluated on the same untouched test "
          "images and compared with paired tests. Logging every result as data was the most effective practical decision: it allowed the "
          "saved models to be re-checked, the duplicate test images to be found, and every number in this report to be regenerated and "
          "cross-checked by script. The literature on evaluating explanations was equally valuable, because the deletion, insertion and "
          "randomisation tests turned the heatmaps from illustrations into evidence, and the background check changed the direction of the "
          "project. The method was least effective on the clinical side: without clinician input, questions about usability and trust remain open."),
        H(2, "9.2 Alternative methodologies and lessons learnt"),
        P("With hindsight, three alternatives would have strengthened the work: a patient-level, cross-validated design; a small qualitative "
          "study with two or three radiologists; and an ablation study planned from the start. The main lessons are practical: check where "
          "every number comes from; check the data for duplicates and source differences before training; be suspicious of results that look "
          "better than the literature; keep pre-processing identical in training and evaluation code; and treat an explanation as a hypothesis "
          "to be tested."),
        H(2, "9.3 Personal reflection"),
        H(3, "9.3.1 Benefits for the researcher"),
        P("The project developed my skills in deep learning, classical machine learning, statistics and explainable AI. More importantly, it "
          "changed how I think about evidence. Seeing that a model could recognise the no-tumour class from the background alone showed me "
          "that a high accuracy figure can hide a weak model, and that the most useful experiments are often the ones designed to break a "
          "result. Writing for clinicians also taught me to separate what a model does from what it appears to do."),
        H(3, "9.3.2 Benefits for industry and healthcare organisations"),
        P("For healthcare organisations, the project offers a transparent template for evaluating imaging AI before adoption: held-out and "
          "duplicate-free testing, intervals, paired tests, and stress tests of explanations. It shows why accuracy claims on public data "
          "should be checked against external data, and why explanation maps need their own validation before they are shown to clinicians."),
    ]
