from engine import *

def ch6():
    lit = [['Minz and Mahobiya (2017)', 'GLCM texture + AdaBoost', 'Brain MRI (tumour types)', '89.90'],
           ['Cheng et al. (2015)', 'Region augmentation + ring partition, bag-of-words', 'figshare CE-T1, 3 classes', '91.28'],
           ['Zhou et al. (2019)', 'DenseNet features + recurrent network', '3D MRI slices', '92.13'],
           ['Deepak and Ameer (2019)', 'GoogLeNet features + SVM / KNN', 'figshare, 3 classes, patient-level 5-fold CV', '98 (mean)'],
           ['Rehman et al. (2020)', 'Fine-tuned VGG16', 'figshare, 3 classes', '98.69'],
           ['Talo et al. (2019)', 'ResNet34 transfer learning', '613 images, normal vs abnormal, 5-fold CV', '100'],
           ['This study', 'Six-predictor hybrid ensemble (majority vote)', 'Nickparvar (n.d.), 4 classes, fixed held-out test (1,600 images)', '<<acc.FINAL_majority_vote>>']]
    return [
        H(1, 'Chapter 6 Discussion', new=True),
        P('This chapter interprets the results, compares them with published work, evaluates the strengths and weaknesses of the data '
          'collection, analysis and methodology, and ends with a plain-language summary for clinical stakeholders.', new=True),
        H(2, '6.1 Principal findings', new=True),
        P('Four findings stand out. First, evaluated honestly on held-out data, the hybrid ensemble reaches <<acc.FINAL_majority_vote>>%, not '
          '98.20%; the original figure was a training-set score. Second, a small CNN trained from scratch beat ImageNet networks with frozen '
          'backbones, both as a classifier and as a feature extractor. This contradicts the original hypothesis but agrees with work showing '
          'that compact task-specific networks can match larger pretrained ones on this kind of data (Badža and Barjaktarović, 2020). Third, '
          'the ensemble’s advantage is real but modest. It comes from correcting errors made by a minority of members, and it cannot recover '
          'the <<st.all_members_wrong>> images on which all members fail. Fourth, the explanations are informative but not reassuring. They '
          'are measurably faithful and pass a sanity check, yet the three networks often look at different places, sometimes outside the brain, '
          'and their disagreement predicts errors less well than the ensemble’s own confidence (Chapter 5).', new=True),
        H(2, '6.2 Comparison with the literature', new=True),
        P('{tab:lit} sets the result beside figures reported in the literature. Most published accuracies are higher. They were obtained on '
          'other datasets (most often the three-class figshare set), sometimes with cross-validation, sometimes with fine-tuning, and in some '
          'cases without a strictly separate test set. These figures are therefore not directly comparable; the point of the table is to show '
          'the range, not to rank studies. The re-run’s performance is in the same region as hand-crafted and early deep approaches. It is '
          'below the best transfer-learning results, which fine-tuned their backbones (Rehman et al., 2020). This is consistent with the '
          'present models having frozen backbones and no augmentation. Kang, Ullah and Gwak (2021) also found that deep features markedly '
          'improve classical classifiers, which the 7.44-point gain of the SVM when moving from raw pixels to CNN features (<<acc.raw_svm>>% to <<acc.cnn_feat_svm>>%) supports. The explainability '
          'findings agree with medical-imaging benchmarks that found saliency maps to be inconsistent localisers (Arun et al., 2021; Saporta '
          'et al., 2022).', new=True),
        TAB('lit', 'Reported accuracy in related studies (as reported by the authors; datasets and protocols differ)', ['Study', 'Method', 'Data and validation', 'Accuracy (%)'],
            lit, new=True, widths=[3.6, 4.4, 5.0, 2.2], font=9),
        H(2, '6.3 Merits, limitations and pitfalls of the data collection and analysis', new=True),
        P('**Data collection.** Using a public, labelled dataset made the study feasible, cheap and free of patient-privacy risk, and it allows '
          'others to reproduce the work. The same choice brings pitfalls. The images are two-dimensional slices without patient identifiers, so '
          'slices from one patient may appear in both training and test folders, which would inflate accuracy. The data mix several sources, '
          'with all no-tumour images from one source, which invites shortcut learning (Geirhos et al., 2020). About a quarter of the meningioma '
          'test images are augmented copies, which makes that class’s test less independent. Labels come from the original collections and '
          'cannot be checked here.', new=True),
        P('**Analysis.** The analysis has clear strengths. A single held-out test set is used for every model. Results are logged as data rather '
          'than screenshots. Uncertainty is quantified with Wilson and bootstrap intervals, and model differences are tested with paired '
          'McNemar tests with Holm correction rather than by eye. The explainability claims are tested with deletion/insertion, randomisation '
          'and agreement measures, not only shown as pictures. The limitations are equally clear. There is one split and one training run per '
          'model, so variation between runs is unmeasured. Hyperparameters were not tuned. The explainability metrics use a sample of 160 '
          'images, and without tumour masks, localisation could only be judged qualitatively. A further pitfall, visible in the HND study, is '
          'that a pipeline which reloads data in a different way (training folder versus test folder, or Keras versus OpenCV resizing) can '
          'silently change the evaluation. Logging per-image predictions was what made this audit possible.', new=True),
        H(2, '6.4 Critical evaluation of the methodology', new=True),
        P('The positivist, deductive design suited the research questions, which are about measurable differences between models. Stating '
          'hypotheses in advance made it possible to reject H1 cleanly rather than reinterpret it. The experimental design also deliberately '
          'favoured fairness over peak accuracy: every model shared the same split, preprocessing, batch size and seed. This makes the '
          'comparison credible, but it probably handicaps the pretrained networks. With frozen backbones, ten epochs and 256-pixel inputs '
          '(instead of their native 299 pixels), InceptionV3 and Xception were not given the conditions under which transfer learning usually '
          'excels. A fairer test of RQ1 would fine-tune the upper blocks with a lower learning rate and use augmentation.', new=True),
        P('Several alternatives were considered. k-fold cross-validation would give more stable estimates and allow tests across folds '
          '(Dietterich, 1998), but it would have multiplied the free-GPU time several-fold. A patient-level split would be more rigorous, but it '
          'is impossible without patient identifiers. For the ensemble, stacking with a meta-learner (as in Kakon et al., 2025) could learn '
          'better weights than a fixed vote, at the cost of needing a separate validation set to avoid leakage. For explainability, '
          'inherently interpretable models (Rudin, 2019) or radiologist-annotated masks would allow stronger claims. The mixed-methods design '
          'promised in the proposal was not delivered: no clinician data were collected, which is the largest methodological shortfall of the '
          'project and is addressed in the recommendations.', new=True),
        H(2, '6.5 Threats to validity', new=True),
        B(['**Construct validity** – accuracy on balanced classes does not capture clinical cost; missing a glioma (false "no tumour") is worse than confusing two tumour types. Per-class recall and the confusion matrices are therefore reported alongside accuracy.',
           '**Internal validity** – possible patient overlap between folders, augmented meningioma test images, and the lack of repeated runs.',
           '**External validity** – one public dataset of 2D slices; performance on scans from other hospitals and scanners is unknown (Zech et al., 2018).',
           '**Conclusion validity** – ten paired tests were Holm-corrected, and intervals were reported. Small differences (for example, majority vs confidence-weighted voting) are not treated as real.'], new=True),
        H(2, '6.6 Summary for clinical stakeholders', new=True),
        P('The following summary is written for radiologists, neurosurgeons and hospital managers who will not read the technical chapters.', new=True),
        B(['**What it does.** The system reads one MRI slice and suggests one of four labels: glioma, meningioma, pituitary tumour or no tumour. It also shows a heatmap of the image regions that most influenced the suggestion.',
           '**How well it works.** On 1,600 images it had never seen, it was right 92.5% of the time. It never called a no-tumour image a tumour, and it was right for 97% of pituitary and 95.5% of meningioma images. It was weakest on gliomas: 22.5% (90 of 400) were mislabelled, usually as meningioma, and 23 of 400 were wrongly called "no tumour".',
           '**When to trust it less.** Be cautious when its confidence is low, or when the six internal models disagree. Most of its mistakes happen in these cases. A low score is a reason for a careful human read, not proof of error.',
           '**What the heatmaps mean.** A heatmap shows where the model looked, not where the tumour is. Sometimes the heatmap sits on the tumour; sometimes it sits on the skull edge or outside the brain. A heatmap that does not overlap the visible lesion is a warning sign.',
           '**What it is not.** It is a research prototype. It has not been tested on images from your hospital, does not use clinical history and cannot grade tumours. It must not be used for diagnosis without radiologist review and proper clinical validation.'], new=True),
        P('A prototype web demonstrator in the project repository (webapp/) presents these elements for an uploaded image: the predicted class, '
          'the six members’ votes, per-model heatmaps and an agreement-based alert. Its alert threshold is still the value proposed before the '
          'experiments (EAA-IoU below 0.40). Chapter 5 shows that this would flag almost every image, so it should be recalibrated before any '
          'user study (Section 7.3).', new=True),
    ]

def ch7():
    ro = [['RO1 Custom CNN', 'Met', 'Best single model: <<acc.CNN_standalone>>% test accuracy; best feature source for classical classifiers.'],
          ['RO2 Pretrained networks vs CNN', 'Met (hypothesis rejected)', 'Xception <<acc.Xception_standalone>>%, InceptionV3 <<acc.InceptionV3_standalone>>%; both significantly below the CNN.'],
          ['RO3 Classical classifiers', 'Met', 'Raw pixels: best RF <<acc.raw_rf>>%. Deep features improve every classifier (CNN features + RF <<acc.cnn_feat_rf>>%).'],
          ['RO4 Ensemble', 'Met', 'Majority vote <<acc.FINAL_majority_vote>>%, significantly better than every member; confidence weighting not better.'],
          ['RO5 Explanations', 'Met', 'Six methods applied to correct and misclassified cases of every class (Chapter 5).'],
          ['RO6 Evaluating explanations', 'Largely met', 'Faithfulness, sanity check, agreement and shortcut checks done; localisation only qualitative (no masks).'],
          ['Clinician feedback (proposal)', 'Not met', 'No responses collected; questionnaire kept as a proposed instrument (Appendix D).']]
    return [
        H(1, 'Chapter 7 Conclusions and Recommendations'),
        P('This chapter states how far the objectives were met, answers the research questions, and makes recommendations for improving the system and for future research.'),
        H(2, '7.1 Extent to which the objectives were met', new=True),
        P('{tab:ro} summarises, objective by objective, how far the outcomes meet the objectives set in Section 1.4.', new=True),
        TAB('ro', 'Achievement of the research objectives', ['Objective', 'Status', 'Evidence'], ro, new=True, widths=[4.0, 2.8, 8.6], font=9),
        H(2, '7.2 Answers to the research questions', new=True),
        P('**RQ1.** No. Under identical conditions, the frozen-backbone InceptionV3 and Xception models did not outperform a custom CNN, either '
          'as classifiers or as feature extractors. H1 is rejected, with the caveat that the backbones were not fine-tuned.', new=True),
        P('**RQ2.** Yes for ensembling, no for confidence weighting. The six-model majority vote is the most accurate configuration '
          '(<<acc.FINAL_majority_vote>>%, 95% bootstrap interval <<st.bootstrap_ci.FINAL_majority.ci95>>%) and significantly better than each '
          'member. Confidence-weighted voting is equally accurate, but it adds a useful confidence score. H2 is supported.', new=True),
        P('**RQ3.** Partly. The explanations are more faithful than random maps and depend on the model weights, so they are not mere edge '
          'detectors. However, they differ markedly between architectures and sometimes highlight regions outside the brain. Lower cross-model '
          'agreement is significantly associated with errors, but agreement is a weaker warning signal than ensemble confidence. The '
          'explanations help explain glioma errors (Chapter 5) but cannot confirm that the models use clinically correct evidence. H3 is '
          'therefore partly supported (see Section 5.10).', new=True),
        H(2, '7.3 Recommendations', new=True),
        NL(['**Report only held-out results.** Keep the test folder untouched, log per-image predictions, and report confidence intervals and paired tests. This single change corrected the project’s main claim.',
            '**Use confidence, not EAA-IoU alone, to trigger review.** Ensemble confidence flagged errors better (AUROC <<st.confidence_auroc>>) than attention agreement (AUROC <<eaa.auroc_low_eaa_predicts_error>>). If EAA-IoU is kept in the web demonstrator, recalibrate its threshold from the observed distribution (for example, 0.05–0.10) rather than the planned 0.40–0.65, and show it as a secondary cue.',
            '**Strengthen the pretrained models fairly.** Fine-tune the upper backbone blocks with a small learning rate, use the native input size, add augmentation and early stopping, then re-test RQ1.',
            '**Fix the data issues.** Remove augmented copies from the test set, check for near-duplicate slices across folders, and, if possible, obtain patient-level splits.',
            '**Validate externally and with clinicians.** Test on a dataset from another source, and run the Appendix D questionnaire with radiologists once ethical approval is in place, including whether the heatmaps change their decisions.',
            '**Add localisation ground truth.** Even a small set of tumour masks would allow pointing-game or IoU evaluation of the explanations against anatomy.'], new=True),
        H(2, '7.4 Limitations'),
        P('The main limitations are: one public dataset of 2D slices without patient identifiers; augmented meningioma test images; a '
          'single-source no-tumour class; one training run per model without cross-validation or hyperparameter tuning; frozen pretrained '
          'backbones; explanation metrics computed on a 160-image sample; no pixel-level ground truth for localisation; and no clinician '
          'evaluation. Ensembles also cost more to run, because six predictors must be evaluated per image.', new=True),
        H(2, '7.5 Future work'),
        P('Beyond the recommendations above, future work should (i) compare post-hoc saliency with inherently interpretable or attention-based '
          'models (Rudin, 2019); (ii) explore stacking and calibrated probability averaging; (iii) extend from 2D slices to 3D volumes and '
          'multi-sequence MRI; (iv) add multi-task outputs such as tumour location; and (v) study, with clinicians, whether showing '
          'explanations and confidence improves decisions or merely increases trust (Ghassemi, Oakden-Rayner and Beam, 2021).', new=True),
    ]

def ch8():
    return [
        H(1, 'Chapter 8 Reflection'),
        P('This chapter reflects on how effective the research methods were, which alternatives would have served better, and what I learned.'),
        H(2, '8.1 Effectiveness of the research methods', new=True),
        P('The quantitative experimental method answered RQ1 and RQ2 decisively, because every model was evaluated on the same untouched test '
          'images and compared with paired tests. Logging results as data proved to be the most effective decision of the project: it exposed '
          'the training-set evaluation in the HND report, and it allowed every number in this report to be regenerated and cross-checked by '
          'script. The secondary research (literature review) was effective in shaping the explainability evaluation, especially the sanity '
          'checks and deletion/insertion tests, which turned the heatmaps from illustrations into evidence. The method was least effective for '
          'the clinical side of the research: without clinician input, questions about usability and trust remain open.', new=True),
        H(2, '8.2 Alternative methodologies and lessons learnt', new=True),
        P('With hindsight, three alternatives would have strengthened the work. A patient-level, cross-validated design would have given more '
          'robust estimates. A smaller qualitative study with two or three radiologists, even informal, would have tested whether the '
          'explanations are useful. And a pre-registered analysis plan would have prevented the original evaluation error. The main lessons '
          'are practical: check where every reported number comes from; distrust results that are much better than the literature; keep '
          'preprocessing identical across training and evaluation code; and treat an explanation as a hypothesis to be tested rather than as an answer.', new=True),
        H(2, '8.3 Personal reflection'),
        H(3, '8.3.1 Benefits for the researcher'),
        P('The project developed my skills in deep learning, classical machine learning, statistical testing and explainable AI. More '
          'importantly, it changed how I think about evidence. Finding that my own headline result was measured on training data was '
          'uncomfortable, but correcting it publicly taught me more about research integrity than any successful experiment. Writing for '
          'clinicians also taught me to separate what a model does from what it appears to do.', new=True),
        H(3, '8.3.2 Benefits for industry and healthcare organisations'),
        P('For healthcare organisations, the project offers a transparent, reproducible template for evaluating imaging AI before procurement: '
          'held-out testing, intervals, paired tests and stress-tests of explanations. It also shows concretely why vendor accuracy claims '
          'should be checked against held-out, external data, and why explanation maps need their own validation before they are shown to clinicians.', new=True),
    ]
