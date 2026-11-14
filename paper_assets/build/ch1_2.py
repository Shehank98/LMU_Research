from engine import *

def ch1():
    return [
        H(1, 'Chapter 1 Introduction'),
        P('This chapter sets out why automated brain tumour classification matters, the problem the project addresses, and the objectives, '
          'research questions and hypotheses that structure the rest of the report. It also states what changed between the first (HND) '
          'stage of the study and this extended version.'),
        H(2, '1.1 Background and motivation'),
        P('Magnetic resonance imaging (MRI) is the main imaging test for suspected brain tumours, and the tumour type strongly influences '
          'treatment and prognosis (Louis et al., 2021). Gliomas, meningiomas and pituitary tumours differ in location, appearance and '
          'behaviour, but on a single two-dimensional slice their appearance can overlap, and visual reporting is time-consuming and varies '
          'between readers (Tripathi and Bag, 2019). Computer-aided classification is therefore attractive as a second reader that is fast '
          'and consistent (Topol, 2019).'),
        P('Deep learning has changed medical image analysis, because convolutional networks learn image features directly from data rather '
          'than relying on hand-crafted descriptors (LeCun, Bengio and Hinton, 2015; Litjens et al., 2017). Pretrained networks such as '
          'InceptionV3 and Xception allow transfer of knowledge from natural images to medical tasks with limited data (Shin et al., 2016), '
          'and ensemble learning can combine several imperfect models into a more reliable one (Dietterich, 2000; Ganaie et al., 2022).'),
        P('High accuracy alone is not enough for clinical adoption. Clinicians need to know what evidence a model used, whether it is '
          'looking at the tumour or at irrelevant artefacts, and when it is uncertain (Kelly et al., 2019; Tjoa and Guan, 2021). The first '
          'stage of this research focused only on accuracy and listed explainability as future work. This report adds that missing layer.', new=True),
        H(2, '1.2 Problem statement'),
        P('Most published brain tumour classifiers evaluate a single model family, and many report accuracy without testing whether the '
          'model attends to clinically meaningful regions. In addition, the first stage of this project reported a final ensemble accuracy '
          'of 98.20%, but an audit for this report found that this value was computed on the 6,985 training images rather than on held-out '
          'data (see AUDIT.md and Section 4.10). The problem addressed here is therefore twofold: (i) to obtain an honest, reproducible '
          'comparison of CNN, transfer-learning and classical approaches and their ensemble on unseen data, and (ii) to explain and '
          'critically evaluate what the resulting models rely on.', new=True),
        H(2, '1.3 Aim'),
        P('To design, evaluate and explain an integrated framework that compares custom CNNs, pretrained transfer-learning models and '
          'classical machine-learning classifiers for four-class brain tumour MRI classification, combines them through ensemble voting, '
          'and assesses the faithfulness and clinical plausibility of their explanations.'),
        H(2, '1.4 Research objectives'),
        P('Objectives RO1–RO4 are carried over from the HND study, reworded for precision. RO5 and RO6 are new.'),
        NL(['RO1 – Evaluate how well a custom CNN extracts discriminative features for brain tumour classification from MRI.',
            'RO2 – Evaluate pretrained InceptionV3 and Xception networks, used as frozen feature extractors with trained heads, against the custom CNN.',
            'RO3 – Assess Random Forest, Decision Tree and Support Vector Machine classifiers on raw pixels and on deep features.',
            'RO4 – Determine whether combining six predictors by majority or confidence-weighted voting improves accuracy and reliability over the best single model.']),
        NL(['RO5 – Apply complementary explanation methods (Grad-CAM, Grad-CAM++, HiResCAM, LIME, SHAP and occlusion) to the trained models for correct and misclassified cases in every class.',
            'RO6 – Quantitatively evaluate the explanations (deletion/insertion faithfulness, weight-randomisation sanity check, cross-model agreement and attention outside the brain) and relate them to the models’ errors, especially for glioma.'], new=True),
        H(2, '1.5 Research questions'),
        B(['RQ1 – Do pretrained networks used as feature extractors outperform a custom CNN for four-class brain tumour MRI classification?',
           'RQ2 – Does a six-model ensemble outperform its individual members on held-out data, and does confidence weighting improve on majority voting?']),
        B(['RQ3 – Are the explanations produced for these models faithful to the models, sensitive to the model parameters, and consistent across architectures, and do they help explain why glioma is the hardest class?'], new=True),
        H(2, '1.6 Hypotheses'),
        B(['H1 – Pretrained (InceptionV3, Xception) models achieve higher test accuracy than the custom CNN.',
           'H2 – The six-model ensemble achieves higher test accuracy than any single member.']),
        B(['H3 – Saliency explanations of the trained models are faithful (deletion AUC lower and insertion AUC higher than a random baseline) '
           'and model-dependent (they change substantially when the model weights are randomised); lower agreement between the explanations of '
           'different models is associated with misclassification.'], new=True),
        H(2, '1.7 Scope and boundaries', new=True),
        P('The study uses one public two-dimensional MRI dataset with four classes. It classifies whole images; it does not segment tumours, '
          'grade them or use patient metadata. Explanations are assessed without pixel-level tumour masks, because the dataset provides none. '
          'The software is a research prototype intended to support, never replace, a radiologist. No new patient data and no human '
          'participants were involved in this stage.', new=True),
        H(2, '1.8 Significance of the research'),
        B(['It provides a like-for-like comparison of three modelling paradigms trained and tested under one protocol, with statistical tests rather than single accuracy figures.',
           'It corrects an inflated accuracy claim and shows how evaluation choices change conclusions, which is a useful lesson for student and practitioner projects.'], new=False),
        B(['It adds a multi-method, quantitatively evaluated explainability analysis, including a cross-architecture agreement score (EAA-IoU) that needs no tumour annotations.',
           'It offers a plain-language summary of the evidence for clinicians and a prototype web demonstrator that shows predictions, heatmaps and agreement.'], new=True),
        H(2, '1.9 Structure of the report', new=True),
        P('Chapter 2 reviews the literature on brain tumour classification, ensembles and explainable AI. Chapter 3 describes the data, models, '
          'training protocol, evaluation statistics, explanation methods and ethics. Chapter 4 presents the classification results. Chapter 5 '
          'presents the new explainability analysis. Chapter 6 discusses the findings against the literature and critically evaluates the '
          'methodology, and closes with a summary for clinical stakeholders. Chapter 7 states conclusions and recommendations, and Chapter 8 '
          'reflects on the research process. The appendices hold code listings, extra tables, the proposed clinician questionnaire, a mapping '
          'to the assessment criteria, a change log and a figures index.', new=True),
    ]

def ch2():
    gap_rows = [
        ['Kang, Ullah and Gwak (2021)', 'Deep features from several pretrained CNNs + classical classifiers', 'Feature concatenation', 'No', 'No'],
        ['Nhlapho et al. (2024)', 'Pretrained CNNs for detection and classification', 'No', 'Yes (Grad-CAM, LIME, SHAP)', 'No'],
        ['Aasem and Javed Iqbal (2024)', 'Chest X-ray, several CNNs', 'Ensemble of CAMs', 'Yes (fused CAM)', 'No (maps compared with box annotations)'],
        ['Kakon et al. (2025)', 'Pretrained CNN ensemble with meta-learner', 'Stacking', 'Yes (Grad-CAM++)', 'No'],
        ['This study', 'Custom CNN + 2 frozen pretrained networks + RF/DT/SVM on deep features', 'Majority and confidence-weighted vote',
         'Yes (6 methods, faithfulness and sanity tested)', 'Yes (EAA-IoU and rank correlation, linked to errors)']]
    return [
        H(1, 'Chapter 2 Literature Review'),
        P('This chapter reviews work on brain tumour MRI classification with classical and deep models, ensemble learning, and explainable AI '
          'in medical imaging. It ends with the research gap and the conceptual framework of the study. The review is organised by technique '
          'rather than by date, so that the strengths and weaknesses of each family are visible.'),
        H(2, '2.1 Clinical context'),
        P('The World Health Organization classification of central nervous system tumours now combines histology with molecular markers '
          '(Louis et al., 2021), so imaging alone cannot provide a final diagnosis. Imaging-based classification is still valuable for triage, '
          'for prioritising reporting work and for flagging inconsistent reports. Key barriers to clinical impact include weak external '
          'validation, dataset shift and a lack of transparency (Kelly et al., 2019), which motivates both the held-out evaluation and the '
          'explainability work in this project.', new=True),
        H(2, '2.2 Classical machine-learning approaches'),
        P('Early systems relied on hand-crafted features. Minz and Mahobiya (2017) extracted grey-level co-occurrence texture features and '
          'classified them with AdaBoost, reporting 89.90% accuracy. Ismael and Abdel-Qader (2018) combined statistical features with a '
          'back-propagation network, and Tripathi and Bag (2019) graded tumours with noise-robust texture and intensity features. Cheng et al. '
          '(2015) showed that enlarging the tumour region of interest and partitioning it into rings raised accuracy on a three-class '
          'contrast-enhanced T1 dataset to 91.28%. These methods depend on careful feature design and often on a manually outlined tumour '
          'region. Support Vector Machines (Cortes and Vapnik, 1995), decision trees (Quinlan, 1986) and Random Forests (Breiman, 2001) remain '
          'strong baselines when good features are available, which is why they are retained here, both on raw pixels and on deep features.'),
        H(2, '2.3 Deep learning and transfer learning'),
        P('Convolutional networks learn hierarchies of features from pixels (LeCun, Bengio and Hinton, 2015) and now dominate medical image '
          'analysis (Litjens et al., 2017). Abiwinanda et al. (2018) showed that even a shallow CNN can separate glioma, meningioma and '
          'pituitary tumours on the public figshare dataset of Cheng et al. (2015), and Sultan, Salem and Al-Atabany (2019) reported a deeper '
          'network for multi-class classification and grading. Related work applies recurrent models to slices of 3D scans: Zhou et al. (2019) '
          'combined DenseNet features with a recurrent network and reported 92.13% accuracy, and Patel et al. (2019) used a CNN–LSTM for '
          'haemorrhage detection in CT. Yousaf et al. (2023) extended deep models to detect several brain conditions.'),
        P('Transfer learning reuses networks pretrained on ImageNet (Deng et al., 2009). Shin et al. (2016) showed that pretrained features '
          'transfer well to medical detection tasks. For brain tumours, Deepak and Ameer (2019) used GoogLeNet features with SVM and KNN '
          'classifiers and reported a mean accuracy of 98% under patient-level five-fold cross-validation, and Rehman et al. (2020) reported '
          '98.69% with a fine-tuned VGG16. Swati et al. (2019) proposed block-wise fine-tuning of VGG19, Badža and Barjaktarović (2020) '
          'showed that a purpose-built lighter CNN can be competitive with pretrained networks, and Talo et al. (2019) reached 100% accuracy '
          'for normal-versus-abnormal classification on 613 images with ResNet34. InceptionV3 factorises large convolutions to reduce cost '
          '(Szegedy et al., 2016), and Xception replaces Inception modules with depthwise separable convolutions (Chollet, 2017); both are used '
          'here. Saeedi et al. (2023) compared CNN models with classical classifiers on a four-class brain MRI dataset of the kind used in this '
          'study. A recurring caution is that figures from different datasets, splits and validation schemes are not directly comparable.'),
        H(2, '2.4 Ensemble learning'),
        P('Ensembles reduce variance and the effect of individual model biases when their members make different errors (Dietterich, 2000). '
          'Ganaie et al. (2022) review deep ensembles and note that diversity among members matters more than the number of members. In '
          'brain MRI, Kang, Ullah and Gwak (2021) extracted features from several pretrained networks and fed them to classical classifiers, '
          'finding that SVMs with an RBF kernel often performed best and that combining deep features helped. Kakon et al. (2025) stacked '
          'several pretrained networks with a meta-learner and added Grad-CAM++ explanations. The HND stage of this project followed the '
          'hybrid idea of Kang, Ullah and Gwak (2021) and combined deep and classical predictors by majority voting.'),
        H(2, '2.5 Explainable AI in medical imaging', new=True),
        P('Explainable AI aims to make model behaviour understandable to people (Barredo Arrieta et al., 2020). In medical imaging, '
          'post-hoc saliency methods dominate (van der Velden et al., 2022; Salahuddin et al., 2022). Gradient-based class activation mapping '
          '(Grad-CAM) weights the feature maps of a convolutional layer by the average gradient of the class score (Selvaraju et al., 2017). '
          'Grad-CAM++ uses higher-order gradient terms to better localise multiple or small objects (Chattopadhay et al., 2018), and HiResCAM '
          'multiplies gradients and activations element-wise, which avoids highlighting regions the model did not use when the network does '
          'not end in global average pooling (Draelos and Carin, 2020). Perturbation methods ask how the prediction changes when parts of the '
          'image are hidden: occlusion slides a blank patch over the image (Zeiler and Fergus, 2014), and LIME fits a simple local surrogate '
          'model on super-pixels (Ribeiro, Singh and Guestrin, 2016). SHAP attributes the prediction to input features using Shapley values '
          '(Lundberg and Lee, 2017); for deep networks its gradient-based variant approximates expected gradients, and for tree ensembles it '
          'can be computed exactly. Relevance propagation methods such as deep Taylor decomposition offer another family (Montavon et al., 2017).', new=True),
        P('For brain tumours, Esmaeili et al. (2021) compared CNN saliency maps with tumour locations, Zeineldin et al. (2022) evaluated '
          'several explanation methods for MRI segmentation and classification networks, and Nhlapho et al. (2024) applied Grad-CAM, LIME and '
          'SHAP to pretrained classifiers. Most of these studies explain a single model and assess the maps visually.', new=True),
        H(2, '2.6 Evaluating explanations', new=True),
        P('A visually convincing heatmap is not necessarily faithful. Adebayo et al. (2018) showed that some saliency methods barely change '
          'when the model weights are randomised, so they reflect the image rather than the model; they proposed parameter-randomisation '
          'sanity checks. Petsiuk, Das and Saenko (2018) introduced deletion and insertion curves, which measure how quickly the class score '
          'falls when the most salient pixels are removed, or rises when they are revealed. In medical imaging, Arun et al. (2021) and '
          'Saporta et al. (2022) found that common saliency methods often fail to localise abnormalities reliably compared with radiologist '
          'annotations. Ghassemi, Oakden-Rayner and Beam (2021) argue that current explanations can give false reassurance, and Rudin (2019) '
          'argues that, for high-stakes decisions, inherently interpretable models should be preferred where possible. These critiques shape '
          'Chapter 5, which treats explanations as hypotheses to be tested rather than as proof.', new=True),
        H(2, '2.7 Shortcut learning', new=True),
        P('Deep networks can succeed by exploiting shortcuts: features that correlate with the label in the training data but are not causally '
          'related to it (Geirhos et al., 2020). Zech et al. (2018) showed that a pneumonia detector partly recognised the hospital that '
          'produced a radiograph. The dataset used here combines images from several sources, and all no-tumour images come from a single '
          'source (Nickparvar, n.d.), so source-specific cues such as image borders, contrast or skull appearance are a realistic risk that '
          'explanations should help to detect.', new=True),
        H(2, '2.8 Research gap'),
        P('Two gaps follow from the review. First, few studies compare custom CNNs, frozen pretrained networks and classical classifiers on '
          'the same split with statistical tests. Second, studies that add explanations usually show heatmaps for one model and seldom test '
          'faithfulness, run sanity checks or compare explanations across architectures. {tab:gap} positions this study against the closest work.', new=True),
        TAB('gap', 'Positioning of this study against the closest related work', ['Study', 'Models', 'Ensemble', 'Explanations', 'Cross-model agreement'],
            gap_rows, new=True, widths=[3.2, 4.2, 2.6, 3.0, 2.8], font=9),
        H(2, '2.9 Conceptual framework'),
        P('{fig:framework} summarises the framework. Pre-processed MRI images feed three deep feature learners. Their outputs are classified by '
          'softmax heads or by classical classifiers, combined by voting, and evaluated with accuracy-based metrics and statistical tests. '
          'The new explainability layer sits across the deep models and the classical stage, and its findings feed a clinician-facing summary.', new=True),
        FIG('framework', 'paper_assets/figures/fig_conceptual_framework.png', 'Conceptual framework of the study, with the new explainability layer', new=True, width=6.2),
    ]
