from engine import *


def ch1():
    return [
        H(1, "Chapter 1 Introduction"),
        P(
            "This chapter explains why automated brain tumour classification matters, states the problem addressed by the project, and sets out "
            "the aim, objectives, research questions and hypotheses that structure the rest of the report."
        ),
        H(2, "1.1 Background and motivation"),
        P(
            "Magnetic resonance imaging (MRI) is the main imaging test for suspected brain tumours, and the tumour type strongly influences "
            "treatment and prognosis (Louis et al., 2021). Gliomas, meningiomas and pituitary tumours differ in location, appearance and "
            "behaviour, but on a single two-dimensional slice their appearance can overlap, and visual reporting is time-consuming and varies "
            "between readers (Tripathi and Bag, 2019). Computer-aided classification is therefore attractive as a fast and consistent second "
            "reader (Topol, 2019)."
        ),
        P(
            "Deep learning has changed medical image analysis because convolutional networks learn image features directly from data rather "
            "than relying on hand-crafted descriptors (LeCun, Bengio and Hinton, 2015; Litjens et al., 2017). Pretrained networks such as "
            "InceptionV3 and Xception allow knowledge learned on natural images to be transferred to medical tasks with limited data (Shin et al., "
            "2016), and ensemble learning can combine several imperfect models into a more reliable one (Dietterich, 2000; Ganaie et al., 2022)."
        ),
        P(
            "High accuracy on a benchmark is not enough for clinical use. Clinicians need to know what evidence a model used, whether it looked "
            "at the tumour or at irrelevant parts of the image, and when it is uncertain (Kelly et al., 2019; Tjoa and Guan, 2021). Explanation "
            "methods are usually added at the end of a project to illustrate a model. In this project they are also used as a diagnostic tool "
            "that feeds back into the design of the classifier."
        ),
        H(2, "1.2 Problem statement"),
        P(
            "Most published brain tumour classifiers evaluate a single model family and report accuracy without checking whether the model "
            "attends to clinically meaningful regions. Public MRI collections are also often assembled from several sources, which can leave "
            "cues such as image size or framing that differ by class, and can place near-identical images in both the training and test "
            "folders. Both problems can make a model look better than it is. The problem addressed here is therefore threefold: to compare CNN, "
            "transfer-learning and classical approaches and their ensemble fairly on held-out data; to explain and test what the resulting "
            "models rely on; and to use what the explanations reveal to build a more reliable classifier."
        ),
        H(2, "1.3 Aim"),
        P(
            "To design, evaluate and explain an integrated framework that compares a custom CNN, ImageNet-pretrained networks and classical "
            "machine-learning classifiers for four-class brain tumour MRI classification, combines them through ensemble voting, and uses "
            "explainable AI to identify and reduce reliance on non-clinical image cues."
        ),
        H(2, "1.4 Research objectives"),
        NL(
            [
                "RO1. Evaluate how well a custom CNN extracts discriminative features for brain tumour classification from MRI.",
                "RO2. Compare pretrained InceptionV3 and Xception networks, used first as frozen feature extractors and then fine-tuned, with the custom CNN.",
                "RO3. Assess Random Forest, Decision Tree and Support Vector Machine classifiers on raw pixels and on deep features.",
                "RO4. Determine whether combining six predictors by majority or confidence-weighted voting improves on the best single model.",
                "RO5. Apply complementary explanation methods (Grad-CAM, Grad-CAM++, HiResCAM, LIME, SHAP and occlusion) to correct and misclassified cases in every class.",
                "RO6. Evaluate the explanations quantitatively and use the findings to improve the pre-processing and training of the classifiers.",
            ]
        ),
        H(2, "1.5 Research questions"),
        B(
            [
                "RQ1. Do pretrained networks outperform a custom CNN for four-class brain tumour MRI classification, and does fine-tuning change the answer?",
                "RQ2. Does a six-model ensemble outperform its individual members on held-out data, and does confidence weighting improve on majority voting?",
                "RQ3. Are the explanations faithful to the models, sensitive to the model parameters and consistent across architectures, and what do they reveal about the errors?",
                "RQ4. Do changes guided by the explanations (head cropping, augmentation and fine-tuning) reduce attention outside the brain and improve accuracy, including on test images that have no duplicate in the training data?",
            ]
        ),
        H(2, "1.6 Hypotheses"),
        B(
            [
                "H1. Pretrained (InceptionV3, Xception) models achieve higher test accuracy than the custom CNN.",
                "H2. The six-model ensemble achieves higher test accuracy than any single member.",
                "H3. Saliency explanations are faithful (lower deletion AUC and higher insertion AUC than a random baseline) and change when the model "
                "weights are randomised; lower agreement between the explanations of different models is associated with misclassification.",
                "H4. Cropping images to the head, augmenting the training data and fine-tuning the pretrained backbones reduce the share of attention "
                "outside the brain and increase accuracy on both the full test set and its duplicate-free subset.",
            ]
        ),
        H(2, "1.7 Scope and boundaries"),
        P(
            "The study uses one public two-dimensional MRI dataset with four classes. It classifies whole images; it does not segment or grade "
            "tumours, and it does not use patient metadata. Explanations are assessed without pixel-level tumour masks because the dataset "
            "provides none. The software is a research prototype intended to support, never replace, a radiologist. No patient data were "
            "collected and no human participants took part in this phase."
        ),
        H(2, "1.8 Significance of the research"),
        B(
            [
                "It compares three modelling paradigms trained and tested under one protocol, with statistical tests rather than single accuracy figures.",
                "It shows how explanation methods can detect a dataset shortcut and guide a concrete fix, rather than only decorating predictions with heatmaps.",
                "It reports accuracy on a duplicate-free subset of the official test set, which gives a more realistic estimate than the usual benchmark figure.",
                "It provides a plain-language summary for clinicians and a prototype web demonstrator that shows predictions, heatmaps and an agreement-based alert.",
            ]
        ),
        H(2, "1.9 Structure of the report"),
        P(
            "Chapter 2 reviews the literature on brain tumour classification, ensembles and explainable AI. Chapter 3 describes the data, the two "
            "experimental configurations, training, evaluation, the explanation methods and ethics. Chapter 4 presents the results of the "
            "baseline configuration. Chapter 5 analyses the baseline models with explainable AI and investigates the cues they rely on. Chapter 6 "
            "presents the improved configuration and checks whether the problems were reduced. Chapter 7 discusses the findings, Chapter 8 states "
            "conclusions and recommendations, and Chapter 9 reflects on the research process."
        ),
    ]


def ch2():
    gap_rows = [
        [
            "Kang, Ullah and Gwak (2021)",
            "Deep features from several pretrained CNNs with classical classifiers",
            "Feature concatenation",
            "No",
            "No",
        ],
        [
            "Nhlapho et al. (2024)",
            "Pretrained CNNs for detection and classification",
            "No",
            "Yes (Grad-CAM, LIME, SHAP)",
            "No",
        ],
        [
            "Aasem and Javed Iqbal (2024)",
            "Chest X-ray, several CNNs",
            "Ensemble of CAMs",
            "Yes (fused CAM)",
            "No (maps compared with box annotations)",
        ],
        [
            "Kakon et al. (2025)",
            "Pretrained CNN ensemble with meta-learner",
            "Stacking",
            "Yes (Grad-CAM++)",
            "No",
        ],
        [
            "This study",
            "Custom CNN, InceptionV3 and Xception, plus RF/DT/SVM on deep features",
            "Majority and confidence-weighted vote",
            "Yes (six methods; faithfulness and sanity tested; used to redesign pre-processing)",
            "Yes (EAA-IoU and rank correlation)",
        ],
    ]
    return [
        H(1, "Chapter 2 Literature Review"),
        P(
            "This chapter reviews work on brain tumour MRI classification with classical and deep models, ensemble learning, and explainable AI "
            "in medical imaging. It ends with the research gap and the conceptual framework of the study. The review is organised by technique "
            "rather than by date so that the strengths and weaknesses of each family are visible."
        ),
        H(2, "2.1 Clinical context"),
        P(
            "The World Health Organization classification of central nervous system tumours now combines histology with molecular markers "
            "(Louis et al., 2021), so imaging alone cannot give a final diagnosis. Image-based classification is still valuable for triage, for "
            "prioritising reporting and for flagging inconsistent reports. Weak external validation, dataset shift and a lack of transparency "
            "are the main barriers to clinical impact (Kelly et al., 2019), which motivates both the held-out evaluation and the explainability "
            "work in this project."
        ),
        H(2, "2.2 Classical machine-learning approaches"),
        P(
            "Early systems relied on hand-crafted features. Minz and Mahobiya (2017) extracted grey-level co-occurrence texture features and "
            "classified them with AdaBoost, reporting 89.90% accuracy. Ismael and Abdel-Qader (2018) combined statistical features with a "
            "back-propagation network, and Tripathi and Bag (2019) graded tumours with noise-robust texture and intensity features. Cheng et al. "
            "(2015) showed that enlarging the tumour region of interest and dividing it into rings raised accuracy on a three-class "
            "contrast-enhanced T1 dataset to 91.28%. These methods depend on careful feature design and often on a manually outlined tumour "
            "region. Support Vector Machines (Cortes and Vapnik, 1995), decision trees (Quinlan, 1986) and Random Forests (Breiman, 2001) remain "
            "strong baselines when good features are available, which is why they are used here on raw pixels and on deep features."
        ),
        H(2, "2.3 Deep learning and transfer learning"),
        P(
            "Convolutional networks learn hierarchies of features from pixels (LeCun, Bengio and Hinton, 2015) and now dominate medical image "
            "analysis (Litjens et al., 2017). Abiwinanda et al. (2018) showed that even a shallow CNN can separate glioma, meningioma and "
            "pituitary tumours on the public figshare dataset of Cheng et al. (2015), and Sultan, Salem and Al-Atabany (2019) reported a deeper "
            "network for multi-class classification and grading. Related work applies recurrent models to slices of 3D scans: Zhou et al. (2019) "
            "combined DenseNet features with a recurrent network and reported 92.13% accuracy, and Patel et al. (2019) used a CNN with LSTM "
            "layers to detect haemorrhage in CT. Yousaf et al. (2023) extended deep models to detect several brain conditions."
        ),
        P(
            "Transfer learning reuses networks pretrained on ImageNet (Deng et al., 2009). Shin et al. (2016) showed that pretrained features "
            "transfer well to medical detection tasks. For brain tumours, Deepak and Ameer (2019) used GoogLeNet features with SVM and KNN "
            "classifiers and reported a mean accuracy of 98% under patient-level five-fold cross-validation, and Rehman et al. (2020) reported "
            "98.69% with a fine-tuned VGG16. Swati et al. (2019) proposed block-wise fine-tuning of VGG19, Badža and Barjaktarović (2020) showed "
            "that a purpose-built lighter CNN can compete with pretrained networks, and Talo et al. (2019) reached 100% accuracy for normal versus "
            "abnormal classification on 613 images with ResNet34. InceptionV3 factorises large convolutions to reduce cost (Szegedy et al., 2016), "
            "and Xception replaces Inception modules with depthwise separable convolutions (Chollet, 2017); both are used here. Saeedi et al. "
            "(2023) compared CNN models with classical classifiers on a four-class brain MRI dataset of the kind used in this study. A recurring "
            "caution is that figures from different datasets, splits and validation schemes are not directly comparable."
        ),
        H(2, "2.4 Ensemble learning"),
        P(
            "Ensembles reduce variance and the effect of individual model biases when their members make different errors (Dietterich, 2000). "
            "Ganaie et al. (2022) review deep ensembles and note that diversity among members matters more than their number. In brain MRI, "
            "Kang, Ullah and Gwak (2021) extracted features from several pretrained networks and fed them to classical classifiers, finding that "
            "SVMs with an RBF kernel often performed best and that combining deep features helped. Kakon et al. (2025) stacked several pretrained "
            "networks with a meta-learner and added Grad-CAM++ explanations. This project follows the hybrid idea of Kang, Ullah and Gwak (2021) "
            "and combines deep and classical predictors by voting."
        ),
        H(2, "2.5 Explainable AI in medical imaging"),
        P(
            "Explainable AI aims to make model behaviour understandable to people (Barredo Arrieta et al., 2020). In medical imaging, post-hoc "
            "saliency methods dominate (van der Velden et al., 2022; Salahuddin et al., 2022). Gradient-weighted class activation mapping "
            "(Grad-CAM) weights the feature maps of a convolutional layer by the average gradient of the class score (Selvaraju et al., 2017). "
            "Grad-CAM++ uses higher-order gradient terms to localise multiple or small objects better (Chattopadhay et al., 2018), and HiResCAM "
            "multiplies gradients and activations element by element, which avoids highlighting regions the model did not use when the network "
            "does not end in global average pooling (Draelos and Carin, 2020). Perturbation methods ask how the prediction changes when parts of "
            "the image are hidden: occlusion slides a blank patch over the image (Zeiler and Fergus, 2014), and LIME fits a simple local "
            "surrogate model on super-pixels (Ribeiro, Singh and Guestrin, 2016). SHAP attributes a prediction to its inputs using Shapley "
            "values (Lundberg and Lee, 2017); for deep networks its gradient-based variant approximates expected gradients, and for tree "
            "ensembles it can be computed exactly. Relevance propagation methods such as deep Taylor decomposition form another family "
            "(Montavon et al., 2017)."
        ),
        P(
            "For brain tumours, Esmaeili et al. (2021) compared CNN saliency maps with tumour locations, Zeineldin et al. (2022) evaluated "
            "several explanation methods for MRI segmentation and classification networks, and Nhlapho et al. (2024) applied Grad-CAM, LIME and "
            "SHAP to pretrained classifiers. Most of these studies explain a single model and assess the maps visually."
        ),
        H(2, "2.6 Evaluating explanations"),
        P(
            "A convincing heatmap is not necessarily a faithful one. Adebayo et al. (2018) showed that some saliency methods barely change when "
            "the model weights are randomised, so they reflect the image rather than the model, and proposed parameter-randomisation sanity "
            "checks. Petsiuk, Das and Saenko (2018) introduced deletion and insertion curves, which measure how quickly the class score falls "
            "when the most salient pixels are removed, or rises when they are revealed. In medical imaging, Arun et al. (2021) and Saporta et al. "
            "(2022) found that common saliency methods often fail to localise abnormalities reliably compared with radiologist annotations. "
            "Ghassemi, Oakden-Rayner and Beam (2021) argue that current explanations can give false reassurance, and Rudin (2019) argues that "
            "inherently interpretable models should be preferred for high-stakes decisions where possible. These critiques shape Chapter 5, "
            "which treats explanations as hypotheses to be tested rather than as proof."
        ),
        H(2, "2.7 Shortcut learning and data leakage"),
        P(
            "Deep networks can succeed by exploiting shortcuts, that is, features that correlate with the label in the training data but are not "
            "causally related to it (Geirhos et al., 2020). Zech et al. (2018) showed that a pneumonia detector partly recognised the hospital "
            "that produced a radiograph. The dataset used here combines images from several collections, and all no-tumour images come from a "
            "single collection (Nickparvar, n.d.), so cues such as image size, borders or contrast are a realistic risk. A related risk is "
            "leakage, where the same image (or a re-saved copy) appears in both training and test data and inflates the measured accuracy."
        ),
        H(2, "2.8 Research gap"),
        P(
            "Two gaps follow from the review. First, few studies compare custom CNNs, pretrained networks and classical classifiers on the same "
            "split with statistical tests. Second, studies that add explanations usually show heatmaps for one model and seldom test their "
            "faithfulness, compare them across architectures, or use them to change the model. {tab:gap} positions this study against the "
            "closest work."
        ),
        TAB(
            "gap",
            "Positioning of this study against the closest related work",
            ["Study", "Models", "Ensemble", "Explanations", "Cross-model agreement"],
            gap_rows,
            widths=[3.2, 4.2, 2.6, 3.0, 2.8],
            font=9,
        ),
        H(2, "2.9 Conceptual framework"),
        P(
            "{fig:framework} summarises the framework. Pre-processed MRI images feed three deep feature learners. Their outputs are classified by "
            "softmax heads or by classical classifiers, combined by voting, and evaluated with accuracy-based metrics and statistical tests. The "
            "explainability layer examines the deep models and the classical stage, and its findings feed back into the pre-processing and "
            "training design."
        ),
        FIG(
            "framework",
            "paper_assets/figures/fig_conceptual_framework.png",
            "Conceptual framework of the study",
            width=6.2,
        ),
    ]
