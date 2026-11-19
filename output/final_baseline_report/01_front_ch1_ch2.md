---
title: "A Comparative Analysis of Ensemble Learning, Fine-Tuned Models and CNNs for Multi-Class Disease Prediction in Medical Imaging: An Integrated Framework with Systematic Optimization"
subtitle: "CS6P05ES Computing Project: Final Project Report"
author:
  - "Pattiyage Shehan Kavishka (Student ID: E187041)"
  - "London Metropolitan University, delivered at ESOFT Metro Campus"
  - "First Supervisor: Ms. Hiruni Wimalaweera | Module Leader: Mr. Thepul Ginige"
date: "November 2026"
lang: en-GB
---

# Abstract {.unnumbered}

Brain tumour type determines treatment, yet the visual reading of magnetic resonance imaging (MRI) is slow and varies between readers. This project, a continuation of the Interim Report, builds and evaluates an integrated framework for four-class brain MRI classification (glioma, meningioma, pituitary tumour and no tumour) on a public dataset of 7,200 images. Under one standardised pipeline (256 × 256 inputs for standalone networks, 128 × 128 inputs for feature extractors, pixel values scaled to [0, 1], batch size 32 and random seed 42), it compares a custom four-block convolutional neural network (CNN), ImageNet-pretrained InceptionV3 and Xception networks with frozen backbones, and Random Forest, Decision Tree, Support Vector Machine and Logistic Regression classifiers trained on raw pixels and on deep features. Six predictors are then combined by majority and confidence-weighted voting. On 1,600 held-out test images, the custom CNN was the strongest single model (90.44% accuracy), ahead of Xception (87.62%) and InceptionV3 (86.62%). Deep features raised classical classifiers substantially, for example from 82.25% to 89.69% for the SVM. The six-predictor majority vote reached 92.50% accuracy (macro F1 0.924), and the confidence-weighted vote 92.25%. Glioma remained the hardest class (77.50% recall), mostly confused with meningioma. To make the system interpretable, Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to the three deep networks, and a new Ensemble Attention Agreement score (EAA-IoU) measured how far their heatmaps overlap without requiring tumour masks. Correctly classified images showed higher agreement (mean 0.1029) than misclassified images (0.0471), and low agreement identified errors with an AUROC of 0.7525, although ensemble confidence remained a stronger signal (0.9127). An alert threshold of EAA-IoU below 0.10 caught 82.26% of errors. These components were combined in NeuroScan, a web-based decision-support prototype. The system is a research tool that supports, and does not replace, a radiologist.

**Keywords:** brain tumour MRI; convolutional neural network; transfer learning; ensemble learning; explainable AI; Grad-CAM; EAA-IoU; clinical decision support; web application

# List of Abbreviations {.unnumbered}

| Abbreviation | Meaning |
|---|---|
| AUROC / ROC AUC | Area under the receiver operating characteristic curve |
| CAM | Class activation map |
| CNN | Convolutional neural network |
| DT | Decision Tree |
| EAA-IoU | Ensemble Attention Agreement, intersection over union |
| F1 | Harmonic mean of precision and recall |
| GAP | Global average pooling |
| Grad-CAM | Gradient-weighted class activation mapping |
| IoU | Intersection over union |
| LR | Logistic Regression |
| MRI | Magnetic resonance imaging |
| PR | Precision-recall |
| RBF | Radial basis function |
| RF | Random Forest |
| RO / RQ / H | Research objective / research question / hypothesis |
| SVM | Support Vector Machine |
| XAI | Explainable artificial intelligence |

# Chapter 1: Introduction

This chapter explains why automated brain tumour classification matters, defines the problem the project addresses, and sets out the aim, objectives, research questions and hypotheses carried forward from the Interim Report.

## 1.1 Background

Magnetic resonance imaging is the principal imaging investigation for suspected brain tumours, and the tumour type strongly influences treatment planning and prognosis (Louis et al., 2021). Gliomas, meningiomas and pituitary tumours differ in location, appearance and behaviour, but on a single two-dimensional slice their appearance can overlap. Visual reporting is time-consuming and can vary between readers (Tripathi and Bag, 2019). Computer-aided classification is therefore attractive as a fast, consistent second reader that can help prioritise reporting workloads (Topol, 2019).

Deep learning has transformed medical image analysis because convolutional networks learn discriminative features directly from pixels instead of relying on hand-crafted descriptors (LeCun, Bengio and Hinton, 2015; Litjens et al., 2017). Networks pretrained on ImageNet, such as InceptionV3 and Xception, allow knowledge learnt from natural images to be transferred to medical tasks with limited data (Shin et al., 2016). Ensemble learning can combine several imperfect models into a more reliable one, provided the members make different errors (Dietterich, 2000; Ganaie et al., 2022).

High benchmark accuracy is not sufficient for clinical adoption. Clinicians need to see what evidence a model used, whether it attended to the lesion, and when it is uncertain (Kelly et al., 2019; Tjoa and Guan, 2021). Explainable artificial intelligence (XAI) addresses this need. In this project, Grad-CAM heatmaps are produced for three architecturally different networks, and the agreement between them is turned into a per-image uncertainty signal that can trigger a request for radiologist review.

## 1.2 Problem Statement

Most published brain tumour classifiers evaluate one model family in isolation and report a single accuracy figure. Comparisons between custom CNNs, pretrained networks and classical machine learning are rarely made on the same data split with the same pre-processing, so it is difficult to know which approach is genuinely better. When explanations are added, they are usually heatmaps for one model, inspected visually, with no measure of whether different models agree or whether disagreement signals an unreliable prediction. The problem addressed by this project is therefore twofold: to compare these modelling approaches and their ensemble fairly on held-out data, and to make the resulting system interpretable and safer through an explanation-based escalation mechanism.

## 1.3 Aim

To design, implement and evaluate an integrated framework that compares a custom CNN, ImageNet-pretrained networks (InceptionV3 and Xception) and classical machine-learning classifiers for four-class brain tumour MRI classification, combines them through six-predictor ensemble voting, and explains the deep models with Grad-CAM and an inter-model attention agreement score delivered through a clinical decision-support web application.

## 1.4 Research Objectives

1. **RO1.** Investigate how well a custom CNN extracts discriminative features for brain tumour identification from MRI.
2. **RO2.** Evaluate ImageNet-pretrained InceptionV3 and Xception networks, adapted through transfer learning, as standalone classifiers and as feature extractors, and compare them with the custom CNN.
3. **RO3.** Assess Random Forest, Decision Tree, Support Vector Machine and Logistic Regression classifiers on raw pixels and on features extracted by the three deep networks.
4. **RO4.** Determine whether combining six predictors through majority or confidence-weighted voting improves accuracy and reliability over the individual models.
5. **RO5.** Apply Grad-CAM to the three deep networks and evaluate the Ensemble Attention Agreement score (EAA-IoU) as a per-image signal of misclassification.
6. **RO6.** Develop a web-based clinical decision-support prototype (NeuroScan) that presents predictions, confidence, per-model votes, heatmaps and an agreement-based escalation alert.

## 1.5 Research Questions and Hypotheses

**Main research question:** How well do transfer-learned (pretrained) models perform for feature extraction compared with a standard CNN, and what effect does their integration with ensemble approaches have on the accuracy and dependability of brain tumour identification from MRI images?

| ID | Research question | Hypothesis |
|---|---|---|
| RQ1 / H1 | Do pretrained networks (InceptionV3, Xception) outperform a custom CNN, both as standalone classifiers and as feature extractors for classical classifiers? | H1: The pretrained networks achieve higher accuracy than the custom CNN. |
| RQ2 | Do deep features improve classical classifiers compared with raw pixels? | (Exploratory; no directional hypothesis was stated in the Interim Report.) |
| RQ3 / H2 | Does a six-predictor ensemble outperform its individual members, and does confidence weighting improve on majority voting? | H2: The ensemble achieves higher accuracy than any single member. |
| RQ4 / H3 | Does inter-model Grad-CAM agreement (EAA-IoU) correlate with misclassification, and can it serve as a per-image clinical escalation trigger? | H3: Images with low EAA-IoU are significantly more likely to be misclassified. |

Table: Table 1. Research questions and hypotheses

## 1.6 Scope

The project uses one public two-dimensional MRI dataset with four classes and classifies whole images. It does not segment or grade tumours, and it does not use patient metadata. All models are trained under the single standardised baseline pipeline defined in the Interim Report. The pretrained networks keep their ImageNet backbones frozen; "fine-tuned" in the project title refers to adapting these pretrained models to the task by training new classification layers on top of them. Explainability is limited to Grad-CAM and the EAA-IoU agreement score. Explanations are evaluated without pixel-level tumour masks, because the dataset provides none. The web application is a research prototype.

## 1.7 Significance

The project contributes in four ways:

- It compares three modelling paradigms (custom CNN, transfer learning and classical machine learning) under one protocol, on the same 1,600 held-out test images.
- It integrates deep and classical predictors in a six-predictor ensemble and analyses when the ensemble is reliable, using the number of agreeing votes.
- It introduces EAA-IoU, an annotation-free measure of whether architecturally different networks attend to the same region, and evaluates it as an escalation signal.
- It delivers a working decision-support prototype that shows clinicians not only a prediction but also where the models looked and whether they agreed.

## 1.8 Structure of the Report

Chapter 2 reviews the literature. Chapter 3 describes the dataset, the standardised framework, the models, the evaluation metrics, the Grad-CAM and EAA-IoU formulation, the design of the NeuroScan web application and the ethical considerations. Chapter 4 presents the classification results. Chapter 5 presents the explainability analysis and the web application. Chapter 6 discusses the findings, evaluates the hypotheses, compares the results with the literature, states the limitations and future work, and closes with a personal reflection.

# Chapter 2: Literature Review

This chapter reviews the clinical context, classical and deep approaches to brain tumour classification, ensemble learning and explainable AI in medical imaging. It ends with the research gap and the conceptual framework of the study.

## 2.1 Clinical Context

The World Health Organization classification of central nervous system tumours combines histology with molecular markers (Louis et al., 2021), so imaging alone cannot give a final diagnosis. Image-based classification is nevertheless valuable for triage, for prioritising reporting and for flagging reports that may need a second look. Kelly et al. (2019) identify weak external validation, dataset shift and lack of transparency as the main barriers to the clinical impact of AI. These barriers motivate the held-out evaluation, the explainability work and the explicit "decision-support only" positioning of this project.

## 2.2 Classical Machine Learning

Early systems relied on hand-crafted features and classical classifiers. Minz and Mahobiya (2017) extracted grey-level co-occurrence texture features and classified them with AdaBoost, reporting 89.90% accuracy. Ismael and Abdel-Qader (2018) combined statistical features with a back-propagation neural network, and Tripathi and Bag (2019) graded tumours with noise-robust texture and intensity features. Cheng et al. (2015) showed that enlarging the tumour region of interest and partitioning it into sub-regions raised accuracy on a three-class contrast-enhanced T1 dataset to 91.28%. These methods depend on careful feature design and often on a manually outlined tumour region.

Support Vector Machines find the maximum-margin boundary between classes and, with kernels, can model non-linear boundaries (Cortes and Vapnik, 1995). Decision trees split the data on single features (Quinlan, 1986), and Random Forests average many decorrelated trees to reduce variance (Breiman, 2001). These classifiers remain strong baselines when good features are available, which is why this project uses them both on raw pixels and on deep features.

## 2.3 Deep Learning and Transfer Learning

Convolutional networks learn hierarchies of features from pixels (LeCun, Bengio and Hinton, 2015) and now dominate medical image analysis (Litjens et al., 2017). Abiwinanda et al. (2018) showed that even a shallow CNN can separate glioma, meningioma and pituitary tumours on the public dataset of Cheng et al. (2015). Sultan, Salem and Al-Atabany (2019) reported a deeper network for multi-class classification and grading, and Badža and Barjaktarović (2020) showed that a purpose-built CNN can compete with much larger pretrained networks. Related work applies recurrent models across slices: Zhou et al. (2019) combined DenseNet features with a recurrent network and reported 92.13% accuracy, and Patel et al. (2019) used a CNN with LSTM layers to detect intracranial haemorrhage in CT. Yousaf et al. (2023) extended deep models to detect several brain conditions.

Transfer learning reuses networks pretrained on ImageNet (Deng et al., 2009). Shin et al. (2016) showed that pretrained features transfer well to medical detection tasks. For brain tumours, Deepak and Ameer (2019) used GoogLeNet features with SVM and k-nearest-neighbour classifiers and reported a mean accuracy of 98% under patient-level five-fold cross-validation. Rehman et al. (2020) reported 98.69% with a fine-tuned VGG16, Swati et al. (2019) proposed block-wise fine-tuning of VGG19, and Talo et al. (2019) reached 100% accuracy for normal versus abnormal classification on 613 images with ResNet34. InceptionV3 factorises large convolutions into smaller ones to reduce computation (Szegedy et al., 2016), and Xception replaces Inception modules with depthwise separable convolutions (Chollet, 2017). Saeedi et al. (2023) compared CNNs with classical classifiers on a four-class brain MRI dataset of the kind used here. A recurring caution in this literature is that figures obtained on different datasets, splits and validation schemes are not directly comparable.

## 2.4 Ensemble Learning

Ensembles reduce variance and the effect of individual model biases when their members make different errors (Dietterich, 2000). Ganaie et al. (2022) review deep ensembles and conclude that diversity among members matters more than their number. Hard (majority) voting counts one vote per member, whereas soft voting sums the members' predicted probabilities, so that confident members carry more weight. In brain MRI, Kang, Ullah and Gwak (2021) extracted features from several pretrained networks and classified them with classical classifiers, finding that an SVM with an RBF kernel often performed best and that combining deep features helped. Kakon et al. (2025) stacked several pretrained networks with a meta-learner and added gradient-based explanations. This project follows the hybrid idea of Kang, Ullah and Gwak (2021), but combines deep classifiers and classical classifiers on deep features in one six-predictor vote.

## 2.5 Grad-CAM and Explainable AI in Medical Imaging

Explainable AI aims to make model behaviour understandable to people (Barredo Arrieta et al., 2020). In medical imaging, post-hoc saliency methods are the most common approach (van der Velden et al., 2022; Salahuddin et al., 2022). Gradient-weighted Class Activation Mapping (Grad-CAM) computes the gradient of a class score with respect to the feature maps of a convolutional layer, uses the averaged gradients as weights for the feature maps, and keeps the positive part of the weighted sum (Selvaraju et al., 2017). The result is a coarse heatmap of the regions that raised the class score. Grad-CAM needs no change to the model and works for any convolutional architecture, which makes it suitable for comparing different networks. Draelos and Carin (2020) showed that when a network does not end in global average pooling, multiplying gradients and activations location by location (rather than averaging the gradients first) gives a map that follows the model's computation more faithfully.

For brain tumours, Esmaeili et al. (2021) compared CNN saliency maps with tumour locations, Zeineldin et al. (2022) evaluated explanation methods for MRI classification and segmentation networks, and Nhlapho et al. (2024) applied Grad-CAM and other methods to pretrained classifiers. Aasem and Javed Iqbal (2024) fused class activation maps from several networks for chest X-rays and compared the fused map with bounding-box annotations. Two cautions apply. First, saliency maps do not always localise abnormalities reliably compared with radiologist annotations (Arun et al., 2021; Saporta et al., 2022). Second, explanations can give false reassurance if they are presented as proof rather than as evidence to be checked (Ghassemi, Oakden-Rayner and Beam, 2021; Rudin, 2019). Deep networks can also exploit shortcuts, features that correlate with the label in the training data but are not clinically meaningful (Geirhos et al., 2020; Zech et al., 2018), which is one reason to inspect where models look.

## 2.6 Research Gap

Table 2 positions this study against the closest work. Two gaps emerge. First, few studies compare custom CNNs, pretrained networks and classical classifiers, and their ensemble, under one standardised protocol. Second, studies that add explanations usually show heatmaps for one model, or compare fused maps with expert annotations that most public datasets do not have. None of the reviewed studies measures whether architecturally different networks agree with each other on where to look, or uses that agreement as a per-image escalation signal.

| Study | Models | Ensemble | Explanations | Inter-model attention agreement |
|---|---|---|---|---|
| Kang, Ullah and Gwak (2021) | Deep features from several pretrained CNNs with classical classifiers | Feature combination | No | No |
| Nhlapho et al. (2024) | Pretrained CNNs | No | Yes (Grad-CAM and others) | No |
| Aasem and Javed Iqbal (2024) | Several CNNs (chest X-ray) | Fused CAMs | Yes | No (maps compared with box annotations) |
| Kakon et al. (2025) | Pretrained CNN ensemble with meta-learner | Stacking | Yes (gradient-based CAM) | No |
| **This study** | Custom CNN, InceptionV3, Xception, plus RF/DT/SVM on deep features | Majority and confidence-weighted vote of six predictors | Grad-CAM on three networks | **Yes (EAA-IoU, used as an escalation signal)** |

Table: Table 2. Positioning of this study against the closest related work

## 2.7 Conceptual Framework

Figure 1 summarises the framework. Pre-processed MRI images feed three deep networks. Their predictions, and the predictions of classical classifiers trained on their dense-layer features, form six predictors that are combined by majority and confidence-weighted voting. The predictions are evaluated with standard classification metrics. In parallel, Grad-CAM heatmaps from the three deep networks are compared through EAA-IoU, and both the prediction and the agreement score are presented to the user in the NeuroScan web application.

![Figure 1. Conceptual framework of the study](paper_assets/figures_final/fig_framework_final.png){width=6.3in}
