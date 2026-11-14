from engine import *

L_DATA = '''# Same split and scaling for every deep model (seed 42, 10% of Training held out for validation)
def tf_datasets(size):
    common = dict(image_size=(size, size), batch_size=BATCH, label_mode='int', seed=SEED)
    tr = tf.keras.utils.image_dataset_from_directory(os.path.join(DATA_DIR, 'Training'),
            validation_split=0.1, subset='training', shuffle=True, **common)
    va = tf.keras.utils.image_dataset_from_directory(os.path.join(DATA_DIR, 'Training'),
            validation_split=0.1, subset='validation', shuffle=True, **common)
    te = tf.keras.utils.image_dataset_from_directory(os.path.join(DATA_DIR, 'Testing'),
            shuffle=False, image_size=(size, size), batch_size=BATCH, label_mode='int')
    norm = lambda x, y: (x / 255.0, y)                     # RGB scaled to [0, 1]
    return tr.map(norm), va.map(norm), te.map(norm)        # Testing is only used for final evaluation'''

L_CNN = '''def build_cnn(size, feat_name='dense'):
    m = models.Sequential([
        layers.Input(shape=(size, size, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu', name=feat_name),   # 256-value feature layer
        layers.Dropout(0.5),
        layers.Dense(N_CLASSES, activation='softmax')])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m'''

L_TRANSFER = '''def _base(kind, size):
    App = tf.keras.applications.InceptionV3 if kind == 'inception' else tf.keras.applications.Xception
    b = App(input_shape=(size, size, 3), weights='imagenet', include_top=False)
    b.trainable = False                       # backbone frozen: transfer learning, not fine-tuning
    return b

def build_transfer_standalone(kind, size=256):  # 256 px classifier: GAP + softmax head
    base = _base(kind, size)
    out = layers.Dense(N_CLASSES, activation='softmax')(layers.GlobalAveragePooling2D()(base.output))
    return Model(base.input, out)

def build_transfer_extractor(kind, feat_name, size=128):   # 128 px extractor with a Dense-256 layer
    base = _base(kind, size)
    x = layers.Dense(256, activation='relu', name=feat_name)(layers.Flatten()(base.output))
    out = layers.Dense(N_CLASSES, activation='softmax')(layers.Dropout(0.5)(x))
    return Model(base.input, out)'''

L_CLASSICAL = '''# Deep features (256 values per image) -> RF, DT and SVM -> soft vote
ftr = extract_features(extractor, 'dense', X_train)   # output of the Dense-256 layer
fte = extract_features(extractor, 'dense', X_test)
specs = {'rf':  RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
         'dt':  DecisionTreeClassifier(random_state=SEED),
         'svm': SVC(probability=True, random_state=SEED)}
fitted = {k: clf.fit(ftr, y_train) for k, clf in specs.items()}
probs = sum(fitted[k].predict_proba(fte) for k in ['rf', 'dt', 'svm']) / 3
soft_vote_pred = probs.argmax(axis=1)'''

L_VOTE = '''# P has shape (6 predictors, N images, 4 classes): three deep softmax outputs and
# three averaged RF/DT/SVM probability vectors from the 128 px feature extractors
summed = P.sum(axis=0)
pred_confidence_weighted = summed.argmax(axis=1)            # argmax of the summed probabilities
confidence = summed.max(axis=1) / summed.sum(axis=1)        # share of the winning class
votes = P.argmax(axis=2)                                    # one hard vote per predictor
pred_majority = np.array([np.bincount(votes[:, i], minlength=4).argmax()
                          for i in range(votes.shape[1])])  # ties -> lowest class index'''

L_GRADCAM = '''def gradcam(cm, X, cls):
    A, G = cm.maps_and_grads(X, cls)       # feature maps and d(logit)/dA at the target layer
    w = G.mean(axis=(1, 2))                # one weight per channel = mean gradient
    cam = np.einsum('bhwc,bc->bhw', A, w)  # weighted sum of feature maps
    return _finish(cam)                    # ReLU, scale to [0, 1], resize to 256 x 256

def hirescam(cm, X, cls):                  # element-wise gradient x activation
    A, G = cm.maps_and_grads(X, cls)
    return _finish((A * G).sum(-1))'''

def ch3():
    hp = [['Input size', '256 × 256 × 3 (standalone deep models); 128 × 128 × 3 (feature extractors); 256 × 256 greyscale flattened to 65,536 values (raw-pixel models)'],
          ['Normalisation', 'Pixel values ÷ 255 (range 0–1) for every model'],
          ['Train / validation split', '90% / 10% of the Training folder, seed 42; Testing folder held out'],
          ['Optimiser and loss', 'Adam (Keras default learning rate 0.001); sparse categorical cross-entropy'],
          ['Epochs', 'Custom CNN 20; InceptionV3 and Xception 10; 128 px extractors 10'],
          ['Batch size', '32 for all deep models'],
          ['Regularisation', 'Dropout 0.5 before the output layer (CNN and extractors); frozen ImageNet backbones'],
          ['Data augmentation', 'None applied during training'],
          ['Model selection', 'No early stopping; the weights after the last epoch were used'],
          ['Random Forest', '100 trees, default depth, random_state 42'],
          ['Decision Tree', 'Default settings (Gini, unlimited depth), random_state 42'],
          ['Support Vector Machine', 'RBF kernel, C = 1, probability estimates enabled, random_state 42'],
          ['Logistic Regression (raw pixels only)', 'max_iter 1,000'],
          ['Ensembles', 'Soft vote of RF + DT + SVM within each feature source; final six-predictor majority and confidence-weighted votes']]
    pm = [['1. Research and planning', '3', 'Completed (HND stage, 2024)', 'Literature review extended with XAI and evaluation sources (2026)'],
          ['2. Individual model development', '4', 'Completed', 'All models retrained under one protocol after the audit; Colab disconnections handled by per-epoch checkpoints'],
          ['3. Ensemble framework', '3', 'Completed', 'Added confidence-weighted voting alongside majority voting'],
          ['4. Performance evaluation', '2', 'Completed', 'Held-out test evaluation replaced the training-set evaluation; statistical tests added'],
          ['5. Application development', '2', 'Prototype', 'Web demonstrator built; not evaluated with users'],
          ['6. Documentation and finalisation', '2', 'Completed', 'Explainability chapter, audit and change log added']]
    return [
        H(1, 'Chapter 3 Methodology'),
        P('This chapter explains how the research was designed and carried out: the philosophy and approach, project management and '
          'feasibility, the dataset, the models and their training, the evaluation and statistical tests, the new explainability methods, '
          'and the reliability and ethical safeguards. Code excerpts are short and taken from the project code; longer excerpts are in Appendix B.'),
        H(2, '3.1 Research philosophy'),
        P('The study takes a positivist stance: claims about model performance are tested against measurable outcomes on data the models '
          'have not seen. It also adopts a critical-realist caution. Benchmark accuracy is an imperfect proxy for clinical value, and the '
          'mechanisms behind a prediction (for example, whether a network uses the tumour or an artefact) matter as much as the score. The '
          'explainability work in Chapter 5 follows directly from this second commitment.'),
        H(2, '3.2 Research approach and strategy'),
        P('The approach is deductive. Hypotheses H1–H3 were stated before the final evaluation, and each is tested with a pre-defined metric and '
          'statistical test. The strategy is experimental and uses secondary data: a public, labelled MRI dataset (Nickparvar, n.d.) and '
          'published literature. The HND proposal planned a mixed-methods design with clinician interviews. No interviews or questionnaire '
          'responses were collected, so this report is purely quantitative. The questionnaire is kept in Appendix D as a proposed instrument '
          'for future clinician validation, rather than presented as collected data.', new=True),
        H(2, '3.3 Project management', new=True),
        P('The project followed the six-phase, sixteen-week plan in the original proposal. {tab:plan} records the planned phases and what '
          'actually happened. The main change was unplanned: the audit (AUDIT.md) showed that the HND evaluation was not on held-out data, so '
          'phases 2–4 were repeated with one standardised training notebook that logs every result to CSV and JSON files and uploads models '
          'and results to Hugging Face (Kavishka, 2026). Free Colab sessions disconnected several times; per-epoch checkpoints let training '
          'resume (the run log records resumes of the CNN at epochs 11 and 15), so no results were lost.', new=True),
        TAB('plan', 'Planned phases and final project record', ['Phase', 'Planned weeks', 'Status', 'Changes and justification'], pm, new=True, widths=[4.0, 1.8, 2.8, 7.0], font=9),
        H(2, '3.4 Feasibility, cost and access', new=True),
        P('Time feasibility was achieved by keeping models small enough to train on a free GPU: each deep model trained in minutes on a '
          'Colab T4 GPU, while the slowest component was the raw-pixel SVM, which took 5812.1 seconds to train. The CNN-feature SVM took '
          '2.7 seconds (results_master.csv). Cost feasibility: all software is open source, training used free Colab, models and results are '
          'hosted free on Hugging Face, and the explainability analysis ran on an ordinary 4-core CPU. Access feasibility: the dataset is public '
          'and anonymised, so no data-sharing agreement or ethics committee approval for patient data was needed. Scope feasibility '
          'led to two deliberate limits: no fine-tuning of the pretrained backbones and no cross-validation, both of which would have multiplied '
          'training time (see Section 6.4).', new=True),
        H(2, '3.5 Dataset'),
        P('The data are the Brain Tumor MRI Dataset curated by Nickparvar (n.d.), which combines images from the figshare dataset of Cheng et al. '
          '(2015), the SARTAJ collection and the Br35H collection. The current release contains 7,200 two-dimensional JPEG slices in four '
          'classes, pre-split into a Training folder (5,600 images, 1,400 per class) and a Testing folder (1,600 images, 400 per class) '
          '(dataset_counts.json). The class balance means that accuracy and macro-averaged scores coincide closely. The XAI analysis used an '
          'identical copy of the Testing folder from a Hugging Face mirror (BJyotibrat, 2026). Re-running the saved models on this copy '
          'reproduced every logged prediction (Section 4.9), which confirms that it is the same test set. The mirror card lists a CC BY 4.0 '
          'licence; the licence on the original Kaggle page should be confirmed before any redistribution.', new=True),
        P('Two properties of the data matter for interpretation. First, 103 of the 400 meningioma test images are augmented copies (file names '
          'beginning "Te-aug-me"), so the meningioma test set is less independent than the others. Second, all no-tumour images come from a '
          'single source (Br35H), which creates a possible shortcut. The HND study used an older 9,047-image copy with 6,984 training and '
          '2,063 test images, so the old and new results are not directly comparable (Section 4.10).', new=True),
        FIG('dist', 'paper_assets/figures/fig_dataset_distribution.png', 'Images per class in the Training and Testing folders', new=True, width=5.4),
        FIG('samples', 'paper_assets/figures/fig_sample_mri.png', 'Two randomly selected test images per class, showing axial, coronal and sagittal views', new=True, width=5.6),
        H(2, '3.6 Pre-processing'),
        P('Images were resized to 256 × 256 pixels for the standalone deep models, and to 128 × 128 pixels for the feature extractors that feed '
          'the classical classifiers. Pixel values were divided by 255 for every model, which removes the mixture of scalings (÷255 and '
          '÷255 − 0.5) used in the HND notebooks. Raw-pixel classifiers used greyscale images flattened to 65,536 values. No noise reduction, '
          'skull stripping, registration or augmentation was applied; earlier text implying otherwise has been removed. {lst:data} shows the '
          'shared data loader.', new=True),
        CODE('data', 'Shared data loading and split (from the standardised training notebook)', L_DATA),
        H(2, '3.7 Models'),
        H(3, '3.7.1 Custom CNN'),
        P('The custom CNN ({fig:cnn}, {lst:cnn}) has four convolution–pooling blocks, a 256-unit dense layer, dropout and a softmax output. It is '
          'trained from scratch. The same design at 128 × 128 pixels serves as the CNN feature extractor.'),
        FIG('cnn', 'paper_assets/figures/fig_arch_cnn.png', 'Architecture of the custom four-block CNN', new=True, width=6.2),
        CODE('cnn', 'Custom CNN definition', L_CNN),
        H(3, '3.7.2 InceptionV3 and Xception (transfer learning with frozen backbones)'),
        P('Both pretrained networks were loaded with ImageNet weights and with their convolutional backbone frozen (trainable = False). Only a new '
          'head was trained ({lst:transfer}). The HND report and the project title call these models "fine-tuned", but no backbone weights '
          'were updated, so they are more accurately described as frozen feature extractors with trained classification heads. True fine-tuning '
          'is listed as future work.', new=True),
        CODE('transfer', 'Frozen-backbone transfer models: 256 px classifiers and 128 px feature extractors', L_TRANSFER),
        H(3, '3.7.3 Classical classifiers and the feature-level ensembles'),
        P('Random Forest, Decision Tree and SVM classifiers were trained (i) on raw greyscale pixels, together with Logistic Regression, and '
          '(ii) on the 256 values of the dense feature layer of each 128-pixel extractor. Within each feature source the three classifiers '
          'were combined by soft voting ({lst:classical}). The HND notebook mistakenly instantiated a Decision Tree in place of the Random '
          'Forest; this is fixed.'),
        CODE('classical', 'Deep feature extraction and the RF/DT/SVM soft-voting ensemble', L_CLASSICAL),
        H(3, '3.7.4 Final six-predictor ensemble'),
        P('The final ensemble ({fig:ens}) combines six probability vectors: the CNN, InceptionV3 and Xception classifiers, and the three '
          'soft-voting classical ensembles built on CNN, InceptionV3 and Xception features. Majority voting gives each predictor one vote. '
          'Confidence-weighted voting sums the six probability vectors and takes the largest, which also yields a confidence score '
          '({lst:vote}).'),
        FIG('ens', 'paper_assets/figures/fig_arch_ensemble.png', 'The six-predictor hybrid ensemble', new=True, width=6.2),
        CODE('vote', 'Majority and confidence-weighted voting', L_VOTE),
        H(2, '3.8 Training set-up and hyperparameters', new=True),
        P('{tab:hp} lists the settings shared across models. Hyperparameters were kept at sensible defaults and identical across models rather '
          'than tuned per model. This favours a fair comparison over peak performance, and it avoids tuning on the test set.', new=True),
        TAB('hp', 'Training set-up and hyperparameters', ['Setting', 'Value'], hp, new=True, widths=[4.5, 11.0], font=9),
        H(2, '3.9 Hardware, software and reproducibility', new=True),
        P('Training ran in Google Colab on an NVIDIA T4 GPU with TensorFlow 2.21.0, Keras 3.13.2 and scikit-learn 1.6.1 (manifest.json) '
          '(Abadi et al., 2016; Pedregosa et al., 2011). Seeds were fixed to 42 for Python, NumPy and TensorFlow. Every result was written '
          'as data: one row per model in results_master.csv, a classification report and confusion matrix per model, per-epoch training '
          'histories, per-image predictions and a timestamped run log. The explainability analysis was run separately on a 4-core CPU with '
          '15 GB RAM, using the saved models downloaded from Hugging Face. The analysis scripts are in paper_assets/scripts (Appendix H).', new=True),
        H(2, '3.10 Evaluation protocol and statistical analysis', new=True),
        P('All models were evaluated once on the 1,600 Testing images, which were never used for training, validation or model selection. '
          'Reported metrics are accuracy, per-class and macro-averaged precision, recall and F1-score, confusion matrices, and one-vs-rest ROC '
          'and precision–recall curves with their areas. Because a single accuracy figure hides sampling uncertainty, 95% intervals are '
          'reported: Wilson score intervals for every model, and percentile bootstrap intervals with 2,000 resamples (Efron, 1979) for the '
          'eight predictors with per-image predictions. Pairs of models evaluated on the same images were compared with the exact McNemar '
          'test (McNemar, 1947), which Dietterich (1998) recommends for comparing two classifiers on one test set. Holm correction was applied '
          'across the ten planned comparisons. The "correlation" and "regression" sections of the HND report were removed, because no '
          'correlation or regression model was fitted.', new=True),
        H(2, '3.11 Explainability methodology', new=True),
        P('{fig:xaipipe} summarises the explainability pipeline. Explanations were computed for the class predicted by the six-model majority '
          'vote, so they show what supported the ensemble decision.', new=True),
        FIG('xaipipe', 'paper_assets/figures/fig_xai_pipeline.png', 'Explainability pipeline: methods (left) and how they are evaluated (right)', new=True, width=6.2),
        B(['**Grad-CAM, Grad-CAM++ and HiResCAM** (Selvaraju et al., 2017; Chattopadhay et al., 2018; Draelos and Carin, 2020) were computed '
           'from the pre-softmax class score at the last spatial layer of each network: conv2d_7 for the CNN, mixed10 for InceptionV3 and '
           'block14_sepconv2_act for Xception ({lst:gradcam}). For InceptionV3 and Xception, which end in global average pooling, Grad-CAM and '
           'HiResCAM are mathematically equivalent. For the CNN, which ends in Flatten + Dense, they differ.',
           '**Occlusion** (Zeiler and Fergus, 2014): a 32 × 32 black patch slid with stride 16; the map is the drop in class probability.',
           '**LIME** (Ribeiro, Singh and Guestrin, 2016): quick-shift super-pixels, 600 perturbed samples for the CNN and 400 for Xception, hidden super-pixels set to black.',
           '**SHAP** (Lundberg and Lee, 2017): the gradient explainer (expected gradients) on the CNN with 50 background test images and 200 samples, and the exact tree explainer on the Random Forest that classifies CNN features.'], new=True),
        P('Example images: for each class, the correctly classified test image with the highest ensemble confidence was selected, together with '
          'a confidently misclassified image from that class’s most frequent error (for example, glioma predicted as meningioma). No-tumour '
          'images were never misclassified, so its error example is the most frequent miss into that class: a glioma predicted as no tumour.', new=True),
        P('Quantitative evaluation used a stratified sample of 160 test images: 30 random images per class plus 40 further images that the '
          'ensemble misclassified, with seed 42. Four checks were applied:', new=True),
        NL(['**Faithfulness** – deletion and insertion curves (Petsiuk, Das and Saenko, 2018): pixels were removed (set to black) or revealed '
            '(on a blurred copy) in order of saliency, in 20 steps, and the area under the class-probability curve was computed. A faithful map '
            'gives a low deletion AUC and a high insertion AUC compared with a random map.',
            '**Sanity check** – cascading model-parameter randomisation (Adebayo et al., 2018): layers were re-initialised from the top down, and '
            'the Spearman rank correlation and structural similarity (SSIM) between the original and randomised Grad-CAM maps were measured on 40 images.',
            '**Cross-model consistency** – the Ensemble Attention Agreement (EAA-IoU) from the project code: Grad-CAM-family maps of the three '
            'networks are binarised at 0.5, the intersection over union of each pair is computed, and the three values are averaged. It was '
            'computed for all 1,600 test images in the training run. It is complemented here by the Spearman rank correlation between maps and '
            'the IoU of their top-20% regions.',
            '**Shortcut check** – the share of Grad-CAM mass falling outside the head (a threshold-based head mask) or in a 12-pixel outer rim '
            'that covers the skull and image edge, compared with the share of image area these regions occupy.'], new=True),
        CODE('gradcam', 'Grad-CAM and HiResCAM as implemented for the analysis', L_GRADCAM),
        H(2, '3.12 Reliability, validity and generalisability'),
        P('Reliability is supported by fixed seeds, a single shared training notebook, and logging of all results as data. The saved models '
          'were also reloaded and re-run, which reproduced every logged prediction (Section 4.9). Internal validity is supported by the '
          'untouched test set and paired statistical tests. Two threats remain: the augmented meningioma test images, and the single-source '
          'no-tumour class. External validity is limited: all images come from one public collection of two-dimensional slices, without '
          'scanner or patient metadata, so results may not transfer to other hospitals (Zech et al., 2018; Kelly et al., 2019). '
          'Cross-validation was not used, so variation due to the training split is not measured.', new=True),
        H(2, '3.13 Ethical considerations'),
        P('The dataset is public and anonymised, and no attempt was made to re-identify anyone. No human participants were involved in this '
          'stage, so informed consent was not required; if the clinician questionnaire in Appendix D is used later, it will need ethical '
          'approval and informed consent. The system is a research prototype for decision support. Its interface states that it is for '
          'research use only and must not be used for diagnosis. Explanations are presented with their limitations, because over-trust in '
          'saliency maps is itself an ethical risk (Ghassemi, Oakden-Rayner and Beam, 2021). Models and code are published openly for '
          'transparency, and no images are redistributed with the report.', new=True),
    ]
