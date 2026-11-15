from engine import *

L_BASE = """# Baseline configuration (notebook Brain_Tumor_Train_And_Log.ipynb)
def build_cnn(size, feat_name='dense'):
    m = models.Sequential([
        layers.Input(shape=(size, size, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu', name=feat_name),
        layers.Dropout(0.5),
        layers.Dense(N_CLASSES, activation='softmax')])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def _base(kind, size):
    App = tf.keras.applications.InceptionV3 if kind == 'inception' else tf.keras.applications.Xception
    b = App(input_shape=(size, size, 3), weights='imagenet', include_top=False)
    b.trainable = False                      # frozen backbone
    return b"""

L_CROP = '''def crop_head(img, margin=4):
    """Crop an MRI slice to the bounding box of the head."""
    grey = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(grey, 45, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < 0.2 * img.shape[1] or h < 0.2 * img.shape[0]:
        return img                      # thresholding failed, keep the full image
    y0, x0 = max(y - margin, 0), max(x - margin, 0)
    return img[y0:y + h + margin, x0:x + w + margin]'''

L_CNN2 = """def build_cnn(size=IMG_SIZE):
    inp = layers.Input((size, size, 3))
    x = augmentation()(inp)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Activation('linear', name='last_conv')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', name='features')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='predictions')(x)
    return Model(inp, out, name='cnn')"""

L_FT = """def build_transfer(kind, size=IMG_SIZE):
    app = {'xception': tf.keras.applications.Xception, 'inceptionv3': tf.keras.applications.InceptionV3}[kind]
    base = app(include_top=False, weights='imagenet', input_shape=(size, size, 3))
    base.trainable = False
    inp = layers.Input((size, size, 3))
    x = augmentation()(inp)
    x = layers.Rescaling(2.0, offset=-1.0)(x)          # [0, 1] -> [-1, 1]
    x = base(x, training=False)                         # batch-norm statistics stay fixed
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', name='features')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='predictions')(x)
    return Model(inp, out, name=kind)

# train.py: two-stage schedule
history = fit(model, train, val, C.HEAD_EPOCHS, 1e-3, history, 'head')        # new head only
unfreeze_top(model, C.UNFREEZE_FROM[kind])                                      # e.g. 'mixed7'
history = fit(model, train, val, C.FINETUNE_EPOCHS, 1e-4, history, 'finetune')"""

L_CLASSICAL = """models = {'rf': RandomForestClassifier(n_estimators=300, random_state=C.SEED, n_jobs=-1),
          'dt': DecisionTreeClassifier(random_state=C.SEED),
          'svm': make_pipeline(StandardScaler(), SVC(probability=True, random_state=C.SEED))}
p_val, p_test = [], []
for name, clf in models.items():
    clf.fit(d['f_train'], d['y_train'])                 # 256 dense-layer features
    log_result(f'{kind}_feat_{name}', d['y_test'], clf.predict(d['f_test']))
    p_val.append(clf.predict_proba(d['f_val'])); p_test.append(clf.predict_proba(d['f_test']))
vote_test = np.mean(p_test, axis=0)                     # soft vote of RF, DT and SVM"""

L_VOTE = """def majority(probs):
    votes = probs.argmax(axis=2)                        # (models, images)
    return np.array([np.bincount(v, minlength=len(C.CLASSES)).argmax() for v in votes.T])

P_test = np.stack([d['test'] for d in data])            # six probability matrices
summed = P_test.sum(axis=0)
log_result('final_majority', y_test, majority(P_test))
log_result('final_confidence_weighted', y_test, summed.argmax(1))"""

L_GRADCAM = """def gradcam(cm, X, cls):
    A, G = cm.maps_and_grads(X, cls)       # feature maps and d(logit)/dA at the target layer
    w = G.mean(axis=(1, 2))                # one weight per channel = mean gradient
    cam = np.einsum('bhwc,bc->bhw', A, w)  # weighted sum of feature maps
    return _finish(cam)                    # ReLU, scale to [0, 1], resize

def hirescam(cm, X, cls):                  # element-wise gradient x activation
    A, G = cm.maps_and_grads(X, cls)
    return _finish((A * G).sum(-1))"""


def ch3():
    hp = [
        [
            "Input size",
            "256 × 256 (deep models); 128 × 128 (feature extractors); 256 × 256 grey, flattened (raw-pixel models)",
            "224 × 224 for all deep models",
        ],
        ["Head cropping", "No", "Yes, bounding box of the head ({lst:crop})"],
        [
            "Normalisation",
            "Pixel values ÷ 255",
            "Pixel values ÷ 255; rescaled to [-1, 1] inside the pretrained models",
        ],
        ["Validation split", "10% of Training, random (seed 42)", "10% of Training, stratified (seed 42)"],
        ["Augmentation", "None", "Horizontal flip, rotation up to ±18°, zoom ±10%, contrast ±10%"],
        [
            "Custom CNN",
            "4 × (Conv + MaxPool), Flatten, Dense 256, Dropout 0.5",
            "4 × (Conv + BatchNorm + ReLU + MaxPool), global average pooling, Dense 256, Dropout 0.5",
        ],
        [
            "Pretrained networks",
            "Frozen backbone; trained head",
            "Head trained for 3 epochs, then the top blocks fine-tuned (from block 10 in Xception, from mixed7 in InceptionV3)",
        ],
        ["Optimiser", "Adam, learning rate 0.001", "Adam, 0.001 (CNN and heads), 0.0001 (fine-tuning)"],
        ["Epochs", "CNN 20; pretrained 10; extractors 10", "CNN up to 30; fine-tuning up to 12"],
        [
            "Model selection",
            "Weights after the last epoch",
            "Early stopping on validation loss (patience 5); learning rate × 0.3 after 2 epochs without improvement",
        ],
        ["Batch size", "32", "32"],
        [
            "Classical classifiers",
            "RF (100 trees), DT, SVM (RBF); on raw pixels and on 128 px extractor features",
            "RF (300 trees), DT, SVM (RBF, standardised features); on the 256 features of each deep model",
        ],
        ["Ensemble", "Majority and confidence-weighted vote of 6 predictors", "Same"],
        ["Hardware", "Google Colab, NVIDIA T4 GPU", "4-core CPU, 15 GB RAM"],
    ]
    pm = [
        [
            "1. Research and planning",
            "3",
            "Completed",
            "Literature review extended with explainability and evaluation sources",
        ],
        [
            "2. Individual model development",
            "4",
            "Completed",
            "Baseline models trained; Colab disconnections handled by per-epoch checkpoints",
        ],
        [
            "3. Ensemble framework",
            "3",
            "Completed",
            "Confidence-weighted voting added alongside majority voting",
        ],
        [
            "4. Performance evaluation",
            "2",
            "Completed",
            "Statistical tests added; duplicate check of the test set added",
        ],
        [
            "5. Application development",
            "2",
            "Prototype",
            "Web demonstrator built; alert threshold recalibrated from the test results",
        ],
        [
            "6. Documentation and finalisation",
            "2",
            "Completed",
            "Explainability findings led to a second, improved configuration (Chapter 6)",
        ],
    ]
    return [
        H(1, "Chapter 3 Methodology"),
        P(
            "This chapter explains how the research was designed and carried out: the philosophy and approach, project management and "
            "feasibility, the dataset and its quality checks, the two experimental configurations, training, the evaluation and statistical "
            "tests, the explanation methods, and the reliability and ethical safeguards. Code excerpts are short and taken from the project "
            "code; longer excerpts are in Appendix B."
        ),
        H(2, "3.1 Research philosophy"),
        P(
            "The study takes a positivist stance: claims about model performance are tested against measurable outcomes on data the models have "
            "not seen. It also adopts a critical-realist caution. Benchmark accuracy is an imperfect proxy for clinical value, and the mechanisms "
            "behind a prediction, for example whether a network uses the tumour or an artefact, matter as much as the score. The explainability "
            "work in Chapter 5 follows directly from this second commitment."
        ),
        H(2, "3.2 Research approach and strategy"),
        P(
            "The approach is deductive. Hypotheses H1 to H4 were stated before the final evaluation, and each is tested with a pre-defined metric and "
            "statistical test. The strategy is experimental and uses secondary data: a public, labelled MRI dataset (Nickparvar, n.d.) and "
            "published literature. The project proposal also planned a qualitative strand with clinicians. It was not carried out in this "
            "phase, so the study is quantitative; the questionnaire is kept in Appendix D as a proposed instrument for future validation."
        ),
        H(2, "3.3 Project management"),
        P(
            "The project followed the six-phase, sixteen-week plan in the proposal. {tab:plan} records the planned phases and what happened. Two "
            "findings changed the plan. The explainability analysis showed that the baseline models paid a large share of attention to the "
            "image background, and a data check showed that part of the test set duplicates training images. A second, improved configuration "
            "was therefore added and evaluated on the full test set and on a duplicate-free subset. All results were logged as CSV and JSON "
            "files and the trained models were published on Hugging Face (Kavishka, 2026)."
        ),
        TAB(
            "plan",
            "Planned phases and final project record",
            ["Phase", "Planned weeks", "Status", "Changes and justification"],
            pm,
            widths=[4.0, 1.8, 2.8, 7.0],
            font=9,
        ),
        H(2, "3.4 Feasibility, cost and access"),
        P(
            "Time feasibility was achieved by keeping the models small enough for free or modest hardware. The baseline models trained in minutes "
            "each on a Colab T4 GPU; the slowest baseline component was the raw-pixel SVM, which took 5812.1 seconds to train, against 2.7 seconds "
            "for an SVM on CNN features. The improved configuration was trained on a 4-core CPU, where one fine-tuning epoch of Xception took "
            "several minutes. Cost feasibility: all software is open source, training used free Colab or an ordinary CPU, and models and results "
            "are hosted free on Hugging Face. Access feasibility: the dataset is public and anonymised, so no data-sharing agreement or approval "
            "for patient data was needed. Scope feasibility led to one deliberate limit: no cross-validation, which would have multiplied the "
            "training time (Section 7.4)."
        ),
        H(2, "3.5 Dataset"),
        P(
            "The data are the Brain Tumor MRI Dataset curated by Nickparvar (n.d.), which combines images from the figshare dataset of Cheng et al. "
            "(2015), the SARTAJ collection and the Br35H collection. The release used contains 7,200 two-dimensional JPEG slices in four classes, "
            "already split into a Training folder (5,600 images, 1,400 per class) and a Testing folder (1,600 images, 400 per class) "
            "(dataset_counts.json). Because the classes are balanced, accuracy and macro-averaged scores are close. A mirror of the dataset on "
            "Hugging Face (BJyotibrat, 2026) was used for the analysis; re-running the saved models on it reproduced every logged prediction "
            "(Section 4.9), which confirms that it is the same test set. The mirror lists a CC BY 4.0 licence."
        ),
        P(
            "Three data-quality checks were carried out ({tab:dq}). First, 103 of the 400 meningioma test images are augmented copies (file names "
            'beginning "Te-aug-me"). Second, the classes differ in image size: most tumour images are 512 × 512 pixels, whereas no-tumour images '
            "are much smaller and vary in size, because they come from a different collection. Third, each test image was compared with all "
            "training images: 336 test images (21%) have a near-identical copy in the Training folder (mean absolute grey-level difference below "
            "0.02 at 64 × 64 pixels), mostly in the no-tumour and meningioma classes. Results are therefore reported on the full official test "
            'set and on the 1,264 test images without a training duplicate (the "clean" subset).'
        ),
        TAB(
            "dq",
            "Data-quality checks on the Testing folder",
            [
                "Class",
                "Images",
                "Median size (px)",
                "Share at 512 × 512 (%)",
                "Augmented copies",
                "Duplicates in Training",
            ],
            [
                [
                    "Glioma",
                    "400",
                    "512 × 512",
                    "<<sinv.size_by_class.glioma.share_512x512_pct>>",
                    "0",
                    "<<dup.by_class.glioma>>",
                ],
                [
                    "Meningioma",
                    "400",
                    "512 × 512",
                    "<<sinv.size_by_class.meningioma.share_512x512_pct>>",
                    "103",
                    "<<dup.by_class.meningioma>>",
                ],
                [
                    "No tumour",
                    "400",
                    "<<sinv.size_by_class.notumor.median_w>> × <<sinv.size_by_class.notumor.median_h>>",
                    "<<sinv.size_by_class.notumor.share_512x512_pct>>",
                    "0",
                    "<<dup.by_class.notumor>>",
                ],
                [
                    "Pituitary",
                    "400",
                    "512 × 512",
                    "<<sinv.size_by_class.pituitary.share_512x512_pct>>",
                    "0",
                    "<<dup.by_class.pituitary>>",
                ],
            ],
            widths=[2.6, 1.6, 2.8, 2.9, 2.4, 3.2],
            font=9,
        ),
        FIG(
            "dist",
            "paper_assets/figures/fig_dataset_distribution.png",
            "Images per class in the Training and Testing folders",
            width=5.4,
        ),
        FIG(
            "samples",
            "paper_assets/figures/fig_sample_mri.png",
            "Two randomly selected test images per class, showing axial, coronal and sagittal views",
            width=5.6,
        ),
        H(2, "3.6 Experimental configurations"),
        P(
            "Two configurations were trained and evaluated in turn. The baseline configuration follows the original design of the framework: "
            "whole images, no augmentation, and pretrained networks with frozen backbones. The improved configuration was designed after the "
            "baseline had been analysed (Chapter 5) and changes three things: every image is cropped to the bounding box of the head ({lst:crop}), "
            "the training images are augmented, and the pretrained backbones are fine-tuned. The custom CNN also receives batch normalisation and "
            "a global-average-pooling head, so that its decision cannot depend on fixed pixel positions such as the image border. {tab:hp} "
            "compares the two configurations. In both, the Testing folder was used only for the final evaluation."
        ),
        TAB(
            "hp",
            "Settings of the baseline and improved configurations",
            ["Setting", "Baseline", "Improved"],
            hp,
            widths=[3.4, 6.0, 6.0],
            font=8.5,
        ),
        CODE("crop", "Head cropping used by the improved configuration (src/data.py)", L_CROP),
        H(2, "3.7 Models"),
        H(3, "3.7.1 Custom CNN"),
        P(
            "The baseline CNN ({fig:cnn}, {lst:base}) has four convolution and pooling blocks, a 256-unit dense layer, dropout and a softmax output, "
            "and is trained from scratch. The improved CNN ({fig:cnn_imp}, {lst:cnn2}) adds batch normalisation, augmentation and global average pooling."
        ),
        FIG(
            "cnn",
            "paper_assets/figures/fig_arch_cnn.png",
            "Architecture of the baseline custom CNN",
            width=6.2,
        ),
        CODE("base", "Baseline CNN and frozen pretrained backbone", L_BASE),
        FIG("cnn_imp", "paper_assets/figures/fig_arch_cnn_improved.png", "Architecture of the improved custom CNN", width=6.2),
        CODE("cnn2", "Improved CNN (src/models.py)", L_CNN2),
        H(3, "3.7.2 InceptionV3 and Xception"),
        P(
            "Both pretrained networks were loaded with ImageNet weights (Szegedy et al., 2016; Chollet, 2017). In the baseline the backbone was "
            "frozen and only a new head was trained, so the networks acted as fixed feature extractors. In the improved configuration the new head "
            "is trained first and the upper blocks of the backbone are then fine-tuned with a learning rate ten times smaller ({lst:ft}). Batch "
            "normalisation layers stay frozen during fine-tuning, which is the usual way to avoid destroying the pretrained statistics with small batches."
        ),
        CODE(
            "ft",
            "Pretrained network with a new head and two-stage fine-tuning (src/models.py and src/train.py)",
            L_FT,
        ),
        H(3, "3.7.3 Classical classifiers"),
        P(
            "Random Forest, Decision Tree and SVM classifiers (Breiman, 2001; Quinlan, 1986; Cortes and Vapnik, 1995) were trained on deep features. "
            "In the baseline the features came from separate 128-pixel extractor networks, and the classifiers, together with Logistic Regression, "
            "were also trained on 65,536 raw greyscale pixels. In the improved configuration the features are the 256 values of the dense layer of "
            "each fine-tuned network ({lst:classical}). Within each feature source the three classifiers are combined by soft voting."
        ),
        CODE("classical", "Classical classifiers on deep features (src/classical.py)", L_CLASSICAL),
        H(3, "3.7.4 Six-predictor ensemble"),
        P(
            "The final ensemble ({fig:ens}) combines six probability vectors: the CNN, InceptionV3 and Xception classifiers and the three soft-voting "
            "classical ensembles built on their features. Majority voting gives each predictor one vote. Confidence-weighted voting sums the six "
            "probability vectors and takes the largest, which also yields a confidence score ({lst:vote})."
        ),
        FIG(
            "ens",
            "paper_assets/figures/fig_arch_ensemble.png",
            "The six-predictor hybrid ensemble",
            width=6.2,
        ),
        CODE("vote", "Majority and confidence-weighted voting (src/ensemble.py)", L_VOTE),
        H(2, "3.8 Hardware, software and reproducibility"),
        P(
            "The baseline was trained in Google Colab on an NVIDIA T4 GPU, and the improved configuration on a 4-core CPU with 15 GB of memory, both "
            "with TensorFlow 2.21.0, Keras 3.13.2 and scikit-learn 1.6.1 (Abadi et al., 2016; Pedregosa et al., 2011). Seeds were fixed to 42. Every "
            "result was written as data: one row per model in a results table, a classification report and confusion matrix per model, per-epoch "
            "training histories and per-image predictions. The code is organised as a small package (src/) with one module per step, and the "
            "analysis scripts are listed in Appendix F."
        ),
        H(2, "3.9 Evaluation protocol and statistical analysis"),
        P(
            "All models were evaluated once on the 1,600 Testing images and, separately, on the 1,264-image clean subset. Reported metrics are "
            "accuracy, per-class and macro-averaged precision, recall and F1-score, confusion matrices, and one-vs-rest ROC and precision-recall "
            "curves. Because a single accuracy figure hides sampling uncertainty, 95% intervals are reported: Wilson score intervals for every "
            "model and percentile bootstrap intervals with 2,000 resamples (Efron, 1979) for the predictors with per-image predictions. Models "
            "evaluated on the same images were compared with the exact McNemar test (McNemar, 1947), which Dietterich (1998) recommends for "
            "comparing two classifiers on one test set, with Holm correction across the planned comparisons."
        ),
        H(2, "3.10 Explainability methodology"),
        P(
            "{fig:xaipipe} summarises the explainability pipeline. Explanations were computed for the class predicted by the six-model majority vote, "
            "so they show what supported the ensemble decision."
        ),
        FIG(
            "xaipipe",
            "paper_assets/figures/fig_xai_pipeline.png",
            "Explainability pipeline: methods (left) and how they are evaluated (right)",
            width=6.2,
        ),
        B(
            [
                "Grad-CAM, Grad-CAM++ and HiResCAM (Selvaraju et al., 2017; Chattopadhay et al., 2018; Draelos and Carin, 2020) were computed from the "
                "pre-softmax class score at the last spatial layer of each network ({lst:gradcam}). For InceptionV3 and Xception, which end in global "
                "average pooling, Grad-CAM and HiResCAM are mathematically equivalent; for the baseline CNN, which ends in Flatten and Dense layers, they differ.",
                "Occlusion (Zeiler and Fergus, 2014): a 32 × 32 black patch slid with a stride of 16 pixels; the map is the drop in class probability.",
                "LIME (Ribeiro, Singh and Guestrin, 2016): quick-shift super-pixels, 600 perturbed samples for the CNN and 400 for Xception, hidden super-pixels set to black.",
                "SHAP (Lundberg and Lee, 2017): the gradient explainer (expected gradients) on the CNN with 50 background images, and the exact tree explainer on the Random Forest that classifies CNN features.",
            ]
        ),
        P(
            "Example images were chosen for each class: the correctly classified test image with the highest ensemble confidence, and a confidently "
            "misclassified image from the class’s most frequent error. No-tumour images were never misclassified by the baseline ensemble, so its "
            "error example is the most frequent miss into that class, a glioma predicted as no tumour. Quantitative evaluation used a stratified "
            "sample of 160 test images (30 per class plus 40 misclassified images, seed 42) and four checks:"
        ),
        NL(
            [
                "Faithfulness: deletion and insertion curves (Petsiuk, Das and Saenko, 2018). Pixels were removed (set to black) or revealed (on a blurred "
                "copy) in order of saliency in 20 steps, and the area under the class-probability curve was computed and compared with a random map.",
                "Sanity check: cascading model-parameter randomisation (Adebayo et al., 2018). Layers were re-initialised from the top down, and the "
                "Spearman rank correlation and structural similarity (SSIM) between the original and randomised maps were measured on 40 images.",
                "Cross-model consistency: the Ensemble Attention Agreement score (EAA-IoU). The maps of the three networks are binarised at 0.5, the "
                "intersection over union of each pair is computed and the three values are averaged. It was computed for all 1,600 test images and "
                "is complemented by the Spearman correlation between maps and the IoU of their top-20% regions.",
                "Shortcut check: the share of Grad-CAM mass that falls outside the head (found by thresholding) or in a 12-pixel outer rim covering "
                "the skull and image edge, compared with the share of image area those regions occupy. Three further experiments tested the cause "
                "(Section 5.6).",
            ]
        ),
        CODE("gradcam", "Grad-CAM and HiResCAM as implemented for the analysis", L_GRADCAM),
        H(2, "3.11 Reliability, validity and generalisability"),
        P(
            "Reliability is supported by fixed seeds, a single code base per configuration and logging of all results as data; the saved models "
            "were also reloaded and re-run, which reproduced every logged prediction. Internal validity is supported by an untouched test set and "
            "paired statistical tests, and is threatened by the duplicated and augmented test images, which is why the clean subset is reported. "
            "External validity is limited: all images come from one public collection of two-dimensional slices without scanner or patient "
            "metadata, so results may not transfer to other hospitals (Zech et al., 2018; Kelly et al., 2019)."
        ),
        H(2, "3.12 Ethical considerations"),
        P(
            "The dataset is public and anonymised, and no attempt was made to re-identify anyone. No human participants were involved, so informed "
            "consent was not required; if the questionnaire in Appendix D is used later, it will need ethical approval and informed consent. The "
            "system is a research prototype for decision support, and its interface states that it must not be used for diagnosis. Explanations "
            "are presented with their limitations, because over-trust in saliency maps is itself an ethical risk (Ghassemi, Oakden-Rayner and "
            "Beam, 2021). Models and code are published openly, and no images are redistributed with the report."
        ),
    ]
