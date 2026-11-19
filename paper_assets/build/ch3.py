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
history = fit(model, train, val, C.HEAD_EPOCHS, 1e-3, history, 'head')  # new head only
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
        CODE("crop", "Head cropping used by the improved configuration", L_CROP),
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
        FIG(
            "cnn_imp",
            "paper_assets/figures/fig_arch_cnn_improved.png",
            "Architecture of the improved custom CNN",
            width=6.2,
        ),
        CODE("cnn2", "Improved CNN", L_CNN2),
        H(3, "3.7.2 InceptionV3 and Xception"),
        P(
            "Both pretrained networks were loaded with ImageNet weights (Szegedy et al., 2016; Chollet, 2017). In the baseline the backbone was "
            "frozen and only a new head was trained, so the networks acted as fixed feature extractors. In the improved configuration the new head "
            "is trained first and the upper blocks of the backbone are then fine-tuned with a learning rate ten times smaller ({lst:ft}). Batch "
            "normalisation layers stay frozen during fine-tuning, which is the usual way to avoid destroying the pretrained statistics with small batches."
        ),
        CODE(
            "ft",
            "Pretrained network with a new head and two-stage fine-tuning",
            L_FT,
        ),
        H(3, "3.7.3 Classical classifiers"),
        P(
            "Random Forest, Decision Tree and SVM classifiers (Breiman, 2001; Quinlan, 1986; Cortes and Vapnik, 1995) were trained on deep features. "
            "In the baseline the features came from separate 128-pixel extractor networks, and the classifiers, together with Logistic Regression, "
            "were also trained on 65,536 raw greyscale pixels. In the improved configuration the features are the 256 values of the dense layer of "
            "each fine-tuned network ({lst:classical}). Within each feature source the three classifiers are combined by soft voting."
        ),
        CODE("classical", "Classical classifiers on deep features", L_CLASSICAL),
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
        CODE("vote", "Majority and confidence-weighted voting", L_VOTE),
        H(2, "3.8 Hardware, software and reproducibility"),
        P(
            "The baseline was trained in Google Colab on an NVIDIA T4 GPU, and the improved configuration on a 4-core CPU with 15 GB of memory, both "
            "with TensorFlow 2.21.0, Keras 3.13.2 and scikit-learn 1.6.1 (Abadi et al., 2016; Pedregosa et al., 2011). Seeds were fixed to 42. Every "
            "result was written as data: one row per model in a results table, a classification report and confusion matrix per model, per-epoch "
            "training histories and per-image predictions. The code is organised as a small package with one module per step, and the "
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
        H(3, "3.9.1 How to read the performance measures"),
        P(
            "The results tables in Chapters 4 and 6 use the same columns: accuracy, macro precision, macro recall, macro F1 and a Wilson 95% "
            "confidence interval. {tab:measures} explains each one and gives an example from the custom CNN of the baseline configuration "
            "(first row of {tab:deep}). Accuracy is given in per cent, and precision, recall and F1 as proportions between 0 and 1."
        ),
        TAB(
            "measures",
            "What each column of the results tables shows, with an example from the baseline custom CNN",
            ["Measure", "What it shows", "How it is calculated", "Example (baseline CNN)"],
            [
                [
                    "Accuracy (%)",
                    "The share of all test images that were classified correctly.",
                    "Correct predictions ÷ all test images × 100.",
                    "1,447 of 1,600 images correct = <<acc.CNN_standalone>>%.",
                ],
                [
                    "Precision (one class)",
                    "When the model says “glioma”, how often it is right. Low precision means many false alarms.",
                    "True positives ÷ (true positives + false positives).",
                    "Glioma precision <<precision.CNN_standalone.glioma>>%: almost every glioma prediction was a real glioma.",
                ],
                [
                    "Recall (one class)",
                    "Of all images that really are glioma, how many the model found. Low recall means missed tumours.",
                    "True positives ÷ (true positives + false negatives).",
                    "Glioma recall <<recall.CNN_standalone.glioma>>%: about one glioma in four was missed.",
                ],
                [
                    "F1 (one class)",
                    "One number that is high only when precision and recall are both high.",
                    "2 × precision × recall ÷ (precision + recall).",
                    "Glioma F1 <<f1c.CNN_standalone.glioma>>, pulled down by the low recall.",
                ],
                [
                    "Macro precision, macro recall, macro F1",
                    "The average over the four classes, each class counting equally, so a weak class pulls the value down.",
                    "Mean of the four per-class values.",
                    "<<prec.CNN_standalone>>, <<rec.CNN_standalone>> and <<f1.CNN_standalone>>.",
                ],
                [
                    "Wilson 95% CI (%)",
                    "The range in which the model’s true accuracy on similar images probably lies. A narrow range means a precise "
                    "estimate; heavily overlapping ranges mean two models may not really differ.",
                    "Wilson score interval for a proportion with n = 1,600 and z = 1.96 (Wilson, 1927).",
                    "<<st.wilson_ci.CNN_standalone>>: the true accuracy is probably between these two values.",
                ],
            ],
            widths=[3.0, 5.0, 4.2, 4.2],
            font=8.5,
        ),
        P(
            "Three points help when reading these numbers. First, every class has 400 test images, so macro recall is the same as accuracy "
            "(for the CNN, <<rec.CNN_standalone>> and <<acc.CNN_standalone>>%), and the macro values mainly add information through precision "
            "and F1. Second, the per-class values matter more clinically than the averages: the CNN’s glioma precision is high but its glioma "
            "recall is low, which means that its typical mistake is to miss a glioma, not to raise a false alarm. Third, an interval is "
            "needed because 1,600 images are a sample. The Wilson interval is used rather than the simpler normal approximation because it "
            "stays reliable when accuracy is close to 100% (Brown, Cai and DasGupta, 2001)."
        ),
        H(3, "3.9.2 How to read the McNemar comparison"),
        P(
            "Two models can have different accuracy by chance. Because all models were tested on the same 1,600 images, the fairest test "
            "looks only at the images on which the two models disagree. {tab:mcnemar_cols} explains the columns of the McNemar tables "
            "({tab:mcnemar}; {tab:imp_mc} uses the same counts, labelled “first right only” and “second right only”)."
        ),
        TAB(
            "mcnemar_cols",
            "Columns of the McNemar tables, with the comparison of the majority vote (A) and the CNN (B) as an example",
            ["Column", "Meaning", "Example"],
            [
                [
                    "Model A, Model B",
                    "The two models being compared on the same test images.",
                    "A = six-model majority vote, B = custom CNN.",
                ],
                [
                    "n01 (A right, B wrong)",
                    "Number of images that model A classifies correctly and model B gets wrong.",
                    "<<mc.FINAL_majority.CNN.n01>> images.",
                ],
                [
                    "n10 (A wrong, B right)",
                    "Number of images that model B classifies correctly and model A gets wrong.",
                    "<<mc.FINAL_majority.CNN.n10>> images.",
                ],
                [
                    "p (exact)",
                    "How likely a split at least this uneven would be if the two models were equally good. Images that both models get right "
                    "(or both get wrong) are ignored because they say nothing about which model is better.",
                    "Exact binomial test on the <<mc.FINAL_majority.CNN.n01>> + <<mc.FINAL_majority.CNN.n10>> disagreements: "
                    "p = <<mc.FINAL_majority.CNN.p>>.",
                ],
                [
                    "p (Holm)",
                    "The same p-value corrected for running ten comparisons, so that the chance of at least one false “significant” "
                    "result stays at 5%. A value below 0.05 counts as significant.",
                    "p = <<mc.FINAL_majority.CNN.ph>>: still significant.",
                ],
            ],
            widths=[3.4, 8.0, 5.0],
            font=8.5,
        ),
        P(
            "In the example, the two models disagree on <<mc.FINAL_majority.CNN.n01>> + <<mc.FINAL_majority.CNN.n10>> images. If they were "
            "equally good, these disagreements would split roughly evenly. The observed split of <<mc.FINAL_majority.CNN.n01>> to "
            "<<mc.FINAL_majority.CNN.n10>> is very unlikely under that assumption, so the majority vote is genuinely better than the CNN. The "
            "difference between n01 and n10, divided by the number of test images, is also the difference in accuracy between the two models "
            "(here 2.00 percentage points in the ensemble pipeline, {tab:boot}). By contrast, Xception and InceptionV3 split "
            "<<mc.Xception.InceptionV3.n01>> to <<mc.Xception.InceptionV3.n10>>, which could easily happen by chance (Holm p = "
            "<<mc.Xception.InceptionV3.ph>>), so neither can be said to be better. The Holm method (Holm, 1979) sorts the ten p-values from "
            "smallest to largest and multiplies them by 10, 9, 8 and so on, which is less strict than multiplying them all by 10 (Bonferroni) "
            "but still controls false positives."
        ),
        H(2, "3.10 Explainability methodology"),
        P(
            "{fig:xaipipe} summarises the explainability pipeline. Explanations were computed for the class predicted by the six-model majority vote, "
            "so they show what supported the ensemble decision. Six explanation methods were used, and the "
            "explanations were then tested with four checks; each has its own subsection below. Each method is described below by the question it "
            "answers, how it is computed, the settings used and how its map should be read. Examples of every method on the same test images "
            "are shown in Chapter 5 ({fig:xai_xcp} and {fig:xai_cnn})."
        ),
        FIG(
            "xaipipe",
            "paper_assets/figures/fig_xai_pipeline.png",
            "Explainability pipeline: methods (left) and how they are evaluated (right)",
            width=6.2,
        ),
        P(
            "Example images were chosen for each class: the correctly classified test image with the highest ensemble confidence, and a confidently "
            "misclassified image from the class’s most frequent error. No-tumour images were never misclassified by the baseline ensemble, so its "
            "error example is the most frequent miss into that class, a glioma predicted as no tumour. Quantitative evaluation used a stratified "
            "sample of 160 test images (30 per class plus 40 misclassified images, seed 42)."
        ),
        H(3, "3.10.1 Grad-CAM"),
        P(
            "**Question answered:** which regions of the image raised the score of the predicted class? Grad-CAM (Selvaraju et al., 2017) "
            "uses the last convolutional layer, where each feature map still has a spatial layout. It measures how much the class score "
            "changes when each feature map changes (the gradient), averages this gradient over the map to obtain one weight per feature map, "
            "adds the feature maps together using these weights, and keeps only the positive part. The result is a coarse grid (for example "
            "7 × 7 cells for Xception at 224 pixels) that is enlarged to the image size and scaled to the range 0 to 1."
        ),
        P(
            "**Settings:** the pre-softmax class score was used, at the last spatial layer of each network (conv2d_7 or last_conv for the "
            "CNN, mixed10 for InceptionV3 and block14_sepconv2_act for Xception); the code is in {lst:gradcam}. **How to read it:** red areas "
            "supported the predicted class most, blue areas hardly at all. Because the grid is coarse, the map shows the approximate region, "
            "not the exact tumour border."
        ),
        CODE("gradcam", "Grad-CAM and HiResCAM as implemented for the analysis", L_GRADCAM),
        H(3, "3.10.2 Grad-CAM++"),
        P(
            "**Question answered:** the same as Grad-CAM, but with better coverage when several regions, or a small region, support the "
            "class. Grad-CAM++ (Chattopadhay et al., 2018) replaces the simple average of the gradients with a weighted average that uses "
            "higher-order gradient terms, so that every location that raises the score contributes, not only the strongest one "
            "({lst:gpp} in Appendix B). **Settings:** same layers and class as Grad-CAM. **How to read it:** as Grad-CAM. If the two maps "
            "look very different, the explanation depends on how the gradients are combined and should be trusted less."
        ),
        H(3, "3.10.3 HiResCAM"),
        P(
            "**Question answered:** which locations actually produced the class score? HiResCAM (Draelos and Carin, 2020) multiplies each "
            "gradient by its feature map value at every location before adding them up, instead of averaging the gradients first. For "
            "networks that end in global average pooling (InceptionV3, Xception and the improved CNN) this gives exactly the same map as "
            "Grad-CAM. For the baseline CNN, which ends in Flatten and Dense layers, the two differ, and HiResCAM is the one that follows "
            "the model’s computation. **How to read it:** as Grad-CAM; differences between the two maps for the baseline CNN show where "
            "averaging the gradients hides spatial detail."
        ),
        H(3, "3.10.4 Occlusion"),
        P(
            "**Question answered:** if part of the image is hidden, does the prediction change? Occlusion (Zeiler and Fergus, 2014) uses no "
            "gradients. A 32 × 32 black patch is moved over the image in steps of 16 pixels (169 positions at 224 pixels, 225 at 256), and at "
            "each position the drop in the probability of the predicted class is recorded. **How to read it:** bright areas are regions the "
            "model needs; hiding them lowers its confidence. An almost empty map means that no single patch matters, which happens when the "
            "model relies on large or repeated cues. The method is slow because it needs one prediction per patch position."
        ),
        H(3, "3.10.5 LIME"),
        P(
            "**Question answered:** which image segments, switched on or off, best explain the prediction? LIME (Ribeiro, Singh and Guestrin, "
            "2016) splits the image into super-pixels (small regions of similar intensity, found with the quick-shift algorithm), creates "
            "many copies of the image with random super-pixels hidden, records the model’s probability for each copy, and fits a simple "
            "linear model that predicts this probability from which super-pixels were visible. The weights of this linear model form the "
            "map. **Settings:** 600 perturbed copies for the CNN and 400 for Xception (fewer because Xception is slower), hidden "
            "super-pixels set to black ({lst:lime} in Appendix B). **How to read it:** highlighted super-pixels support the class. Because "
            "super-pixels follow image edges, LIME maps can include the skull or background when those regions help the model."
        ),
        H(3, "3.10.6 SHAP"),
        P(
            "**Question answered:** how much did each input contribute, positively or negatively, compared with a reference? SHAP "
            "(Lundberg and Lee, 2017) is based on Shapley values from game theory: each input receives a share of the difference between the "
            "model’s output for this image and its average output, and the shares add up exactly to that difference. Two versions were "
            "used ({lst:shap} in Appendix B). For the CNN, the gradient explainer (expected gradients) with 50 background images gives a "
            "value for each pixel. For the Random Forest that classifies CNN features, the exact tree explainer gives a value for each of the "
            "256 features. **How to read it:** positive values push the prediction towards the class and negative values push it away; for "
            "the Random Forest, a few features with large values mean that the decision relies on a small part of the CNN’s representation "
            "(Section 5.10)."
        ),
        H(3, "3.10.7 Faithfulness check: deletion and insertion"),
        P(
            "A map can look convincing and still not reflect what the model uses. Deletion and insertion curves (Petsiuk, Das and Saenko, "
            "2018) test this ({lst:di} in Appendix B). For deletion, pixels are removed (set to black) in order of importance, in 20 steps of "
            "5% of the image, and the probability of the predicted class is recorded after each step. If the map is faithful, the probability "
            "falls quickly and the area under the curve (deletion AUC) is small. For insertion, the process is reversed: starting from a "
            "blurred copy, the most important pixels are revealed first, and a faithful map gives a quick rise and a large insertion AUC. "
            "Each method is compared with a random map; a method is useful only if it does clearly better than random."
        ),
        H(3, "3.10.8 Sanity check: model-parameter randomisation"),
        P(
            "Some saliency methods produce similar maps even for a network that has learnt nothing, because they mostly follow image edges. "
            "The cascading randomisation test (Adebayo et al., 2018) checks this ({lst:sanity} in Appendix B). The weights of the network "
            "are replaced with random values one layer at a time, from the output layer downwards, and after each step the new map is "
            "compared with the original using the Spearman rank correlation and the structural similarity index (SSIM) on 40 images. A "
            "method passes if the similarity falls towards zero as more layers are randomised, which shows that the map depends on what the "
            "model learnt."
        ),
        H(3, "3.10.9 Cross-model agreement: the EAA-IoU score"),
        P(
            "The Ensemble Attention Agreement score (EAA-IoU) asks a different question: do the three deep networks look at the same part of "
            "the image? It needs no tumour outlines drawn by a radiologist, so it can be computed for every image, including new ones in "
            "the web demonstrator. It is calculated in five steps ({lst:eaa} in Appendix B):"
        ),
        NL(
            [
                "Compute the Grad-CAM map of the CNN, Xception and InceptionV3 for the class predicted by the ensemble, each scaled to 0 to 1.",
                "Turn each map into a mask by keeping the pixels with a value of 0.5 or more (the region the model considers important).",
                "For each of the three pairs of models, count the pixels inside both masks (the intersection) and inside at least one mask "
                "(the union).",
                "Divide intersection by union to obtain the pair’s IoU, a number between 0 (no overlap) and 1 (identical regions).",
                "Average the three pair IoUs. This average is the EAA-IoU of the image.",
            ]
        ),
        P(
            "{fig:eaa_ex} and {tab:eaa_ex} work through three real test images with the improved models. In image A (a glioma, correctly "
            "classified) all three masks cover the upper brain and the lesion, the pair IoUs are <<wex.0.p0.iou>>, <<wex.0.p1.iou>> and "
            "<<wex.0.p2.iou>>, and the EAA-IoU is <<wex.0.eaa>>, which counts as strong agreement. In image B (a pituitary tumour, also "
            "correct) the two pretrained networks agree (IoU <<wex.1.p2.iou>>), but the CNN highlights the skull edges, so its pairs score "
            "almost zero and the EAA-IoU drops to <<wex.1.eaa>>. In image C (a glioma that the ensemble called no tumour with only "
            "<<wex.2.conf>>% confidence) every pair has an IoU below 0.05, the EAA-IoU is <<wex.2.eaa>>, and the image would be flagged for "
            "review. Image B also shows a weakness of the score: one model looking elsewhere is enough to lower the agreement, even when the "
            "prediction is correct."
        ),
        TAB(
            "eaa_ex",
            "The numbers behind the EAA-IoU worked example",
            ["Quantity", "A. High agreement", "B. Moderate agreement", "C. Low agreement"],
            [
                ["True class", "<<wex.0.true>>", "<<wex.1.true>>", "<<wex.2.true>>"],
                [
                    "Ensemble prediction",
                    "<<wex.0.pred>> (correct)",
                    "<<wex.1.pred>> (correct)",
                    "<<wex.2.pred>> (wrong)",
                ],
                ["Ensemble confidence (%)", "<<wex.0.conf>>", "<<wex.1.conf>>", "<<wex.2.conf>>"],
                [
                    "Mask size, CNN / Xception / InceptionV3 (% of image)",
                    "<<wex.0.area.CNN>> / <<wex.0.area.Xception>> / <<wex.0.area.InceptionV3>>",
                    "<<wex.1.area.CNN>> / <<wex.1.area.Xception>> / <<wex.1.area.InceptionV3>>",
                    "<<wex.2.area.CNN>> / <<wex.2.area.Xception>> / <<wex.2.area.InceptionV3>>",
                ],
            ]
            + [
                [f"{pair}: intersection / union (pixels) = IoU"]
                + [
                    f"<<wex.{k}.p{i}.inter>> / <<wex.{k}.p{i}.union>> = <<wex.{k}.p{i}.iou>>"
                    for k in range(3)
                ]
                for i, pair in enumerate(["CNN vs Xception", "CNN vs InceptionV3", "Xception vs InceptionV3"])
            ]
            + [
                ["EAA-IoU (mean of the three IoUs)", "<<wex.0.eaa>>", "<<wex.1.eaa>>", "<<wex.2.eaa>>"],
                [
                    "Agreement band and action",
                    "Strong (0.20 or above): no alert",
                    "Moderate (0.10 to 0.20): no alert",
                    "Below 0.10: alert, radiologist review",
                ],
            ],
            widths=[4.6, 3.9, 3.9, 3.9],
            font=8.5,
        ),
        FIG(
            "eaa_ex",
            "paper_assets/figures_improved/fig_eaa_worked_example.png",
            "EAA-IoU step by step for three test images (improved models). For each image, top row: input and Grad-CAM of each network, "
            "with the 0.5 mask outlined in white; bottom row: overlap of each pair of masks and the resulting IoU",
            width=4.8,
        ),
        P(
            "The score was computed for all 1,600 test images in both configurations. The planned bands (alert below 0.40, strong agreement "
            "from 0.65) were replaced by bands taken from the observed distribution (alert below 0.10, strong from 0.20); Section 5.8 explains "
            "why and reports how errors are distributed across the bands."
        ),
        H(3, "3.10.10 Shortcut check"),
        P(
            "The last check asks whether the models look at the brain at all. The head was found by thresholding the image, and a 12-pixel "
            "outer rim was defined to cover the skull and the image edge. For each Grad-CAM map, the share of its total value (its mass) "
            "that falls outside the head or in the rim was compared with the share of the image area these regions occupy. A model that "
            "ignored position would put about the same share of its attention there as the area they cover; a model that focuses on the "
            "brain would put much less. Three further experiments tested the cause of any excess attention (Section 5.7)."
        ),
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
