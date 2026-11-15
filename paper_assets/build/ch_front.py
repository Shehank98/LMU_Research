"""Front matter: title page, declaration, dedication, acknowledgements, abstract, lists."""

from engine import *

TITLE = (
    "A Comparative Analysis of Ensemble Learning, Fine-Tuned Models and CNNs for Multi-Class Disease Prediction "
    "in Medical Imaging: An Integrated Framework with Systematic Optimization and Explainable AI"
)
STUDENT = "Pattiyage Shehan Kavishka"
SID = "E187041"


def title_page(doc, r):
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, "London Metropolitan University", size=16)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, "CS6P05ES Computing Project: Final Project Report", size=13)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.2
    add_runs(p, "**" + TITLE + "**", size=17)
    doc.add_paragraph()
    for line in [
        f"Name: {STUDENT}",
        f"London Met ID / Student ID: {SID}",
        "Programme: BSc (Hons) Data Science (top-up), delivered at ESOFT Metro Campus",
        "First supervisor: Ms. Hiruni Wimalaweera",
        "Module leader: Mr. Thepul Ginige",
        "Submission: November 2026",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(p, line, size=12)


def declaration(doc, r):
    p = doc.add_heading(level=1)
    add_runs(p, "Declaration")
    p.paragraph_format.page_break_before = True
    for t in [
        f"Module: CS6P05ES Computing Project        Student ID: {SID}",
        "Module leader: Mr. Thepul Ginige",
        "I confirm that I have read and understood the university regulations on cheating, plagiarism and collusion. "
        "Every source used in this report is acknowledged with an in-text citation and an entry in the reference list, and every "
        "experimental number was produced by the code and result files listed in Appendix F.",
        "Signature: Pattiyage Shehan Kavishka                Date: ____ / ____ / 2026",
    ]:
        add_runs(doc.add_paragraph(), t)


def list_title(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.page_break_before = True
    p.paragraph_format.space_after = Pt(12)
    add_runs(p, "**" + text + "**", size=16)


def blocks():
    return [
        RAW(title_page),
        RAW(declaration),
        H(1, "Dedication"),
        P("To my family, for their patience and encouragement throughout this project."),
        H(1, "Acknowledgements"),
        P(
            "I thank my supervisor, Ms. Hiruni Wimalaweera, for her guidance on the experimental design and for encouraging me to test my "
            "results critically. I am also grateful to the module team at ESOFT Metro Campus and London Metropolitan University, and to the "
            "researchers who made the public MRI data and the open-source libraries used in this work available."
        ),
        H(1, "Abstract"),
        P(
            "Brain tumour type affects treatment, yet reading MRI scans is slow and varies between readers. This project builds and "
            "tests an integrated framework for four-class brain MRI classification (glioma, meningioma, pituitary tumour and no "
            "tumour) that compares a custom convolutional neural network (CNN), two ImageNet-pretrained networks (InceptionV3 and "
            "Xception) and classical classifiers (Random Forest, Decision Tree and Support Vector Machine) trained on deep features, "
            "and combines six predictors by majority and confidence-weighted voting. A baseline configuration reached "
            "<<clean.baseline.final_majority.full.acc>>% accuracy on 1,600 held-out test images. Explainable AI methods (Grad-CAM, "
            "Grad-CAM++, HiResCAM, LIME, SHAP and occlusion), tested for faithfulness and with a weight-randomisation sanity check, "
            "showed that the models paid much attention to the image background. Further experiments explained why: the classes come "
            "from different source collections, simple image statistics predict the class with <<sinv.metadata_rf_cv_accuracy_pct>>% "
            "accuracy, and 21% of the test images duplicate training images. An improved configuration therefore cropped every image "
            "to the head, added augmentation and fine-tuned the pretrained networks. Its majority vote reached "
            "<<clean.improved.final_majority.full.acc>>% on the full test set and <<clean.improved.final_majority.clean.acc>>% on the "
            "duplicate-free subset, a statistically significant improvement over the baseline (<<clean.baseline.final_majority.clean.acc>>% "
            "on the same subset). Fine-tuned InceptionV3 and Xception outperformed the custom CNN, whose accuracy fell once the "
            "background cue was removed. <<ABS_XAI>> Glioma remained the hardest class "
            "(85% recall). The work shows that explanation methods are most useful when their findings are tested and fed back "
            "into the design of the model. The system is a research prototype for decision support, not a diagnostic device."
        ),
        P(
            "**Keywords:** brain tumour MRI; convolutional neural networks; transfer learning; ensemble learning; explainable AI; Grad-CAM; SHAP; LIME",
            new=True,
        ),
        RAW(lambda d, r: list_title(d, "Table of Contents")),
        RAW(lambda d, r: toc(d, "toc")),
        RAW(lambda d, r: list_title(d, "List of Figures")),
        RAW(lambda d, r: toc(d, "fig")),
        RAW(lambda d, r: list_title(d, "List of Tables")),
        RAW(lambda d, r: toc(d, "tab")),
        RAW(lambda d, r: list_title(d, "List of Listings")),
        RAW(lambda d, r: toc(d, "lst")),
        H(1, "List of Abbreviations"),
        TAB(
            "abbr",
            "Abbreviations used in this report",
            ["Abbreviation", "Meaning"],
            [
                ["AUC / AUROC", "Area under the (receiver operating characteristic) curve"],
                ["CAM", "Class activation map"],
                ["CI", "Confidence interval"],
                ["CNN", "Convolutional neural network"],
                ["DT", "Decision Tree"],
                ["EAA-IoU", "Ensemble Attention Agreement – intersection over union"],
                ["Grad-CAM", "Gradient-weighted class activation mapping"],
                ["HiResCAM", "High-resolution class activation mapping"],
                ["IoU", "Intersection over union"],
                ["LIME", "Local Interpretable Model-agnostic Explanations"],
                ["LR", "Logistic Regression"],
                ["MRI", "Magnetic resonance imaging"],
                ["PR", "Precision–recall"],
                ["RF", "Random Forest"],
                ["RO / RQ / H", "Research objective / question / hypothesis"],
                ["ROC", "Receiver operating characteristic"],
                ["SHAP", "SHapley Additive exPlanations"],
                ["SVM", "Support Vector Machine"],
                ["XAI", "Explainable artificial intelligence"],
            ],
            widths=[4, 11],
        ),
    ]
