"""Front matter: title page, declaration, dedication, acknowledgements, abstract, lists."""
from engine import *

TITLE = ('A Comparative Analysis of Ensemble Learning, Fine-Tuned Models and CNNs for Multi-Class Disease Prediction '
         'in Medical Imaging: An Integrated Framework with Systematic Optimization and Explainable AI')
STUDENT = 'Pattiyage Shehan Kavishka'; SID = 'E187041'

def title_page(doc, r):
    for _ in range(3): doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; add_runs(p, 'London Metropolitan University', size=16)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; add_runs(p, 'CS6P05ES Computing Project – Final Project Report', size=13)
    doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.line_spacing = 1.2
    add_runs(p, '**' + TITLE + '**', size=17)
    doc.add_paragraph()
    for line in [f'Name: {STUDENT}', f'London Met ID / Student ID: {SID}', 'Programme: BSc (Hons) Data Science (top-up), delivered at ESOFT Metro Campus',
                 'First supervisor: Ms. Hiruni Wimalaweera', 'Module leader: Mr. Thepul Ginige', 'Submission: November 2026']:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; add_runs(p, line, size=12)
    doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, '<i>This report extends the author’s Pearson HND Unit 16 research report (October 2024) with a standardised re-run '
                'of all models on held-out data and a new explainable-AI analysis.</i>', size=11)

def declaration(doc, r):
    p = doc.add_heading(level=1); add_runs(p, 'Declaration'); p.paragraph_format.page_break_before = True
    for t in [f'Module: CS6P05ES Computing Project        Student ID: {SID}', 'Module leader: Mr. Thepul Ginige',
              'I confirm that I have read and understood the university regulations on cheating, plagiarism and collusion, '
              'including falsifying experimental data and presenting the work of others as my own. This report is my own work. '
              'Every source I have used is acknowledged with an in-text citation and an entry in the reference list. '
              'All experimental numbers in this report were produced by the code and result files listed in Appendix H.',
              'Signature: Pattiyage Shehan Kavishka                Date: ____ / ____ / 2026']:
        add_runs(doc.add_paragraph(), t)

def list_title(doc, text):
    p = doc.add_paragraph(); p.paragraph_format.page_break_before = True; p.paragraph_format.space_after = Pt(12)
    add_runs(p, '**' + text + '**', size=16)

def blocks():
    return [
        RAW(title_page), RAW(declaration),
        H(1, 'Dedication'),
        P('To my family, whose patience and encouragement made both stages of this project possible.'),
        H(1, 'Acknowledgements'),
        P('I thank my supervisor, Ms. Hiruni Wimalaweera, for her guidance on the experimental design and on presenting the work honestly, and '
          'Mr. Dileepa Lakshan, who supervised the first (HND) stage of this research and introduced me to ensemble learning. '
          'I am also grateful to the module team at ESOFT Metro Campus and London Metropolitan University, and to the researchers who '
          'made the public MRI data and open-source libraries used in this work available.'),
        H(1, 'Abstract', new=True),
        P('Brain tumour type affects treatment, yet reading MRI scans is slow and varies between readers. This project builds and tests an '
          'integrated framework that compares three families of models for four-class brain MRI classification (glioma, meningioma, '
          'pituitary tumour and no tumour): a custom convolutional neural network (CNN), two ImageNet-pretrained networks (InceptionV3 and '
          'Xception) used with frozen backbones, and classical classifiers (Random Forest, Decision Tree, Support Vector Machine) trained on '
          'raw pixels or on deep features. Six predictors are combined by majority and confidence-weighted voting. The work extends an '
          'earlier HND study whose headline 98.20% accuracy was found, on audit, to have been measured on training images. All models were '
          'therefore retrained under one standardised protocol and evaluated on 1,600 held-out test images. The six-model majority vote '
          'reached 92.50% accuracy (95% bootstrap interval 91.25–93.69%) and was significantly better than every single model. The custom '
          'CNN (90.44%) outperformed both pretrained networks, so the hypothesis that pretrained models would extract better features was '
          'not supported. Glioma remained the hardest class (77.50% recall), mostly confused with meningioma. A new explainability layer '
          'applied Grad-CAM, Grad-CAM++, HiResCAM, LIME, SHAP and occlusion to the saved models and evaluated the explanations with '
          'deletion and insertion tests, a weight-randomisation sanity check, a cross-model agreement score (EAA-IoU) and a check for '
          'attention outside the brain. Explanations were more faithful than a random baseline and passed the sanity check, but the three '
          'networks often attended to different regions, and agreement between them flagged errors less well than the ensemble’s own '
          'confidence. The framework is a transparent research prototype for decision support, not a diagnostic device; clinical use would '
          'need external validation and radiologist review.', new=True),
        P('**Keywords:** brain tumour MRI; convolutional neural networks; transfer learning; ensemble learning; explainable AI; Grad-CAM; SHAP; LIME', new=True),
        RAW(lambda d, r: list_title(d, 'Table of Contents')), RAW(lambda d, r: toc(d, 'toc')),
        RAW(lambda d, r: list_title(d, 'List of Figures')), RAW(lambda d, r: toc(d, 'fig')),
        RAW(lambda d, r: list_title(d, 'List of Tables')), RAW(lambda d, r: toc(d, 'tab')),
        RAW(lambda d, r: list_title(d, 'List of Listings')), RAW(lambda d, r: toc(d, 'lst')),
        H(1, 'List of Abbreviations'),
        TAB('abbr', 'Abbreviations used in this report', ['Abbreviation', 'Meaning'], [
            ['AUC / AUROC', 'Area under the (receiver operating characteristic) curve'], ['CAM', 'Class activation map'], ['CI', 'Confidence interval'],
            ['CNN', 'Convolutional neural network'], ['DT', 'Decision Tree'], ['EAA-IoU', 'Ensemble Attention Agreement – intersection over union'],
            ['Grad-CAM', 'Gradient-weighted class activation mapping'], ['HiResCAM', 'High-resolution class activation mapping'],
            ['IoU', 'Intersection over union'], ['LIME', 'Local Interpretable Model-agnostic Explanations'], ['LR', 'Logistic Regression'],
            ['MRI', 'Magnetic resonance imaging'], ['PR', 'Precision–recall'], ['RF', 'Random Forest'], ['RO / RQ / H', 'Research objective / question / hypothesis'],
            ['ROC', 'Receiver operating characteristic'], ['SHAP', 'SHapley Additive exPlanations'], ['SVM', 'Support Vector Machine'],
            ['XAI', 'Explainable artificial intelligence']], widths=[4, 11]),
    ]
