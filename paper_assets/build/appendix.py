import os
from engine import *

L_GPP = '''def gradcam_pp(cm, X, cls):
    A, G = cm.maps_and_grads(X, cls)
    g2, g3 = G ** 2, G ** 3                     # closed form of Chattopadhay et al. (2018)
    denom = 2 * g2 + A.sum(axis=(1, 2), keepdims=True) * g3
    alpha = np.divide(g2, denom, out=np.zeros_like(g2), where=denom != 0)
    w = (alpha * np.maximum(G, 0)).sum(axis=(1, 2))   # positive gradients only
    return _finish(np.einsum('bhwc,bc->bhw', A, w))'''
L_LIME = '''def lime_map(model, x, cls, num_samples=600, seed=42):
    exp = lime_image.LimeImageExplainer(random_state=seed)
    seg = lambda im: quickshift(im, kernel_size=4, max_dist=200, ratio=0.2, rng=seed)
    e = exp.explain_instance(x.astype('double'),
                             lambda b: model.predict(b, batch_size=64, verbose=0),
                             labels=(cls,), top_labels=None, hide_color=0,
                             num_samples=num_samples, segmentation_fn=seg, random_seed=seed)
    w = dict(e.local_exp[cls])                       # weight of each super-pixel
    m = np.vectorize(lambda s: w.get(s, 0.0))(e.segments)
    return np.maximum(m, 0) / max(m.max(), 1e-12)    # keep evidence FOR the class'''
L_SHAP = '''# (a) Deep model: expected gradients with 50 background images
ex = shap.GradientExplainer(cnn, X[bg_idx])
sv = ex.shap_values(X_examples, nsamples=200, rseed=42)        # one map per class
# (b) Feature-level ensemble: exact TreeSHAP on the RF that reads CNN dense features
F = cnn_dense_features(X128)                                   # (1600, 256)
rf = joblib.load('cnn_feat_rf.pkl')
tree_sv = shap.TreeExplainer(rf).shap_values(F[idx])           # (n, 256, 4)
importance = np.abs(tree_sv).mean(0)                           # mean |SHAP| per unit and class'''
L_DI = '''def deletion_insertion(model, X, maps, cls, steps=20):
    order = np.argsort(-maps.reshape(len(X), -1), axis=1)        # most salient first
    blur = np.stack([cv2.GaussianBlur(x, (51, 51), 25) for x in X])
    for t in range(steps + 1):
        k = round(256 * 256 * t / steps)
        deleted  = X.copy();  deleted.reshape(len(X), -1, 3)[rows, order[:, :k]] = 0
        inserted = blur.copy(); inserted.reshape(len(X), -1, 3)[rows, order[:, :k]] = \\
                   X.reshape(len(X), -1, 3)[rows, order[:, :k]]
        p_del[t] = model.predict(deleted)[range(len(X)), cls]
        p_ins[t] = model.predict(inserted)[range(len(X)), cls]
    return np.trapezoid(p_del, dx=1 / steps, axis=0), np.trapezoid(p_ins, dx=1 / steps, axis=0)'''
L_SANITY = '''def cascade(name, groups):                      # Adebayo et al. (2018)
    m = load(name); base = gradcam(CamModel(m, TARGET[name]), Xr, cr)
    for g in groups:                            # from the output layer downwards
        for ln in g:
            l = m.get_layer(ln)
            l.set_weights([rng.normal(0, np.std(w) + 1e-3, w.shape) for w in l.get_weights()])
        r = gradcam(CamModel(m, TARGET[name]), Xr, cr)
        rho = np.mean([spearmanr(base[k].ravel(), r[k].ravel()).statistic for k in range(len(Xr))])
        s = np.mean([ssim(base[k], r[k], data_range=1.0) for k in range(len(Xr))])
        print(g[-1], rho, s)                     # similarity should collapse towards 0'''
L_EAA = '''def iou(a, b, t=0.5):                           # from the training notebook
    A, B = a >= t, b >= t
    union = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum() / union) if union > 0 else 0.0

maps = {nm: gradcam_batch(m, X_batch, ensemble_pred) for nm, m in models.items()}
p = [iou(maps['CNN'][k], maps['Xception'][k]), iou(maps['CNN'][k], maps['InceptionV3'][k]),
     iou(maps['Xception'][k], maps['InceptionV3'][k])]
eaa_iou = np.mean(p)                            # Ensemble Attention Agreement'''
L_STATS = '''# exact McNemar test on paired predictions
n01 = int((correct_a & ~correct_b).sum()); n10 = int((~correct_a & correct_b).sum())
p = binomtest(n01, n01 + n10, 0.5).pvalue
# percentile bootstrap 95% CI for accuracy (2,000 resamples)
boot = rng.integers(0, n, (2000, n))
ci = np.percentile(correct[boot].mean(axis=1), [2.5, 97.5])'''

def appendices(V):
    full = []
    import pandas as pd, os
    from values import ROOT
    master = pd.read_csv(os.path.join(ROOT, 'metrics', 'results_master.csv'))
    for _, r in master.iterrows():
        full.append([r.model, f'{100 * r.accuracy:.2f}', f'{r.precision_macro:.3f}', f'{r.recall_macro:.3f}', f'{r.f1_macro:.3f}',
                     f'{r.f1_glioma:.3f}', f'{r.f1_meningioma:.3f}', f'{r.f1_notumor:.3f}', f'{r.f1_pituitary:.3f}', f'{r.train_time_sec:.1f}'])
    thr = pd.read_csv(os.path.join(ROOT, 'metrics', 'eaa_iou_threshold_table.csv')).fillna('–')
    thr_rows = [[str(r.threshold), str(r.flagged_pct), str(r.errors_caught_pct), str(r.error_rate_in_flagged_pct), str(r.error_rate_in_unflagged_pct)] for _, r in thr.iterrows()]
    for q in ['0.05', '0.1']:
        thr_rows.insert(0, [q, V[f'st.eaa_recomputed.threshold_{q}.flagged_pct'], V[f'st.eaa_recomputed.threshold_{q}.errors_caught_pct'],
                            V[f'st.eaa_recomputed.threshold_{q}.error_rate_flagged_pct'], V[f'st.eaa_recomputed.threshold_{q}.error_rate_unflagged_pct']])
    thr_rows = sorted(thr_rows, key=lambda r: float(r[0]))
    old_cm = [['Glioma', '1688', '34', '6', '5', '1,733'], ['Meningioma', '42', '1636', '8', '13', '1,699'], ['No tumour', '5', '5', '1837', '0', '1,847'], ['Pituitary', '4', '4', '0', '1698', '1,706']]
    crit = [['P3', 'Conduct primary and secondary research using appropriate methods, considering costs, access and ethics', 'Ch. 2 (secondary); 3.3–3.5, 3.13 (primary experiments, cost, access, ethics)'],
            ['P4', 'Apply appropriate analytical tools, analyse research findings and data', 'Ch. 4 and 5; 3.10–3.11'],
            ['M2', 'Discuss merits, limitations and pitfalls of approaches to data collection and analysis', '6.3, 6.5'],
            ['D1', 'Critically evaluate research methodologies and processes to justify chosen methods and analysis', '6.4, 3.1–3.2'],
            ['P5', 'Communicate research outcomes in an appropriate manner for the intended audience', 'Ch. 4–5 (technical); 6.6 (clinicians); web demonstrator'],
            ['M3', 'Analyse the extent to which outcomes meet set research objectives', '7.1 (Table of objectives), 4.11, 5.10'],
            ['D2', 'Evaluate outcomes and make valid, justified recommendations', '7.2–7.3'],
            ['P6', 'Discuss the effectiveness of research methods applied', '8.1'],
            ['P7', 'Discuss alternative research methodologies and lessons learnt', '8.2, 6.4'],
            ['M4', 'Analyse results in recommended actions for improvements and future research', '7.3, 7.5'],
            ['D3', 'Demonstrate reflection and engagement in the research process leading to recommended actions', '8.1–8.3, AUDIT.md']]
    changes = [
        ['Corrected', 'Abstract, 4.5, 4.10', 'Headline accuracy changed from 98.20% (training images) to the held-out 92.50%; old figure kept only as a labelled historical value.'],
        ['Corrected', '1.6, 4.11, 7.2', 'H1 now stated as rejected; H2 supported with McNemar tests.'],
        ['Corrected', '3.7.2', '"Fine-tuned" models described accurately as frozen-backbone transfer learning.'],
        ['Corrected', '4.2', 'Wrong statement that InceptionV3 beat Xception (84.63 < 86.91) removed; new comparison is tested statistically.'],
        ['Removed', '1.1, 2.2, Fig. 4', 'Unsupported mentions of KNN, U-Net and Naive Bayes.'],
        ['Removed', '3.2–3.4, Abstract', 'Claims of focus groups / interviews and "qualitative views" (no data were collected).'],
        ['Replaced', 'Ch. 4–5 (old)', '"Correlation analysis" and "Regression analysis" sections replaced by per-class metrics, confusion analysis, ROC/PR, McNemar tests, bootstrap and Wilson intervals, learning curves.'],
        ['Corrected', 'Annex B (old)', '"SPSS charts" title removed; all analysis was done in Python.'],
        ['Corrected', '3.5, Table 1 (old)', 'Dataset described accurately (7,200-image release, 5,600 / 1,600); "PITIUTRY" typo, 6,984 vs 6,985 inconsistency explained.'],
        ['Corrected', '3.6, 3.12 (old 1.5, 3.9)', 'Claims of noise reduction, registration, cross-validation and augmentation removed (not performed).'],
        ['Corrected', 'References', 'All references completed and verified (DOIs on Crossref); Cheng et al. accuracy corrected to 91.28%; Cortes and Vapnik (1995), Ismael and Abdel-Qader (2018), Tripathi and Bag (2019) and Yousaf et al. (2023) details fixed.'],
        ['Corrected', '8.3', '"Personnel Reflection" renamed "Personal reflection".'],
        ['Added', '1.4–1.7', 'Objectives RO5–RO6, research question RQ3, hypothesis H3, scope section and report structure.'],
        ['Added', '2.1, 2.5–2.8', 'Clinical context, explainable AI, evaluation of explanations, shortcut learning, research-gap table.'],
        ['Added', '2.9', 'New conceptual framework figure including the XAI layer.'],
        ['Added', '3.3–3.4, 3.8–3.11', 'Project management record, feasibility (time, cost, access, scope), hyperparameter table, hardware/software and reproducibility, statistical protocol, XAI methodology, code listings.'],
        ['Added', 'Ch. 4', 'Complete re-run results: training curves, all confusion matrices, accuracy with intervals, F1 heatmap, ROC/PR, McNemar and bootstrap tables, reproducibility checks, comparison with the HND results.'],
        ['Added', 'Ch. 5', 'New explainable-AI chapter: Grad-CAM, Grad-CAM++, HiResCAM, LIME, SHAP, occlusion; deletion/insertion; sanity check; EAA-IoU and consistency; shortcut check; SHAP on the Random Forest; clinical trust.'],
        ['Added', 'Ch. 6', 'Discussion with literature comparison, merits/limitations/pitfalls, critical evaluation of methodology, threats to validity and a plain-language stakeholder summary.'],
        ['Added', '7.1–7.3, Ch. 8', 'Objective-by-objective evaluation, answers to RQs, recommendations, reflection on methods and alternatives.'],
        ['Added', 'Appendices B–H', 'Code listings, extra tables, proposed questionnaire (reframed), criteria mapping, this change log, figures index, reproducibility artefacts.']]
    q = ['Section 1 – About you: role (radiologist, neurosurgeon, oncologist, trainee, other); years of experience (0–5, 6–10, 11–15, 16+); previous use of AI decision-support tools (yes / no).',
         'Section 2 – Attitudes: familiarity with AI-based diagnostic tools (very familiar … not familiar); confidence in AI assistance for brain tumour MRI (very confident … not confident).',
         'Section 3 – Case review (new): for 10 anonymised test images, record your own class, then the model’s suggestion with its confidence and heatmap; record whether you would change your answer.',
         'Section 4 – Explanations (new): for each heatmap, rate “the highlighted region is clinically plausible” (5-point scale) and “the heatmap made me more / less / equally confident”.',
         'Section 5 – Usefulness: expected effect on workflow (improve significantly … slow down); usefulness overall (very useful … not useful); open suggestions for improvement.']
    return [
        H(1, 'Appendices'),
        H(2, 'Appendix A Glossary of terms'),
        P('{tab:gloss} defines the technical terms used in the report in plain language.'),
        TAB('gloss', 'Glossary', ['Term', 'Meaning'], [
            ['Accuracy', 'Share of all test images classified correctly.'], ['Precision', 'Of the images given a class, the share that truly belong to it.'],
            ['Recall (sensitivity)', 'Of the images of a class, the share that the model finds.'], ['F1-score', 'Harmonic mean of precision and recall.'],
            ['Confusion matrix', 'Table of true classes (rows) against predicted classes (columns).'], ['ROC AUC', 'Probability that a random positive case receives a higher score than a random negative case.'],
            ['Transfer learning', 'Reusing a network trained on another task (here ImageNet) as a starting point.'], ['Frozen backbone', 'Pretrained layers whose weights are not updated during training.'],
            ['Fine-tuning', 'Continuing to train some or all pretrained layers on the new task (not done here).'], ['Ensemble', 'Combination of several models’ predictions.'],
            ['Majority vote', 'Each model casts one vote; the class with most votes wins.'], ['Confidence-weighted vote', 'Class probabilities of all models are summed and the largest total wins.'],
            ['Saliency map / heatmap', 'Image showing how much each region contributed to a prediction.'], ['Grad-CAM / Grad-CAM++ / HiResCAM', 'Gradient-based class activation maps computed from a convolutional layer.'],
            ['LIME', 'Explains one prediction by fitting a simple model to predictions on perturbed copies of the image.'], ['SHAP', 'Attributes a prediction to inputs using Shapley values from game theory.'],
            ['Occlusion', 'Hides one patch at a time and records the drop in the class probability.'], ['Deletion / insertion AUC', 'Faithfulness metrics: how fast the score falls (or rises) when salient pixels are removed (or revealed).'],
            ['Sanity check', 'Test that an explanation changes when the model’s weights are randomised.'], ['EAA-IoU', 'Mean pairwise overlap of the binarised heatmaps of the three deep models.'],
            ['Shortcut learning', 'Reliance on spurious cues that correlate with the label in the training data.'], ['McNemar test', 'Paired test of whether two classifiers err on different numbers of the same cases.'],
            ['Bootstrap interval', 'Uncertainty interval from re-sampling the test set with replacement.'], ['Over-fitting', 'Learning noise in the training data, so that performance drops on new data.']],
            widths=[4.2, 11.2], font=9),
        H(2, 'Appendix B Additional code listings', new=True),
        P('The listings below complete the code excerpts in Chapter 3: Grad-CAM++ ({lst:gpp}), LIME ({lst:lime}), SHAP ({lst:shap}), the deletion and '
          'insertion test ({lst:di}), the randomisation sanity check ({lst:sanity}), the EAA-IoU agreement score ({lst:eaa}) and the statistical tests ({lst:stats}). '
          'They are shortened for readability; the full scripts are listed in Appendix H.', new=True),
        CODE('gpp', 'Grad-CAM++ weights', L_GPP), CODE('lime', 'LIME super-pixel explanation', L_LIME), CODE('shap', 'SHAP for the CNN and for the Random Forest stage', L_SHAP),
        CODE('di', 'Deletion and insertion faithfulness (simplified)', L_DI), CODE('sanity', 'Cascading weight-randomisation sanity check', L_SANITY),
        CODE('eaa', 'Ensemble Attention Agreement (EAA-IoU)', L_EAA), CODE('stats', 'Exact McNemar test and bootstrap interval', L_STATS),
        H(2, 'Appendix C Additional tables', new=True),
        P('{tab:full} lists every configuration of the re-run, {tab:thr} gives the EAA-IoU threshold analysis, and {tab:oldcm} reproduces, for '
          'transparency, the HND confusion matrix that was computed on training images.', new=True),
        TAB('full', 'Full results of the standardised re-run (results_master.csv; n = 1,600; F1 per class; training time of classical models in seconds)',
            ['Model', 'Acc. (%)', 'Macro P', 'Macro R', 'Macro F1', 'F1 G', 'F1 M', 'F1 N', 'F1 P', 'Train s'], full, new=True,
            widths=[4.4, 1.3, 1.3, 1.3, 1.3, 1.1, 1.1, 1.1, 1.1, 1.3], font=7.5,
            note='Training time is recorded only for the classical classifiers; deep-model rows show 0.0 because training time was not logged for them.'),
        TAB('thr', 'EAA-IoU threshold analysis on all 1,600 test images (thresholds 0.05 and 0.1 added in this report)',
            ['Threshold', 'Images flagged (%)', 'Errors caught (%)', 'Error rate in flagged (%)', 'Error rate in unflagged (%)'], thr_rows, new=True, widths=[2.2, 3.2, 3.2, 3.4, 3.4], font=9),
        TAB('oldcm', 'Historical: final-ensemble confusion matrix from the HND report (computed on the 6,985 training images, not test data)',
            ['True \\ Predicted', 'Glioma', 'Meningioma', 'No tumour', 'Pituitary', 'Support'], old_cm, new=False, widths=[3.2, 2.2, 2.4, 2.2, 2.2, 2.0], font=9),
        H(2, 'Appendix D Proposed clinician-validation questionnaire', new=True),
        P('The HND report attached a questionnaire for healthcare professionals, but no responses were collected. It is retained and extended '
          'here as a proposed instrument for a future, ethically approved validation study (Section 7.3), not as data.', new=True),
        NL(q, new=True),
        H(2, 'Appendix E Mapping to the Pearson Unit 16 assessment criteria', new=True),
        P('{tab:crit} shows where each Unit 16 criterion (P3–D3) is addressed in this report.', new=True),
        TAB('crit', 'Where each assessment criterion is addressed', ['Criterion', 'Requirement', 'Sections'], crit, new=True, widths=[1.6, 8.0, 5.8], font=9),
        H(2, 'Appendix F Change log (differences from the HND report)', new=True),
        P('{tab:changes} lists every correction, removal and addition relative to the HND report, with the section where it appears. The audit '
          'that motivated the corrections is in AUDIT.md.', new=True),
        TAB('changes', 'Change log', ['Type', 'Section', 'Description'], changes, new=True, widths=[2.2, 3.2, 10.0], font=8.5),
        H(2, 'Appendix G Figures index', new=True),
        P('The index below maps every figure to the image file in paper_assets/figures/ that was generated by the analysis scripts at 300 dpi.', new=True),
        RAW(lambda d, r: figures_index(d, r)),
        H(2, 'Appendix H Reproducibility artefacts', new=True),
        B(['Trained models and logged results: Hugging Face repository shehank98/brain-tumor-mri-models (Kavishka, 2026), folders models/ and results_2026-10-22/.',
           'Copies of the logged results used in this report: metrics/, confusion/, history/ and logs/ in the project repository.',
           'Training code: final_code_set_2026-10-22/Brain_Tumor_Train_And_Log.ipynb (one notebook per model in individual_models/).',
           'Analysis code for this report: paper_assets/scripts/01_verify_models.py to 11_excel_charts.py (03b_xai_summary_sanity.py completes the quantitative XAI stage); outputs in paper_assets/xai_results/; figures in paper_assets/figures/; editable charts in output/Paper_Charts_Editable.xlsx.',
           'Document build and checks: paper_assets/build/ (build_paper.py, check_paper.py); audit of the HND report: AUDIT.md.'], new=True),
    ]

def figures_index(doc, r):
    rows = [[f'Figure {n}', os.path.basename(r.figpaths[k]), t] for n, k, t in r.index['fig']]
    from engine import Renderer
    t = doc.add_table(rows=1, cols=3); t.style = 'Table Grid'; t.autofit = False
    for i, h in enumerate(['Figure', 'File (paper_assets/figures/)', 'What it shows']):
        c = t.rows[0].cells[i]; c.text = ''; add_runs(c.paragraphs[0], '**' + h + '**', size=9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''; pp = cells[i].paragraphs[0]; pp.paragraph_format.line_spacing = 1.0; add_runs(pp, r.resolve(v), size=8.5)
    for row in t.rows:
        for i, w in enumerate([2.0, 5.2, 8.2]): row.cells[i].width = Cm(w)
