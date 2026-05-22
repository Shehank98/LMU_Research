"""
Page 3 — XAI Analysis
GRAD-CAM methodology, EAA-IoU novelty explanation, live XAI when models are available.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui_components import inject_css, section_header, confidence_badge, model_status_sidebar
from research_data import CLASS_NAMES

st.set_page_config(page_title='XAI Analysis · NeuroScan AI',
                   page_icon='🧠', layout='wide')
inject_css()

try:
    from model_loader import load_all_models
    from xai_engine import (
        preprocess, get_gradcam, overlay_heatmap,
        build_consensus_map, average_pairwise_iou,
        confidence_weighted_predict, iou_score,
        CONFIDENCE_THRESHOLD,
    )
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

with st.sidebar:
    st.markdown('<h2 style="color:#00b4d8;margin:0">🧠 NeuroScan AI</h2>',
                unsafe_allow_html=True)
    st.markdown('---')
    if TF_AVAILABLE:
        models, extractors, missing = load_all_models()
        model_status_sidebar(models)
        loaded_count = sum(1 for v in models.values() if v is not None)
    else:
        models, extractors, missing = {}, {}, []
        loaded_count = 0
        st.markdown('<p class="status-miss">● TensorFlow not available</p>',
                    unsafe_allow_html=True)

st.markdown('<h2 style="color:#0d1b2a">🧠 XAI Analysis — Explainable AI</h2>',
            unsafe_allow_html=True)
st.markdown(
    '<p style="color:#6b7c93">Gradient-weighted Class Activation Mapping · '
    'Ensemble Attention Agreement · Inter-Model IoU</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Section 1: What is GRAD-CAM ───────────────────────────────────────────────
section_header('What is GRAD-CAM?', 'Gradient-weighted Class Activation Mapping')

col_a, col_b = st.columns([3, 2])
with col_a:
    st.markdown(
        '<div class="clinical-card">'
        '<b>GRAD-CAM</b> (Gradient-weighted Class Activation Mapping) is an '
        'explainability technique for deep neural networks. It uses the gradients '
        'of the target class flowing into the final convolutional layer to produce '
        'a coarse localisation map highlighting which regions of an input image were '
        'important for the prediction.<br><br>'
        '<b>How it works:</b><br>'
        '1. Forward pass: compute the class prediction<br>'
        '2. Backward pass: compute gradients of the predicted class score w.r.t. '
        'the last conv layer activations<br>'
        '3. Global average pool the gradients to get neuron importance weights<br>'
        '4. Weighted combination of activation maps → ReLU → normalise to [0, 1]<br>'
        '5. Upsample to original image size and overlay as a colour heatmap<br><br>'
        '<b>Red regions</b> = high importance · <b>Blue regions</b> = low importance'
        '</div>',
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        '<div class="clinical-card">'
        '<b>Why GRAD-CAM matters in clinical AI:</b><br><br>'
        '• Clinicians can verify the model is looking at the tumour region, '
        'not an imaging artefact<br><br>'
        '• Reveals <i>where</i> a model attends even for wrong predictions<br><br>'
        '• Required for regulatory compliance in medical device AI (EU AI Act, FDA guidance)<br><br>'
        '• Enables debugging: if heatmap highlights skull/background, the model is not '
        'learning tumour morphology'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Section 2: EAA-IoU Methodology ───────────────────────────────────────────
section_header('Ensemble Attention Agreement (EAA-IoU)',
               'Novel inter-model saliency consensus score — the key XAI contribution')

st.markdown(
    '<div class="clinical-card" style="border-left:4px solid #f39c12">'
    '<b style="color:#f39c12">Novel Contribution:</b> Unlike existing ensemble-CAM methods '
    'that compare model attention against radiologist-drawn ground-truth bounding boxes '
    '(which require expensive annotation), EAA-IoU measures whether the models '
    '<i>agree with each other</i> — no annotations required.<br><br>'
    'This enables a <b>self-supervised uncertainty signal</b>: when architecturally diverse '
    'models (CNN, Xception, InceptionV3) all highlight the same region, the system is more '
    'likely to be correct. When they disagree, that scan should be escalated to a radiologist.'
    '</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    section_header('Pipeline', '')
    steps = [
        ('1', 'Apply GRAD-CAM independently to CNN, Xception, and InceptionV3 on the same MRI scan'),
        ('2', 'Binarise each heatmap at threshold t = 0.5 (attended vs. non-attended regions)'),
        ('3', 'Compute pairwise IoU between all three maps: IoU(CNN, Xcp), IoU(CNN, Inc), IoU(Xcp, Inc)'),
        ('4', 'Average the three pairwise scores → EAA-IoU (single scalar per image)'),
        ('5', 'Apply clinical decision rule based on EAA-IoU threshold'),
        ('6', 'Build consensus heatmap: confidence-weighted pixel average of all three maps'),
    ]
    for num, text in steps:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;margin-bottom:.6rem;">'
            f'<span style="background:#00b4d8;color:white;border-radius:50%;'
            f'width:24px;height:24px;display:flex;align-items:center;justify-content:center;'
            f'font-size:.75rem;font-weight:700;flex-shrink:0;margin-right:.8rem;margin-top:.1rem">'
            f'{num}</span>'
            f'<span style="font-size:.88rem;color:#2c3e50">{text}</span></div>',
            unsafe_allow_html=True,
        )

with col2:
    section_header('IoU Threshold Interpretation', '')
    thresholds = [
        ('≥ 0.65', 'Strong Agreement',   '#27ae60',
         'Models focus on the same tumour region. High confidence in spatial localisation. '
         'Prediction is reliable for clinical review.'),
        ('0.40 – 0.65', 'Moderate Agreement', '#f39c12',
         'Partial overlap. Models agree on the general area but differ on boundaries. '
         'Radiologist review is recommended.'),
        ('< 0.40', 'Low Agreement ⚠',    '#e74c3c',
         'Models attend to different regions. High uncertainty. '
         'Escalation to expert review is required regardless of class confidence.'),
    ]
    for rng, label, color, desc in thresholds:
        st.markdown(
            f'<div class="clinical-card" style="border-left:4px solid {color};margin-bottom:.6rem">'
            f'<b style="color:{color}">{rng} — {label}</b><br>'
            f'<span style="font-size:.85rem;color:#2c3e50">{desc}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Section 3: IoU formula visualisation ─────────────────────────────────────
section_header('IoU Formula', 'Intersection over Union between two binarised attention maps')

st.markdown(
    '<div class="clinical-card">'
    '<b>IoU(A, B)</b> = |A ∩ B| / |A ∪ B|<br><br>'
    'where A and B are binary masks (pixel value = 1 if attention ≥ 0.5, else 0).<br><br>'
    '• <b>|A ∩ B|</b> = number of pixels both models attended to<br>'
    '• <b>|A ∪ B|</b> = number of pixels at least one model attended to<br><br>'
    'EAA-IoU = ( IoU(CNN, Xception) + IoU(CNN, InceptionV3) + IoU(Xception, InceptionV3) ) / 3'
    '</div>',
    unsafe_allow_html=True,
)

# Visual IoU bar
fig_iou = go.Figure()
iou_scenarios = ['All same pixel\n(perfect agreement)', 'Partial overlap\n(moderate)', 'No overlap\n(complete disagreement)']
iou_vals = [1.0, 0.52, 0.0]
iou_colors = ['#27ae60', '#f39c12', '#e74c3c']

fig_iou.add_trace(go.Bar(
    x=iou_scenarios, y=iou_vals,
    marker_color=iou_colors,
    text=[f'IoU = {v:.2f}' for v in iou_vals],
    textposition='outside',
    width=0.4,
))
fig_iou.update_layout(
    yaxis=dict(range=[0, 1.3], title='IoU Score'),
    margin=dict(t=10, b=10, l=10, r=10),
    height=220,
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
)
st.plotly_chart(fig_iou, use_container_width=True)

st.divider()

# ── Section 4: Novelty positioning ───────────────────────────────────────────
section_header('Novelty Positioning', 'What this work adds vs. existing literature')

import pandas as pd

novelty_data = {
    'Capability': [
        'Per-model GRAD-CAM heatmaps',
        'Confidence-weighted ensemble voting',
        'Heatmap fusion / consensus map',
        'Inter-model IoU (EAA-IoU)',
        'IoU as clinical uncertainty trigger',
    ],
    'Kakon et al. 2025': ['✓', '✓', '✗', '✗', '✗'],
    'Adv. Dynamic Ens. 2025': ['✓ + SHAP + LIME', '✓', '✗', '✗', '✗'],
    'Majority Voting 2025': ['✓', '✗', '✗', '✗', '✗'],
    'TSO-Optimised 2024': ['✗', '✓', '✗', '✗', '✗'],
    'This Work': ['✓', '✓', '✓ (novel)', '✓ (novel)', '✓ (novel)'],
}
df_nov = pd.DataFrame(novelty_data).set_index('Capability')

def highlight_this_work(val):
    if '(novel)' in str(val):
        return 'background-color: #d4edda; color: #155724; font-weight: bold'
    elif val == '✗':
        return 'color: #c0392b'
    return ''

st.dataframe(
    df_nov.style.applymap(highlight_this_work),
    use_container_width=True,
)

st.markdown(
    '<div class="clinical-card" style="border-left:4px solid #27ae60;margin-top:1rem">'
    '<b style="color:#27ae60">Positioning Statement:</b> '
    'Unlike existing ensemble-CAM methods that compare model attention against '
    'radiologist-annotated bounding boxes, this work introduces Ensemble Attention '
    'Agreement (EAA-IoU) — a per-sample inter-model saliency consensus score computed '
    'from architecturally distinct deep models (CNN, Xception, InceptionV3). EAA-IoU '
    'is used not as an accuracy metric but as a <b>clinical uncertainty signal</b>: '
    'low inter-model agreement triggers escalation to radiologist review, independent '
    'of the ensemble\'s final class prediction.'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Section 5: Class-conditional XAI analysis ─────────────────────────────────
section_header('Class-Conditional Attention Analysis',
               'Which regions each model attends to per tumor type')

class_info = {
    'Glioma': {
        'color': '#e74c3c',
        'desc': 'Diffuse infiltration pattern. GRAD-CAM typically highlights irregular, '
                'heterogeneous regions spread across the white matter. Models often disagree '
                'on exact tumour margins due to ill-defined boundaries — leading to the '
                'lowest EAA-IoU scores and highest misclassification rate among all classes.',
        'expected_iou': 'Low–Moderate (0.35–0.55)',
        'recall': '0.97 (Final Ensemble) · 0.56 (CNN+Classical)',
    },
    'Meningioma': {
        'color': '#e67e22',
        'desc': 'Extra-axial mass, typically spherical with clear boundaries. '
                'Models tend to agree on the outer margin but differ on internal texture. '
                'Confusion with glioma occurs when tumour is near the cortex.',
        'expected_iou': 'Moderate (0.45–0.65)',
        'recall': '0.96 (Final Ensemble) · 0.87 (CNN+Classical)',
    },
    'No Tumor': {
        'color': '#27ae60',
        'desc': 'Healthy brain scan. GRAD-CAM heatmaps are diffuse with no focal hotspot. '
                'All three models tend to attend to similar anatomical landmarks '
                '(ventricles, sulci), producing higher inter-model agreement.',
        'expected_iou': 'High (0.60–0.80)',
        'recall': '0.99 (Final Ensemble) · 0.98 (CNN+Classical)',
    },
    'Pituitary': {
        'color': '#2980b9',
        'desc': 'Sellar region mass with very consistent location. The pituitary gland '
                'sits in a fixed anatomical position, so all models learn to attend to '
                'the same region. This produces the highest EAA-IoU scores and '
                'near-perfect recall across all models.',
        'expected_iou': 'High (0.65–0.85)',
        'recall': '1.00 (Final Ensemble) · 0.93 (CNN+Classical)',
    },
}

cls_cols = st.columns(4)
for col, (cls_name, info) in zip(cls_cols, class_info.items()):
    with col:
        st.markdown(
            f'<div class="clinical-card" style="border-top:4px solid {info["color"]}">'
            f'<b style="color:{info["color"]};font-size:1rem">{cls_name}</b><br><br>'
            f'<span style="font-size:.82rem;color:#2c3e50">{info["desc"]}</span><br><br>'
            f'<b style="font-size:.78rem;color:#6b7c93">Expected EAA-IoU:</b><br>'
            f'<span style="font-size:.82rem">{info["expected_iou"]}</span><br><br>'
            f'<b style="font-size:.78rem;color:#6b7c93">Ensemble Recall:</b><br>'
            f'<span style="font-size:.82rem">{info["recall"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Section 6: Live XAI (if models available) ────────────────────────────────
if not TF_AVAILABLE or loaded_count == 0:
    st.info(
        'Live XAI requires model files. Upload models to `MODELS/` or configure '
        '`HF_REPO_ID` to enable live GRAD-CAM analysis. '
        'The sections above show the methodology and expected results.'
    )
    st.stop()

gradcam_models = {
    'CNN':         models.get('cnn_feat'),
    'Xception':    models.get('xcp_feat'),
    'InceptionV3': models.get('inc_feat'),
}
available = {k: v for k, v in gradcam_models.items() if v is not None}

if not available:
    st.info('GRAD-CAM requires the CNN, Xception, or InceptionV3 feature-extractor models.')
    st.stop()

section_header('Live GRAD-CAM Analysis', 'Upload an MRI scan to compute real-time heatmaps')

if missing:
    st.warning(f'Partial model set ({loaded_count}/9). Missing: `{"`, `".join(missing)}`.')

uploaded = st.file_uploader(
    'Upload an MRI brain scan (JPG / PNG)',
    type=['jpg', 'jpeg', 'png'],
    label_visibility='collapsed',
)
if uploaded is None:
    st.info('Upload an MRI image above to begin live XAI analysis.')
    st.stop()

pil_image = Image.open(uploaded).convert('RGB')

with st.spinner('Running inference and computing GRAD-CAM heatmaps…'):
    final_class, ensemble_conf, per_model, combined_probs = \
        confidence_weighted_predict(pil_image, models, extractors)
    class_name = CLASS_NAMES[final_class]

    heatmaps = {}
    model_confs = {}
    for label, gmodel in available.items():
        img_in = preprocess(pil_image, gmodel.input_shape[1])
        try:
            hm = get_gradcam(gmodel, img_in, final_class)
            heatmaps[label] = hm
            model_confs[label] = float(np.max(gmodel.predict(img_in, verbose=0)[0]))
        except Exception:
            pass

col_orig, col_pred = st.columns([1, 1])
with col_orig:
    st.markdown('**Original MRI**')
    st.image(pil_image, use_container_width=True)
with col_pred:
    from research_data import CLASS_COLORS
    color = CLASS_COLORS.get(class_name, '#0d1b2a')
    st.markdown(
        f'<div class="clinical-card">'
        f'<p class="pred-class" style="color:{color}">{class_name}</p>'
        f'<p class="pred-conf">Ensemble confidence: {ensemble_conf*100:.1f}%</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(int(ensemble_conf * 100), text=f'**{ensemble_conf*100:.1f}%**')
    st.markdown(confidence_badge(ensemble_conf), unsafe_allow_html=True)

if heatmaps:
    st.divider()
    section_header('GRAD-CAM Heatmaps + Consensus Map', '')

    n_cols = len(heatmaps) + (1 if len(heatmaps) >= 2 else 0)
    cam_cols = st.columns(n_cols)
    labels_ord = list(heatmaps.keys())

    for i, label in enumerate(labels_ord):
        with cam_cols[i]:
            st.markdown(f'**{label}**')
            st.image(overlay_heatmap(pil_image, heatmaps[label]), use_container_width=True)
            st.caption(f'Confidence: {model_confs.get(label, 0)*100:.1f}%')

    if len(heatmaps) >= 2:
        maps_list    = [heatmaps[l] for l in labels_ord]
        weights_list = [model_confs.get(l, 1.0) for l in labels_ord]
        consensus    = build_consensus_map(maps_list, weights_list)
        iou          = average_pairwise_iou(maps_list)

        with cam_cols[len(heatmaps)]:
            st.markdown('**Consensus Map**')
            st.image(overlay_heatmap(pil_image, consensus), use_container_width=True)
            iou_color = '#27ae60' if iou >= 0.65 else ('#f39c12' if iou >= 0.40 else '#e74c3c')
            iou_label = 'Strong' if iou >= 0.65 else ('Moderate' if iou >= 0.40 else 'Low ⚠')
            st.markdown(
                f'<span style="color:{iou_color};font-weight:700">'
                f'EAA-IoU: {iou:.3f} — {iou_label}</span>',
                unsafe_allow_html=True,
            )

        st.divider()
        section_header('Pairwise IoU Breakdown', 'Inter-model attention agreement (EAA-IoU)')
        pairs = [(labels_ord[i], labels_ord[j])
                 for i in range(len(labels_ord))
                 for j in range(i+1, len(labels_ord))]
        iou_cols = st.columns(max(len(pairs), 1))
        for col, (a, b) in zip(iou_cols, pairs):
            score = iou_score(heatmaps[a], heatmaps[b])
            level = 'Strong' if score >= 0.65 else ('Moderate' if score >= 0.40 else 'Low')
            col.metric(f'{a} vs {b}', f'{score:.3f}', delta=level)

st.divider()
st.caption(
    '⚠ **Research Demonstration Only.** This system is for academic purposes at ESOFT Metro Campus. '
    'It must not be used for clinical diagnosis.'
)
