"""
Page 2 — Live MRI Diagnosis
Upload an MRI image → ensemble prediction, GRAD-CAM, IoU, per-model breakdown.
Falls back gracefully when model files are missing.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui_components import inject_css, section_header, confidence_badge, kpi_card, model_status_sidebar
from research_data import CLASS_NAMES, CLASS_COLORS

st.set_page_config(page_title='Live Diagnosis · NeuroScan AI',
                   page_icon='🔬', layout='wide')
inject_css()

# ── Import heavy modules lazily so the dashboard works without TF ─────────────
try:
    from model_loader import load_all_models
    from xai_engine import (
        preprocess, get_gradcam, overlay_heatmap,
        build_consensus_map, average_pairwise_iou,
        confidence_weighted_predict,
        CONFIDENCE_THRESHOLD,
    )
    TF_AVAILABLE = True
except Exception as _e:
    TF_AVAILABLE = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

st.markdown('<h2 style="color:#0d1b2a">🔬 Live MRI Diagnosis</h2>',
            unsafe_allow_html=True)

# ── Model availability banner ─────────────────────────────────────────────────
if not TF_AVAILABLE:
    st.error('TensorFlow is not installed in this environment. '
             'Install dependencies and restart the app to enable live inference.')
    st.stop()

total_models = 9
if loaded_count == 0:
    st.warning(
        '**No model files found.** Upload model files to `MODELS/` or configure '
        '`HF_REPO_ID` in environment variables to auto-download from HuggingFace Hub. '
        'Browse other pages to see pre-computed research results.'
    )
    if os.environ.get('HF_REPO_ID'):
        if st.button('⬇ Download Models from HuggingFace Hub'):
            with st.spinner('Downloading model files…'):
                try:
                    from model_loader import ensure_models_downloaded
                    ensure_models_downloaded()
                    st.success('Models downloaded. Please refresh the page.')
                except Exception as e:
                    st.error(f'Download failed: {e}')
    st.stop()

if missing:
    st.warning(
        f'**Partial model set loaded** ({loaded_count}/{total_models} models). '
        f'Missing: `{"`, `".join(missing)}`. '
        'Predictions will use available models only.'
    )

st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
section_header('Upload MRI Scan', 'JPG or PNG brain MRI image')

uploaded = st.file_uploader(
    'Upload an MRI brain scan image (JPG / PNG)',
    type=['jpg', 'jpeg', 'png'],
    label_visibility='collapsed',
)
if uploaded is None:
    st.info('Upload an MRI image above to begin analysis.')
    st.markdown(
        '<div class="clinical-card" style="margin-top:1rem">'
        '<b>How to use:</b><br>'
        '1. Upload a brain MRI scan (axial or coronal view, any resolution)<br>'
        '2. The ensemble of up to 6 models runs automatically<br>'
        '3. GRAD-CAM heatmaps show <i>where</i> each model focused<br>'
        '4. The IoU score measures how much models agree on the tumor region'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

pil_image = Image.open(uploaded).convert('RGB')

# ── Inference ─────────────────────────────────────────────────────────────────
with st.spinner('Running ensemble prediction…'):
    final_class, ensemble_conf, per_model, combined_probs = \
        confidence_weighted_predict(pil_image, models, extractors)

class_name  = CLASS_NAMES[final_class]
class_color = CLASS_COLORS[class_name]
conf_pct    = ensemble_conf * 100

# ── Row 1: image + prediction ─────────────────────────────────────────────────
col_img, col_pred = st.columns([1, 2])

with col_img:
    section_header('Original MRI')
    st.image(pil_image, use_container_width=True)

with col_pred:
    section_header('Ensemble Prediction')

    st.markdown(
        f'<div class="clinical-card">'
        f'<p class="pred-class" style="color:{class_color}">{class_name}</p>'
        f'<p class="pred-conf">Ensemble confidence</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(int(conf_pct), text=f'**{conf_pct:.1f}%** confidence')
    st.markdown(confidence_badge(ensemble_conf), unsafe_allow_html=True)

    if ensemble_conf >= CONFIDENCE_THRESHOLD:
        st.success('✔ HIGH CONFIDENCE — models strongly agree.')
    elif ensemble_conf >= 0.50:
        st.warning('⚠ MODERATE CONFIDENCE — some disagreement. Radiologist review recommended.')
    else:
        st.error('✖ LOW CONFIDENCE — significant disagreement. Expert review required.')

    # Class probability bar chart
    st.markdown('**Class probability distribution**')
    prob_norm = combined_probs / combined_probs.sum() * 100
    fig_probs = go.Figure(go.Bar(
        x=CLASS_NAMES,
        y=prob_norm.tolist(),
        marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES],
        text=[f'{v:.1f}%' for v in prob_norm],
        textposition='outside',
    ))
    fig_probs.update_layout(
        yaxis=dict(range=[0, 115], title='Probability (%)'),
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
    )
    st.plotly_chart(fig_probs, use_container_width=True)

st.divider()

# ── GRAD-CAM heatmaps ─────────────────────────────────────────────────────────
section_header('GRAD-CAM Heatmaps', 'Regions each model attended to for its prediction')

gradcam_sources = {
    'CNN':         models.get('cnn_feat'),
    'Xception':    models.get('xcp_feat'),
    'InceptionV3': models.get('inc_feat'),
}
heatmaps    = {}
gradcam_conf = {}

for label, gmodel in gradcam_sources.items():
    if gmodel is None:
        continue
    with st.spinner(f'Computing GRAD-CAM for {label}…'):
        img_in = preprocess(pil_image, gmodel.input_shape[1])
        try:
            hm = get_gradcam(gmodel, img_in, final_class)
            heatmaps[label] = hm
            gradcam_conf[label] = float(np.max(gmodel.predict(img_in, verbose=0)[0]))
        except Exception:
            pass

if heatmaps:
    n_cols   = len(heatmaps) + (1 if len(heatmaps) >= 2 else 0)
    cam_cols = st.columns(n_cols)

    for i, (label, hm) in enumerate(heatmaps.items()):
        with cam_cols[i]:
            st.markdown(f'**{label}**')
            overlay = overlay_heatmap(pil_image, hm)
            st.image(overlay, use_container_width=True)
            conf_val = gradcam_conf.get(label, 0.0)
            st.caption(f'Confidence: {conf_val*100:.1f}%')

    if len(heatmaps) >= 2:
        labels_ord  = list(heatmaps.keys())
        maps_list   = [heatmaps[l] for l in labels_ord]
        weights_list = [gradcam_conf.get(l, 1.0) for l in labels_ord]
        consensus   = build_consensus_map(maps_list, weights_list)
        iou         = average_pairwise_iou(maps_list)

        with cam_cols[len(heatmaps)]:
            st.markdown('**Consensus Map**')
            c_overlay = overlay_heatmap(pil_image, consensus)
            st.image(c_overlay, use_container_width=True)

            if iou >= 0.65:
                st.caption(f'EAA-IoU: **{iou:.3f}** — Strong agreement')
            elif iou >= 0.40:
                st.caption(f'EAA-IoU: **{iou:.3f}** — Moderate agreement')
            else:
                st.caption(f'EAA-IoU: **{iou:.3f}** — Low agreement ⚠')

        # Pairwise IoU breakdown
        st.divider()
        section_header('Pairwise IoU Breakdown', 'Inter-model attention agreement (EAA-IoU)')
        from xai_engine import iou_score
        pairs = [(labels_ord[i], labels_ord[j])
                 for i in range(len(labels_ord))
                 for j in range(i+1, len(labels_ord))]
        iou_cols = st.columns(len(pairs))
        for col, (a, b) in zip(iou_cols, pairs):
            score = iou_score(heatmaps[a], heatmaps[b])
            level = 'Strong' if score >= 0.65 else ('Moderate' if score >= 0.40 else 'Low')
            col.metric(f'{a} vs {b}', f'{score:.3f}', delta=level)

else:
    st.info('GRAD-CAM requires the CNN, Xception, or InceptionV3 feature-extractor models.')

st.divider()

# ── Per-model vote breakdown ──────────────────────────────────────────────────
if per_model:
    section_header('Per-Model Vote Breakdown', 'Top predicted class and confidence per model')

    vote_labels = list(per_model.keys())
    vote_confs  = [float(np.max(v)) * 100 for v in per_model.values()]
    vote_preds  = [CLASS_NAMES[int(np.argmax(v))] for v in per_model.values()]
    vote_colors = [CLASS_COLORS[p] for p in vote_preds]

    fig_votes = go.Figure(go.Bar(
        x=vote_labels, y=vote_confs,
        marker_color=vote_colors,
        text=[f'{p}<br>{c:.1f}%' for p, c in zip(vote_preds, vote_confs)],
        textposition='outside',
    ))
    fig_votes.update_layout(
        yaxis=dict(range=[0, 120], title='Top-class confidence (%)'),
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
    )
    st.plotly_chart(fig_votes, use_container_width=True)

    with st.expander('Show detailed probability table'):
        import pandas as pd
        rows = []
        for model_name, probs in per_model.items():
            row = {'Model': model_name,
                   'Predicted': CLASS_NAMES[int(np.argmax(probs))]}
            for cls, p in zip(CLASS_NAMES, probs):
                row[cls] = f'{p*100:.1f}%'
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index('Model'),
                     use_container_width=True)

st.divider()
st.caption(
    '⚠ **Research Demonstration Only.** This system is for academic purposes at ESOFT Metro Campus. '
    'It must not be used for clinical diagnosis. Always consult a qualified radiologist.'
)
