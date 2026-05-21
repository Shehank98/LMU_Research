import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from model_loader import load_all_models
from xai_engine import (
    CLASS_NAMES, CLASS_COLORS, CONFIDENCE_THRESHOLD,
    preprocess, get_gradcam, overlay_heatmap,
    build_consensus_map, average_pairwise_iou,
    confidence_weighted_predict,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier — XAI Ensemble",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor MRI Classifier")
st.caption(
    "Ensemble Consensus XAI System · CNN + Xception + InceptionV3 + Classical ML "
    "· Confidence-Weighted Voting · Shehan Kavishka E187041 · ESOFT Metro Campus"
)
st.divider()

# ── Load models ──────────────────────────────────────────────────────────────
models, extractors, missing = load_all_models()

if missing:
    st.warning(
        f"**Missing model files:** {', '.join(missing)}  \n"
        "Place them in the `MODELS/` folder. "
        "Some predictions will be skipped until all files are present."
    )

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an MRI brain scan image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.info("Upload an MRI image above to begin analysis.")
    st.stop()

pil_image = Image.open(uploaded).convert("RGB")

# ── Run inference ─────────────────────────────────────────────────────────────
with st.spinner("Running ensemble prediction..."):
    final_class, ensemble_conf, per_model, combined_probs = \
        confidence_weighted_predict(pil_image, models, extractors)

class_name = CLASS_NAMES[final_class]
class_color = CLASS_COLORS[class_name]
conf_pct = ensemble_conf * 100

# ── Top row: image + prediction ───────────────────────────────────────────────
col_img, col_pred = st.columns([1, 2])

with col_img:
    st.subheader("Original MRI")
    st.image(pil_image, use_container_width=True)

with col_pred:
    st.subheader("Prediction")
    st.markdown(
        f"<h2 style='color:{class_color};margin-bottom:0'>{class_name}</h2>",
        unsafe_allow_html=True,
    )
    st.progress(int(conf_pct), text=f"Ensemble confidence: **{conf_pct:.1f}%**")

    if ensemble_conf >= CONFIDENCE_THRESHOLD:
        st.success("HIGH CONFIDENCE — models strongly agree.")
    elif ensemble_conf >= 0.50:
        st.warning(
            "MODERATE CONFIDENCE — some disagreement between models. "
            "Radiologist review recommended."
        )
    else:
        st.error(
            "LOW CONFIDENCE — models disagree significantly. "
            "This case should be reviewed by a qualified radiologist."
        )

    # Class probability bar chart
    st.markdown("**Class probability distribution**")
    fig_probs = go.Figure(go.Bar(
        x=CLASS_NAMES,
        y=(combined_probs / combined_probs.sum() * 100).tolist(),
        marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES],
        text=[f"{v:.1f}%" for v in (combined_probs / combined_probs.sum() * 100)],
        textposition="outside",
    ))
    fig_probs.update_layout(
        yaxis_title="Probability (%)",
        yaxis_range=[0, 110],
        margin=dict(t=10, b=10),
        height=260,
        showlegend=False,
    )
    st.plotly_chart(fig_probs, use_container_width=True)

st.divider()

# ── GRAD-CAM heatmaps ─────────────────────────────────────────────────────────
st.subheader("GRAD-CAM Heatmaps — Where Each Model Looked")

img_256 = preprocess(pil_image, 256)
heatmaps = {}
gradcam_models = {
    'CNN':         models.get('cnn_feat'),   # use feature extractor (has conv layers)
    'Xception':    models.get('xcp_feat'),
    'InceptionV3': models.get('inc_feat'),
}
gradcam_confs = {}

for label, gmodel in gradcam_models.items():
    if gmodel is None:
        continue
    with st.spinner(f"Computing GRAD-CAM for {label}..."):
        img_in = preprocess(pil_image, gmodel.input_shape[1])
        try:
            hm = get_gradcam(gmodel, img_in, final_class)
            heatmaps[label] = hm
            # confidence = max of softmax output for this model
            raw_pred = gmodel.predict(img_in, verbose=0)[0]
            gradcam_confs[label] = float(np.max(raw_pred))
        except Exception:
            pass  # model architecture may not support this GRAD-CAM approach

cam_cols = st.columns(len(heatmaps) + 1)

for i, (label, hm) in enumerate(heatmaps.items()):
    with cam_cols[i]:
        st.markdown(f"**{label}**")
        overlay = overlay_heatmap(pil_image, hm)
        st.image(overlay, use_container_width=True)
        conf_val = gradcam_confs.get(label, 0.0)
        st.caption(f"Model confidence: {conf_val*100:.1f}%")

# Consensus map
if len(heatmaps) >= 2:
    labels_ordered = list(heatmaps.keys())
    maps_list = [heatmaps[l] for l in labels_ordered]
    weights_list = [gradcam_confs.get(l, 1.0) for l in labels_ordered]
    consensus = build_consensus_map(maps_list, weights_list)
    iou = average_pairwise_iou(maps_list)

    with cam_cols[len(heatmaps)]:
        st.markdown("**Consensus Map**")
        consensus_overlay = overlay_heatmap(pil_image, consensus)
        st.image(consensus_overlay, use_container_width=True)

        if iou >= 0.65:
            st.caption(f"Agreement (IoU): **{iou:.2f}** — Strong")
        elif iou >= 0.40:
            st.caption(f"Agreement (IoU): **{iou:.2f}** — Moderate")
        else:
            st.caption(f"Agreement (IoU): **{iou:.2f}** — Low")

st.divider()

# ── Per-model vote breakdown ───────────────────────────────────────────────────
st.subheader("Per-Model Vote Breakdown")

if per_model:
    vote_labels = list(per_model.keys())
    vote_confs = [float(np.max(v)) * 100 for v in per_model.values()]
    vote_preds = [CLASS_NAMES[int(np.argmax(v))] for v in per_model.values()]
    vote_colors = [CLASS_COLORS[p] for p in vote_preds]

    fig_votes = go.Figure(go.Bar(
        x=vote_labels,
        y=vote_confs,
        marker_color=vote_colors,
        text=[f"{p}<br>{c:.1f}%" for p, c in zip(vote_preds, vote_confs)],
        textposition="outside",
    ))
    fig_votes.update_layout(
        yaxis_title="Top-class confidence (%)",
        yaxis_range=[0, 115],
        margin=dict(t=10, b=10),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_votes, use_container_width=True)

    # Detailed table
    with st.expander("Show detailed probability table"):
        import pandas as pd
        rows = []
        for model_name, probs in per_model.items():
            row = {"Model": model_name, "Predicted": CLASS_NAMES[int(np.argmax(probs))]}
            for cls, p in zip(CLASS_NAMES, probs):
                row[cls] = f"{p*100:.1f}%"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

st.divider()

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.caption(
    "⚠ **Research Demonstration Only.** This system is built for academic purposes "
    "as part of an HND Computing Research Project at ESOFT Metro Campus. "
    "It must not be used for clinical diagnosis. "
    "Always consult a qualified radiologist for medical decisions."
)
