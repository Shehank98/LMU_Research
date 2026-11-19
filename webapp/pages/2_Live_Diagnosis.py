"""
Page 2: Live MRI Diagnosis
Upload an MRI image -> six-predictor ensemble prediction, Grad-CAM of the three deep networks,
consensus map, EAA-IoU agreement score and an escalation alert when agreement is low.
Prediction and explanation logic is shared with the API (backend/xai_engine.py), so the
Streamlit app and the API always give the same answer for the same image.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))

from ui_components import inject_css, section_header

st.set_page_config(page_title="Live Diagnosis · NeuroScan AI", page_icon="🔬", layout="wide")
inject_css()

try:
    import model_loader as ML
    from xai_engine import CLASS_NAMES, EAA_HIGH, EAA_LOW, explain, predict_all

    TF_AVAILABLE = True
except Exception as exc:  # TensorFlow missing: the other pages still work
    TF_AVAILABLE = False
    IMPORT_ERROR = str(exc)

CLASS_COLORS = {"Glioma": "#e74c3c", "Meningioma": "#e67e22", "No Tumor": "#27ae60", "Pituitary": "#2980b9"}


@st.cache_resource(show_spinner="Loading the six predictors…")
def load_models():
    return ML.get_models()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h2 style="color:#00b4d8;margin:0">🧠 NeuroScan AI</h2>', unsafe_allow_html=True)
    st.markdown("---")
    if TF_AVAILABLE:
        models, extractors, missing = load_models()
        loaded = sum(v is not None for v in models.values())
        st.markdown(f"**Models loaded:** {loaded}/{len(models)}")
        st.caption(f"Alert below EAA-IoU {EAA_LOW:.2f} · strong agreement from {EAA_HIGH:.2f}")

st.markdown('<h2 style="color:#0d1b2a">🔬 Live MRI Diagnosis</h2>', unsafe_allow_html=True)
st.caption("Six-predictor ensemble with Grad-CAM explanations and an inter-model agreement check (EAA-IoU).")

if not TF_AVAILABLE:
    st.error("TensorFlow is not installed in this environment: " + IMPORT_ERROR)
    st.stop()
if not any(v is not None for v in models.values()):
    st.warning("No model files found. Set MODELS_DIR to the folder with the trained models, or HF_REPO_ID.")
    st.stop()
if missing:
    st.warning("Partial model set loaded. Missing: " + ", ".join(missing))

st.divider()
section_header("Upload MRI Scan", "JPG or PNG brain MRI image")
uploaded = st.file_uploader(
    "Upload an MRI brain scan image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)
if uploaded is None:
    st.info("Upload an MRI image above to begin the analysis.")
    st.stop()

pil_image = Image.open(uploaded).convert("RGB")

with st.spinner("Running the six predictors…"):
    final_class, confidence, per_model, fused = predict_all(pil_image, models, extractors)
with st.spinner("Computing Grad-CAM for the three deep networks…"):
    xai = explain(pil_image, models, final_class, per_model)

class_name = CLASS_NAMES[final_class]
eaa = xai["eaa_iou"]

# ── Row 1: image, prediction and alert ────────────────────────────────────────
col_img, col_pred = st.columns([1, 2])
with col_img:
    section_header("Uploaded MRI")
    st.image(pil_image, use_container_width=True)
with col_pred:
    section_header("Ensemble Prediction")
    st.markdown(
        f'<div class="clinical-card"><p class="pred-class" style="color:{CLASS_COLORS[class_name]}">{class_name}</p>'
        f'<p class="pred-conf">Confidence-weighted ensemble score</p></div>',
        unsafe_allow_html=True,
    )
    st.progress(int(round(confidence * 100)), text=f"**{confidence * 100:.1f}%** of the summed probability")
    if xai["alert"]:
        st.error(
            f"⚠ ESCALATE FOR RADIOLOGIST REVIEW: the three networks look at different regions "
            f"(EAA-IoU {eaa:.3f} < {EAA_LOW:.2f}). Treat this prediction with caution."
        )
    elif eaa is not None and eaa >= EAA_HIGH:
        st.success(f"✔ Strong attention agreement between the networks (EAA-IoU {eaa:.3f}).")
    elif eaa is not None:
        st.info(f"Moderate attention agreement (EAA-IoU {eaa:.3f}).")

    st.markdown("**Class probability distribution**")
    fig = go.Figure(
        go.Bar(
            x=[p * 100 for p in fused],
            y=CLASS_NAMES,
            orientation="h",
            marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES],
            text=[f"{p * 100:.1f}%" for p in fused],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 115], title="%"),
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: Grad-CAM heatmaps and consensus map ────────────────────────────────
st.divider()
section_header("Grad-CAM Heatmaps", "Where each deep network looked when predicting " + class_name)
names = list(xai["maps"])
cols = st.columns(len(names) + 1)
for col, n in zip(cols, names):
    col.markdown(f"**{n}**")
    col.image(xai["maps"][n], use_container_width=True)
if xai["consensus"] is not None:
    cols[-1].markdown("**Consensus map**")
    cols[-1].image(xai["consensus"], use_container_width=True)
    cols[-1].caption("Confidence-weighted average of the three maps")

pair_cols = st.columns(len(xai["pairwise_iou"]) + 1)
for col, p in zip(pair_cols, xai["pairwise_iou"]):
    col.metric(p["pair"], f"{p['iou']:.3f}")
if eaa is not None:
    pair_cols[-1].metric("EAA-IoU (mean)", f"{eaa:.3f}", delta=xai["agreement_level"], delta_color="off")

# ── Row 3: per-model vote breakdown ───────────────────────────────────────────
st.divider()
section_header(
    "Per-Model Vote Breakdown", "Predicted class and its probability for each of the six predictors"
)
rows = []
for name, family, p in per_model:
    rows.append(
        {
            "Predictor": name,
            "Type": family,
            "Vote": CLASS_NAMES[int(np.argmax(p))],
            "Probability (%)": round(float(p.max()) * 100, 1),
        }
    )
votes = pd.DataFrame(rows)
fig_v = go.Figure(
    go.Bar(
        x=votes.Predictor,
        y=votes["Probability (%)"],
        marker_color=[CLASS_COLORS[v] for v in votes.Vote],
        text=[f"{v}<br>{c:.1f}%" for v, c in zip(votes.Vote, votes["Probability (%)"])],
        textposition="outside",
    )
)
fig_v.update_layout(
    yaxis=dict(range=[0, 120], title="Probability of voted class (%)"),
    height=300,
    margin=dict(t=10, b=10, l=10, r=10),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig_v, use_container_width=True)
majority = votes.Vote.value_counts()
st.caption(f"Majority vote: {majority.index[0]} ({majority.iloc[0]} of {len(votes)} predictors).")
with st.expander("Show the full probability table"):
    table = pd.DataFrame(
        [{"Predictor": n, **{c: f"{v * 100:.1f}%" for c, v in zip(CLASS_NAMES, p)}} for n, _, p in per_model]
    ).set_index("Predictor")
    st.dataframe(table, use_container_width=True)

st.divider()
st.caption(
    "⚠ **Research demonstration only.** This prototype supports, and does not replace, a qualified radiologist. "
    "It must not be used for clinical diagnosis."
)
