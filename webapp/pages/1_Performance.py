"""
Page 1 — Model Performance Dashboard
All charts from pre-computed research results. No model files required.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui_components import inject_css, section_header
from research_data import (
    CLASS_NAMES, CLASS_COLORS,
    MODEL_ACCURACY, MODEL_FAMILY, FAMILY_COLORS,
    CONFUSION_MATRICES, CLASSIFICATION_REPORTS,
    FEATURE_ML_ACCURACY, RAW_PIXEL_ACCURACY,
)

st.set_page_config(page_title='Performance Dashboard · NeuroScan AI',
                   page_icon='📊', layout='wide')
inject_css()

with st.sidebar:
    st.markdown('<h2 style="color:#00b4d8;margin:0">🧠 NeuroScan AI</h2>',
                unsafe_allow_html=True)

st.markdown('<h2 style="color:#0d1b2a">📊 Model Performance Dashboard</h2>',
            unsafe_allow_html=True)
st.markdown('<p style="color:#6b7c93">Pre-computed results from the full research study '
            '(2,063 test images, 4 classes)</p>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Accuracy Overview — grouped bar
# ─────────────────────────────────────────────────────────────────────────────
section_header('Accuracy Overview', 'Standalone vs ensemble accuracy per backbone')

groups = {
    'CNN':         ('CNN Standalone',   'CNN + Classical Ensemble'),
    'InceptionV3': ('InceptionV3 Standalone', 'InceptionV3 + Classical Ens.'),
    'Xception':    ('Xception Standalone', 'Xception + Classical Ens.'),
}

fig_grp = go.Figure()
bar_types = [
    ('Standalone',        ['CNN Standalone', 'InceptionV3 Standalone', 'Xception Standalone'],
     '#2980b9', False),
    ('+ Classical Ens.',  ['CNN + Classical Ensemble', 'InceptionV3 + Classical Ens.', 'Xception + Classical Ens.'],
     '#8e44ad', False),
    ('Final Ensemble',    ['Final Ensemble (6-model)'],
     '#f39c12', False),
]
for trace_name, model_keys, color, _ in bar_types:
    vals = [MODEL_ACCURACY.get(k, 0) for k in model_keys]
    fig_grp.add_trace(go.Bar(
        name=trace_name,
        x=[k.split(' Standalone')[0].split(' +')[0] for k in model_keys],
        y=vals,
        marker_color=color,
        text=[f'{v:.2f}%' for v in vals],
        textposition='outside',
    ))

fig_grp.update_layout(
    barmode='group',
    yaxis=dict(range=[60, 105], title='Accuracy (%)'),
    margin=dict(l=10, r=10, t=10, b=10),
    height=340,
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
)
st.plotly_chart(fig_grp, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion Matrices — 4-panel
# ─────────────────────────────────────────────────────────────────────────────
section_header('Confusion Matrices', 'Rows = actual class · Columns = predicted class')

cm_names = list(CONFUSION_MATRICES.keys())
fig_cm = make_subplots(
    rows=2, cols=2,
    subplot_titles=cm_names,
    horizontal_spacing=0.12,
    vertical_spacing=0.18,
)

for idx, (name, matrix) in enumerate(CONFUSION_MATRICES.items()):
    r, c = divmod(idx, 2)
    mat = np.array(matrix)
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_pct  = (mat / row_sums * 100).round(1)

    annot = [[f'{mat[i,j]}<br>({mat_pct[i,j]}%)' for j in range(4)] for i in range(4)]

    fig_cm.add_trace(
        go.Heatmap(
            z=mat_pct,
            x=CLASS_NAMES, y=CLASS_NAMES,
            colorscale='Blues',
            showscale=False,
            text=annot,
            texttemplate='%{text}',
            textfont={'size': 10},
        ),
        row=r+1, col=c+1,
    )

fig_cm.update_layout(
    height=680,
    margin=dict(l=10, r=10, t=60, b=10),
    paper_bgcolor='white',
)
for i in range(1, 5):
    fig_cm.update_xaxes(tickfont_size=10, row=(i-1)//2+1, col=(i-1)%2+1)
    fig_cm.update_yaxes(tickfont_size=10, row=(i-1)//2+1, col=(i-1)%2+1)

st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-Class Metrics Table with colour-coded F1
# ─────────────────────────────────────────────────────────────────────────────
section_header('Per-Class Classification Metrics', 'Precision · Recall · F1-Score by model')

rows = []
for model_name, report in CLASSIFICATION_REPORTS.items():
    for cls in CLASS_NAMES:
        m = report[cls]
        rows.append({
            'Model':     model_name,
            'Class':     cls,
            'Precision': f"{m['precision']:.2f}",
            'Recall':    f"{m['recall']:.2f}",
            'F1-Score':  f"{m['f1']:.2f}",
        })
df_metrics = pd.DataFrame(rows)
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

# Radar chart — F1 per class per model
section_header('F1-Score Radar', 'Per-class F1 across models')

fig_radar = go.Figure()
radar_colors = ['#00b4d8', '#e74c3c', '#27ae60', '#f39c12']
for i, (model_name, report) in enumerate(CLASSIFICATION_REPORTS.items()):
    f1_vals = [report[cls]['f1'] for cls in CLASS_NAMES]
    f1_vals.append(f1_vals[0])  # close polygon
    cats = CLASS_NAMES + [CLASS_NAMES[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=f1_vals, theta=cats,
        fill='toself', opacity=0.35,
        name=model_name,
        line_color=radar_colors[i % len(radar_colors)],
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(range=[0, 1.05])),
    margin=dict(l=40, r=40, t=20, b=20),
    height=400,
    paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=-0.15),
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Classical ML breakdown by feature source
# ─────────────────────────────────────────────────────────────────────────────
section_header('Classical ML by Feature Source',
               'RF / DT / SVM accuracy when trained on different feature extractors')

clf_names    = ['RF', 'DT', 'SVM', 'Ensemble']
clf_colors   = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12']
feature_srcs = list(FEATURE_ML_ACCURACY.keys())

fig_feat = go.Figure()
for clf, color in zip(clf_names, clf_colors):
    fig_feat.add_trace(go.Bar(
        name=clf,
        x=feature_srcs,
        y=[FEATURE_ML_ACCURACY[src].get(clf, 0) for src in feature_srcs],
        marker_color=color,
        text=[f"{FEATURE_ML_ACCURACY[src].get(clf, 0):.1f}%" for src in feature_srcs],
        textposition='outside',
    ))
fig_feat.update_layout(
    barmode='group',
    yaxis=dict(range=[50, 100], title='Accuracy (%)'),
    margin=dict(l=10, r=10, t=10, b=10),
    height=340,
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
)
st.plotly_chart(fig_feat, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Class difficulty — worst-performing class per model (lowest recall)
# ─────────────────────────────────────────────────────────────────────────────
section_header('Class Difficulty Analysis',
               'Recall per class — lower = harder to correctly identify')

fig_diff = go.Figure()
for model_name, report in CLASSIFICATION_REPORTS.items():
    recall_vals = [report[cls]['recall'] for cls in CLASS_NAMES]
    fig_diff.add_trace(go.Bar(
        name=model_name.split(' Ens.')[0].split(' + ')[0],
        x=CLASS_NAMES,
        y=recall_vals,
        text=[f'{v:.2f}' for v in recall_vals],
        textposition='outside',
    ))
fig_diff.update_layout(
    barmode='group',
    yaxis=dict(range=[0, 1.1], title='Recall'),
    margin=dict(l=10, r=10, t=10, b=10),
    height=340,
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
)
st.plotly_chart(fig_diff, use_container_width=True)

st.caption('Glioma consistently has the lowest recall — the most frequently misclassified class. '
           'This is expected as glioma can appear morphologically similar to meningioma on MRI.')
