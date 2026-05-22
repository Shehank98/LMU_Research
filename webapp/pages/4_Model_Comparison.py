"""
Page 4 — Model Comparison
Literature positioning, ensemble value-add, feature source breakdown, model complexity scatter.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui_components import inject_css, section_header
from research_data import (
    MODEL_ACCURACY, MODEL_FAMILY, FAMILY_COLORS,
    FEATURE_ML_ACCURACY, RAW_PIXEL_ACCURACY,
    ENSEMBLE_BUILDUP, LITERATURE, MODEL_PARAMS,
)

st.set_page_config(page_title='Model Comparison · NeuroScan AI',
                   page_icon='📈', layout='wide')
inject_css()

with st.sidebar:
    st.markdown('<h2 style="color:#00b4d8;margin:0">🧠 NeuroScan AI</h2>',
                unsafe_allow_html=True)

st.markdown('<h2 style="color:#0d1b2a">📈 Model Comparison & Literature Context</h2>',
            unsafe_allow_html=True)
st.markdown(
    '<p style="color:#6b7c93">Where this system sits relative to published work · '
    'Ensemble value-add · Feature source analysis</p>',
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Literature comparison table
# ─────────────────────────────────────────────────────────────────────────────
section_header('Literature Comparison',
               'This work positioned against the closest published ensemble-XAI systems')

df_lit = pd.DataFrame(LITERATURE)

def highlight_this_work(val):
    return 'background-color: #fff3cd; font-weight: bold' if '**' in str(val) else ''

st.dataframe(
    df_lit.style.applymap(highlight_this_work),
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    '<div class="clinical-card" style="border-left:4px solid #27ae60;margin-top:.5rem">'
    '<b style="color:#27ae60">Key Differentiator:</b> '
    'No published paper combines all five capabilities: per-model GRAD-CAM, '
    'confidence-weighted voting, weighted heatmap fusion, <b>inter-model IoU</b>, '
    'and using that IoU as a <b>clinical escalation trigger</b>. '
    'Columns 4 and 5 are unoccupied in the prior literature table.'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Ensemble build-up — accuracy gain as models are added
# ─────────────────────────────────────────────────────────────────────────────
section_header('Ensemble Build-Up Analysis',
               'Accuracy gain as each model is added to the ensemble (1 → 6 models)')

fig_build = go.Figure()
x_vals  = list(range(1, len(ENSEMBLE_BUILDUP) + 1))
y_vals  = [e['accuracy'] for e in ENSEMBLE_BUILDUP]
labels  = [e['models'] for e in ENSEMBLE_BUILDUP]

fig_build.add_trace(go.Scatter(
    x=x_vals, y=y_vals,
    mode='lines+markers+text',
    text=[f"{v}%" for v in y_vals],
    textposition='top center',
    line=dict(color='#00b4d8', width=3),
    marker=dict(size=12, color='#00b4d8'),
    hovertext=labels,
    hoverinfo='text+y',
))

# Shade the gain region
fig_build.add_trace(go.Scatter(
    x=x_vals, y=y_vals,
    fill='tozeroy', fillcolor='rgba(0,180,216,0.08)',
    line=dict(color='rgba(0,0,0,0)'),
    showlegend=False, hoverinfo='skip',
))

fig_build.update_layout(
    xaxis=dict(
        tickvals=x_vals,
        ticktext=[l.split('(')[0].strip()[:30] for l in labels],
        tickangle=-25,
        title='Models included',
    ),
    yaxis=dict(range=[85, 101], title='Test Accuracy (%)'),
    margin=dict(l=10, r=10, t=20, b=100),
    height=380,
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
)
fig_build.add_hline(y=98.20, line_dash='dash', line_color='#f39c12',
                    annotation_text='Final 98.20%',
                    annotation_position='bottom right')

st.plotly_chart(fig_build, use_container_width=True)

# Gain annotations
gain_cols = st.columns(5)
for i, col in enumerate(gain_cols):
    gain = round(y_vals[i+1] - y_vals[i], 2)
    col.metric(
        f'Add model {i+2}',
        f'+{gain}%',
        delta=ENSEMBLE_BUILDUP[i+1]['models'].split('+')[-1].strip()[:20],
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature source comparison — deep features vs raw pixels
# ─────────────────────────────────────────────────────────────────────────────
section_header('Feature Source Comparison',
               'Classical ML accuracy depending on which features are used')

clf_names  = ['RF', 'DT', 'SVM', 'Ensemble']
clf_colors = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12']
feat_srcs  = list(FEATURE_ML_ACCURACY.keys())

fig_feat = go.Figure()
for clf, color in zip(clf_names, clf_colors):
    fig_feat.add_trace(go.Bar(
        name=clf,
        x=feat_srcs,
        y=[FEATURE_ML_ACCURACY[src].get(clf, 0) for src in feat_srcs],
        marker_color=color,
        text=[f"{FEATURE_ML_ACCURACY[src].get(clf, 0):.1f}%" for src in feat_srcs],
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

col_obs1, col_obs2 = st.columns(2)
with col_obs1:
    st.markdown(
        '<div class="clinical-card">'
        '<b>Key finding — CNN features dominate:</b><br>'
        'CNN-extracted features outperform both Xception and InceptionV3 features for '
        'classical ML, despite CNN being the simplest architecture. '
        'RF on CNN features achieves 85.45% vs 73–74% on transfer-learning features.<br><br>'
        '<b>Possible reason:</b> The CNN backbone was specifically trained on this dataset '
        '(no ImageNet pretraining), making its learned features more domain-specific.'
        '</div>',
        unsafe_allow_html=True,
    )
with col_obs2:
    st.markdown(
        '<div class="clinical-card">'
        '<b>Raw pixels surprisingly competitive:</b><br>'
        'SVM on raw 256×256 pixels achieves 88.26% — better than SVM on '
        'any deep feature set (66.6% CNN, 57.3% Inc, 56.7% Xcp).<br><br>'
        '<b>Implication:</b> SVM hyperplane is sensitive to the feature distribution. '
        'Deep features are 256-dimensional structured vectors; raw pixels are 65,536-dimensional '
        'but the high dimensionality suits SVM\'s kernel trick in this task.'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Raw pixel classical ML breakdown
# ─────────────────────────────────────────────────────────────────────────────
section_header('Classical ML on Raw Pixels',
               'SVM / LR / DT / RF accuracy when trained directly on pixel values (no deep features)')

raw_models = list(RAW_PIXEL_ACCURACY.keys())
raw_vals   = list(RAW_PIXEL_ACCURACY.values())
raw_colors = ['#8e44ad' if 'SVM' in m else
              '#2980b9' if 'Logistic' in m else
              '#16a085' if 'Decision' in m else
              '#e74c3c' if 'Random' in m else '#f39c12'
              for m in raw_models]

fig_raw = go.Figure(go.Bar(
    x=raw_models, y=raw_vals,
    marker_color=raw_colors,
    text=[f'{v:.2f}%' for v in raw_vals],
    textposition='outside',
))
fig_raw.update_layout(
    yaxis=dict(range=[70, 96], title='Accuracy (%)'),
    margin=dict(l=10, r=10, t=10, b=10),
    height=280,
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
)
st.plotly_chart(fig_raw, use_container_width=True)
st.caption(
    'SVM achieves 88.26% on raw pixel values — the best single classical ML result. '
    'Ensemble of DT+RF+SVM reaches 83%, lower than SVM alone due to weaker DT/RF performance dragging the vote.'
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Model complexity vs accuracy scatter
# ─────────────────────────────────────────────────────────────────────────────
section_header('Model Complexity vs Accuracy',
               'Parameter count (millions) vs test accuracy — efficiency tradeoff')

fig_scatter = go.Figure()
for model_name, params in MODEL_PARAMS.items():
    acc = MODEL_ACCURACY.get(model_name, None)
    if acc is None:
        continue
    family = MODEL_FAMILY.get(model_name, '')
    color  = FAMILY_COLORS.get(family, '#95a5a6')
    fig_scatter.add_trace(go.Scatter(
        x=[params], y=[acc],
        mode='markers+text',
        marker=dict(size=16, color=color, line=dict(width=1, color='white')),
        text=[model_name.split(' Standalone')[0].split(' +')[0][:18]],
        textposition='top center',
        textfont=dict(size=10),
        name=family,
        showlegend=False,
        hovertext=f'{model_name}<br>{params}M params<br>{acc:.2f}% accuracy',
        hoverinfo='text',
    ))

fig_scatter.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=10, color='#f39c12'),
    name='Final Ensemble',
    showlegend=True,
))

fig_scatter.update_layout(
    xaxis=dict(title='Model Parameters (millions)', type='log'),
    yaxis=dict(range=[65, 96], title='Test Accuracy (%)'),
    margin=dict(l=10, r=10, t=20, b=20),
    height=380,
    plot_bgcolor='white',
    paper_bgcolor='white',
)

# Add family legend via dummy traces
for fam, col in FAMILY_COLORS.items():
    if fam == 'Final':
        continue
    fig_scatter.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color=col),
        name=fam, showlegend=True,
    ))

st.plotly_chart(fig_scatter, use_container_width=True)
st.caption(
    'CNN has ~0.9M parameters but achieves 91.18% — the best efficiency ratio. '
    'Xception and InceptionV3 use 22–23M parameters for 84–87% standalone accuracy. '
    'The 6-model ensemble combines them all but the combined parameter count is not additive '
    'in inference (models run in parallel).'
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Full model accuracy comparison — horizontal bar
# ─────────────────────────────────────────────────────────────────────────────
section_header('Complete Accuracy Ranking', 'All model configurations on 2,063 test images')

models_sorted = sorted(MODEL_ACCURACY.items(), key=lambda x: x[1])
fig_rank = go.Figure(go.Bar(
    x=[m[1] for m in models_sorted],
    y=[m[0] for m in models_sorted],
    orientation='h',
    marker_color=[FAMILY_COLORS.get(MODEL_FAMILY.get(m[0], ''), '#95a5a6') for m in models_sorted],
    text=[f'{m[1]:.2f}%' for m in models_sorted],
    textposition='outside',
))
fig_rank.update_layout(
    xaxis=dict(range=[60, 104], title='Test Accuracy (%)'),
    margin=dict(l=10, r=80, t=10, b=10),
    height=340,
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
)
fig_rank.add_vline(x=98.20, line_dash='dash', line_color='#f39c12',
                   annotation_text='98.20%', annotation_position='top right')
st.plotly_chart(fig_rank, use_container_width=True)

leg_cols = st.columns(len(FAMILY_COLORS))
for col, (family, color) in zip(leg_cols, FAMILY_COLORS.items()):
    col.markdown(
        f'<div style="display:flex;align-items:center;gap:.4rem;font-size:.8rem">'
        f'<div style="width:14px;height:14px;border-radius:3px;background:{color};'
        f'flex-shrink:0"></div>{family}</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    '⚠ **Research Demonstration Only.** This system is built for academic purposes '
    'at ESOFT Metro Campus. It must not be used for clinical diagnosis.'
)
