"""
NeuroScan AI — Brain Tumor MRI Classifier
Home page: research overview, KPIs, accuracy overview, dataset stats.
All charts render from pre-computed data — no model files required.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from ui_components import inject_css, kpi_card, section_header
from research_data import (
    MODEL_ACCURACY, MODEL_FAMILY, FAMILY_COLORS,
    DATASET, CLASS_COLORS,
    ENSEMBLE_BUILDUP,
)

st.set_page_config(
    page_title='NeuroScan AI · Brain Tumor Classifier',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)
inject_css()

# ── Sidebar branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h2 style="color:#00b4d8;margin:0">🧠 NeuroScan AI</h2>'
        '<p style="color:#a0b4c8;font-size:.8rem;margin:0">Brain Tumor MRI Classifier</p>',
        unsafe_allow_html=True,
    )
    st.markdown('---')
    st.markdown(
        '<p style="color:#a0b4c8;font-size:.75rem">'
        'Ensemble Consensus XAI System<br>'
        'CNN · Xception · InceptionV3 · Classical ML<br><br>'
        '<b style="color:#e0eaf4">Shehan Kavishka · E187041</b><br>'
        'ESOFT Metro Campus · HND Computing<br>'
        'Research Project 2024</p>',
        unsafe_allow_html=True,
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="color:#0d1b2a;margin-bottom:.2rem">🧠 NeuroScan AI</h1>'
    '<p style="color:#6b7c93;font-size:1.05rem;margin-bottom:1.5rem">'
    'Ensemble Consensus XAI System for Brain Tumor Classification from MRI'
    '</p>',
    unsafe_allow_html=True,
)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
cards = [
    (k1, '98.20%', 'Final Ensemble Accuracy', '', ''),
    (k2, '9,047',  'MRI Scans',               'Kaggle dataset', ''),
    (k3, '4',      'Tumor Classes',            '', ''),
    (k4, '6',      'Combined Models',          '', 'accent'),
    (k5, '0.98',   'Weighted F1-Score',        '', 'success'),
]
for col, val, label, sub, variant in cards:
    col.markdown(kpi_card(label, val, sub, variant), unsafe_allow_html=True)

st.markdown('')

# ── Abstract ─────────────────────────────────────────────────────────────────
with st.expander('📄 Research Abstract', expanded=False):
    st.markdown(
        '<div class="clinical-card">'
        '<b>Title:</b> A Comparative Analysis of Ensemble Learning, Fine-Tuned Models '
        'and CNNs for Multi-Class Disease Prediction in Medical Imaging: An Integrated '
        'Framework with Systematic Optimization.<br><br>'
        'This study examines how well fine-tuned models perform in feature extraction '
        'compared to ordinary CNNs and how their integration with ensemble techniques '
        'affects the precision and dependability of identifying brain tumors from MRI '
        'images. The purpose is to assess the performance of SVM, RF, DT, Xception, '
        'and InceptionV3 classifiers using a mixed-methods approach. '
        'The final ensemble model, composed of six saved models, outperformed the CNN\'s '
        'highest individual accuracy of 94% with a final accuracy of <b>98.20%</b>.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Two-column: objectives + accuracy table ───────────────────────────────────
col_obj, col_tbl = st.columns([1, 1])

with col_obj:
    section_header('Research Objectives', 'Mixed-methods, deductive approach')
    objectives = [
        ('RO1', 'Evaluate CNN feature extraction for brain tumor identification'),
        ('RO2', 'Assess RF/DT/SVM performance on CNN-extracted features'),
        ('RO3', 'Investigate Xception & InceptionV3 for feature extraction'),
        ('RO4', 'Evaluate ML classifiers on pre-trained model features'),
        ('RO5', 'Assess accuracy gains from ensemble combination'),
        ('RO6', 'Compare CNN-based vs fine-tuned feature extraction'),
    ]
    for code, text in objectives:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;margin-bottom:.5rem;">'
            f'<span style="background:#00b4d8;color:white;padding:.1rem .5rem;'
            f'border-radius:4px;font-size:.72rem;font-weight:700;'
            f'margin-right:.6rem;flex-shrink:0;margin-top:.1rem">{code}</span>'
            f'<span style="font-size:.88rem;color:#2c3e50">{text}</span></div>',
            unsafe_allow_html=True,
        )

with col_tbl:
    section_header('Accuracy Summary', 'All models on official test set')
    rows = [
        {'Model': k, 'Accuracy (%)': f'{v:.2f}'}
        for k, v in sorted(MODEL_ACCURACY.items(), key=lambda x: -x[1])
    ]
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ── Accuracy bar chart ────────────────────────────────────────────────────────
section_header('Model Accuracy Comparison', 'All configurations on 2,063 test images')

models_sorted = sorted(MODEL_ACCURACY.items(), key=lambda x: x[1])
labels  = [m[0] for m in models_sorted]
values  = [m[1] for m in models_sorted]
colours = [FAMILY_COLORS.get(MODEL_FAMILY.get(l, ''), '#95a5a6') for l in labels]

fig_acc = go.Figure(go.Bar(
    x=values, y=labels, orientation='h',
    marker_color=colours,
    text=[f'{v:.2f}%' for v in values],
    textposition='outside',
))
fig_acc.update_layout(
    xaxis=dict(range=[60, 104], title='Test Accuracy (%)'),
    margin=dict(l=10, r=70, t=10, b=10),
    height=380,
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
)
fig_acc.add_vline(x=98.20, line_dash='dash', line_color='#f39c12',
                  annotation_text='Final 98.20%',
                  annotation_position='top right')
st.plotly_chart(fig_acc, use_container_width=True)

# Colour legend
leg_cols = st.columns(len(FAMILY_COLORS))
for col, (family, color) in zip(leg_cols, FAMILY_COLORS.items()):
    col.markdown(
        f'<div style="display:flex;align-items:center;gap:.4rem;font-size:.8rem">'
        f'<div style="width:14px;height:14px;border-radius:3px;background:{color};'
        f'flex-shrink:0"></div>{family}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Ensemble build-up line chart ──────────────────────────────────────────────
section_header('Ensemble Build-Up', 'Accuracy gain as models are progressively combined')

fig_build = go.Figure()
fig_build.add_trace(go.Scatter(
    x=list(range(1, len(ENSEMBLE_BUILDUP) + 1)),
    y=[e['accuracy'] for e in ENSEMBLE_BUILDUP],
    mode='lines+markers+text',
    text=[f"{e['accuracy']}%" for e in ENSEMBLE_BUILDUP],
    textposition='top center',
    line=dict(color='#00b4d8', width=3),
    marker=dict(size=10, color='#00b4d8'),
    hovertext=[e['models'] for e in ENSEMBLE_BUILDUP],
    hoverinfo='text+y',
))
fig_build.update_layout(
    xaxis=dict(
        tickvals=list(range(1, 7)),
        ticktext=[e['models'].split('(')[0].strip()[:25] for e in ENSEMBLE_BUILDUP],
        tickangle=-30,
    ),
    yaxis=dict(range=[85, 100], title='Accuracy (%)'),
    margin=dict(l=10, r=10, t=20, b=90),
    height=340,
    plot_bgcolor='white',
    paper_bgcolor='white',
)
st.plotly_chart(fig_build, use_container_width=True)

st.divider()

# ── Dataset distribution ──────────────────────────────────────────────────────
section_header('Dataset Distribution', 'Kaggle Brain Tumor MRI Dataset — 9,047 scans')

d_col1, d_col2 = st.columns(2)
for col, split in zip([d_col1, d_col2], ['Training', 'Testing']):
    counts = DATASET[split]
    fig_d = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        marker_colors=[CLASS_COLORS[c] for c in counts.keys()],
        hole=0.45,
        textinfo='label+percent',
    ))
    fig_d.update_layout(
        title_text=f'{split} Set — {sum(counts.values()):,} images',
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        paper_bgcolor='white',
        showlegend=False,
    )
    col.plotly_chart(fig_d, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#a0aab4;font-size:.78rem;padding:1.5rem 0">'
    '⚠ <b>Research Demonstration Only.</b> This system is built for academic purposes '
    'as part of an HND Computing Research Project at ESOFT Metro Campus. '
    'It must not be used for clinical diagnosis. '
    'Always consult a qualified radiologist for medical decisions.'
    '</div>',
    unsafe_allow_html=True,
)
