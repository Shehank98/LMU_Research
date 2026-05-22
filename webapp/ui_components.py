"""
Shared clinical CSS theme and card helper functions.
Call inject_css() once at the top of every page.
"""
import streamlit as st


CLINICAL_CSS = """
<style>
/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a2e44 100%);
}
[data-testid="stSidebar"] * {
    color: #e0eaf4 !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(0,180,216,0.15) !important;
    border-left: 3px solid #00b4d8 !important;
    border-radius: 6px;
}

/* ── Main background ─────────────────────────────── */
.main .block-container { padding-top: 1.5rem; }
.stApp { background-color: #f0f4f8; }

/* ── KPI cards ───────────────────────────────────── */
.kpi-card {
    background: white;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
    border-top: 4px solid #00b4d8;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
    margin-bottom: .5rem;
}
.kpi-card .kpi-val {
    font-size: 2rem;
    font-weight: 800;
    color: #0d1b2a;
    line-height: 1.1;
}
.kpi-card .kpi-label {
    font-size: .78rem;
    color: #6b7c93;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-top: .3rem;
}
.kpi-card.accent  { border-top-color: #f39c12; }
.kpi-card.success { border-top-color: #27ae60; }
.kpi-card.danger  { border-top-color: #e74c3c; }

/* ── Section header ──────────────────────────────── */
.section-header {
    border-left: 4px solid #00b4d8;
    padding: .3rem 0 .3rem .8rem;
    margin-bottom: 1rem;
}
.section-header h3 { margin: 0; color: #0d1b2a; font-weight: 700; }
.section-header p  { margin: 0; color: #6b7c93; font-size: .85rem; }

/* ── Info / disclaimer card ──────────────────────── */
.clinical-card {
    background: white;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    margin-bottom: 1rem;
}

/* ── Confidence badge ────────────────────────────── */
.badge-high   { background:#27ae60;color:white;padding:.2rem .7rem;border-radius:20px;font-size:.78rem;font-weight:700; }
.badge-mod    { background:#f39c12;color:white;padding:.2rem .7rem;border-radius:20px;font-size:.78rem;font-weight:700; }
.badge-low    { background:#e74c3c;color:white;padding:.2rem .7rem;border-radius:20px;font-size:.78rem;font-weight:700; }

/* ── Prediction big label ────────────────────────── */
.pred-class {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.1;
}
.pred-conf { color:#6b7c93; font-size:.9rem; margin-top:.25rem; }

/* ── Model status dot ────────────────────────────── */
.status-ok   { color: #27ae60; font-weight:700; }
.status-miss { color: #e74c3c; font-weight:700; }
.status-warn { color: #f39c12; font-weight:700; }

/* ── Plotly chart background ─────────────────────── */
.js-plotly-plot .plotly { border-radius: 8px; }

/* ── Table styling ───────────────────────────────── */
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
"""


def inject_css():
    st.markdown(CLINICAL_CSS, unsafe_allow_html=True)


def kpi_card(title: str, value: str, subtitle: str = '', variant: str = '') -> str:
    cls = f'kpi-card {variant}'.strip()
    sub = f'<div class="kpi-label">{subtitle}</div>' if subtitle else ''
    return f"""
    <div class="{cls}">
      <div class="kpi-val">{value}</div>
      <div class="kpi-label">{title}</div>
      {sub}
    </div>"""


def section_header(title: str, subtitle: str = '') -> None:
    sub = f'<p>{subtitle}</p>' if subtitle else ''
    st.markdown(
        f'<div class="section-header"><h3>{title}</h3>{sub}</div>',
        unsafe_allow_html=True,
    )


def confidence_badge(conf: float, threshold: float = 0.70) -> str:
    if conf >= threshold:
        return f'<span class="badge-high">HIGH CONFIDENCE {conf*100:.1f}%</span>'
    elif conf >= 0.50:
        return f'<span class="badge-mod">MODERATE {conf*100:.1f}%</span>'
    else:
        return f'<span class="badge-low">LOW CONFIDENCE {conf*100:.1f}%</span>'


def model_status_sidebar(models: dict) -> None:
    """Render per-model status dots in sidebar."""
    labels = {
        'cnn':       'CNN Standalone',
        'inception': 'InceptionV3',
        'xception':  'Xception',
        'cnn_feat':  'CNN Feature Extractor',
        'inc_feat':  'Inc Feature Extractor',
        'xcp_feat':  'Xcp Feature Extractor',
        'cnn_ens':   'CNN Classical Ens.',
        'inc_ens':   'Inc Classical Ens.',
        'xcp_ens':   'Xcp Classical Ens.',
    }
    loaded = sum(1 for v in models.values() if v is not None)
    total  = len(labels)

    if loaded == total:
        st.sidebar.markdown('<p class="status-ok">● All models loaded</p>',
                            unsafe_allow_html=True)
    elif loaded == 0:
        st.sidebar.markdown('<p class="status-miss">● No models — demo mode</p>',
                            unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<p class="status-warn">● {loaded}/{total} models</p>',
                            unsafe_allow_html=True)

    with st.sidebar.expander('Model details'):
        for key, label in labels.items():
            ok = models.get(key) is not None
            dot = '<span class="status-ok">●</span>' if ok else '<span class="status-miss">●</span>'
            st.markdown(f'{dot} {label}', unsafe_allow_html=True)
