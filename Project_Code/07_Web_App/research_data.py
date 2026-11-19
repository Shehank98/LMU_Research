"""
All pre-computed research results from CLAUDE.md.
Every chart and table on the dashboard works without model files.
"""

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_COLORS = {
    'Glioma':     '#e74c3c',
    'Meningioma': '#e67e22',
    'No Tumor':   '#27ae60',
    'Pituitary':  '#2980b9',
}

# ── Overall model accuracy ────────────────────────────────────────────────────
MODEL_ACCURACY = {
    'Final Ensemble (6-model)':       98.20,
    'CNN Standalone':                 91.18,
    'Xception Standalone':            86.91,
    'Traditional ML (raw pixels)':    83.00,
    'CNN + Classical Ensemble':       83.71,
    'InceptionV3 Standalone':         84.63,
    'InceptionV3 + Classical Ens.':   71.20,
    'Xception + Classical Ens.':      70.28,
}

# Model families for grouped chart coloring
MODEL_FAMILY = {
    'Final Ensemble (6-model)':       'Final',
    'CNN Standalone':                 'CNN',
    'CNN + Classical Ensemble':       'CNN',
    'InceptionV3 Standalone':         'InceptionV3',
    'InceptionV3 + Classical Ens.':   'InceptionV3',
    'Xception Standalone':            'Xception',
    'Xception + Classical Ens.':      'Xception',
    'Traditional ML (raw pixels)':    'Classical',
}

FAMILY_COLORS = {
    'Final':       '#f39c12',
    'CNN':         '#8e44ad',
    'InceptionV3': '#2980b9',
    'Xception':    '#16a085',
    'Classical':   '#7f8c8d',
}

# ── Classical ML on raw pixels ────────────────────────────────────────────────
RAW_PIXEL_ACCURACY = {
    'SVM':               88.26,
    'Logistic Regression': 84.00,
    'Decision Tree':      80.02,
    'Random Forest':      79.52,
    'Ensemble (DT+RF+SVM)': 83.00,
}

# ── Classical ML by feature source ───────────────────────────────────────────
FEATURE_ML_ACCURACY = {
    'CNN features':          {'RF': 85.45, 'DT': 80.99, 'SVM': 66.60, 'Ensemble': 83.71},
    'InceptionV3 features':  {'RF': 74.64, 'DT': 68.05, 'SVM': 57.34, 'Ensemble': 71.20},
    'Xception features':     {'RF': 73.68, 'DT': 68.20, 'SVM': 56.71, 'Ensemble': 70.28},
    'Raw pixels':            {'RF': 79.52, 'DT': 80.02, 'SVM': 88.26, 'Ensemble': 83.00},
}

# ── Confusion matrices (rows=actual, cols=predicted) ─────────────────────────
# Order: Glioma=0, Meningioma=1, No Tumor=2, Pituitary=3
CONFUSION_MATRICES = {
    'Final Ensemble': [
        [1688, 34,  6,   5],
        [42,   1636, 8,  13],
        [5,    5,  1837,  0],
        [4,    4,    0, 1698],
    ],
    'CNN + Classical Ens.': [
        [284, 136, 29, 55],
        [40,  442, 16, 12],
        [0,     4, 534,  7],
        [14,   17,   6, 467],
    ],
    'Xception + Classical Ens.': [
        [226, 127, 42, 109],
        [69,  368, 25,  48],
        [16,   19, 492,  18],
        [58,   56,  26, 364],
    ],
    'InceptionV3 + Classical Ens.': [
        [225, 148, 46,  85],
        [48,  376, 41,  45],
        [15,   40, 468,  22],
        [43,   32,  29, 400],
    ],
}

# ── Per-class classification reports ─────────────────────────────────────────
# precision, recall, f1 per class
CLASSIFICATION_REPORTS = {
    'Final Ensemble': {
        'Glioma':     {'precision': 0.97, 'recall': 0.97, 'f1': 0.97},
        'Meningioma': {'precision': 0.97, 'recall': 0.96, 'f1': 0.97},
        'No Tumor':   {'precision': 0.99, 'recall': 0.99, 'f1': 0.99},
        'Pituitary':  {'precision': 0.99, 'recall': 1.00, 'f1': 0.99},
        'accuracy':   0.982,
    },
    'CNN + Classical Ens.': {
        'Glioma':     {'precision': 0.84, 'recall': 0.56, 'f1': 0.67},
        'Meningioma': {'precision': 0.74, 'recall': 0.87, 'f1': 0.80},
        'No Tumor':   {'precision': 0.91, 'recall': 0.98, 'f1': 0.95},
        'Pituitary':  {'precision': 0.86, 'recall': 0.93, 'f1': 0.89},
        'accuracy':   0.8371,
    },
    'InceptionV3 + Classical Ens.': {
        'Glioma':     {'precision': 0.68, 'recall': 0.45, 'f1': 0.54},
        'Meningioma': {'precision': 0.63, 'recall': 0.74, 'f1': 0.68},
        'No Tumor':   {'precision': 0.80, 'recall': 0.86, 'f1': 0.83},
        'Pituitary':  {'precision': 0.72, 'recall': 0.79, 'f1': 0.76},
        'accuracy':   0.7120,
    },
    'Xception + Classical Ens.': {
        'Glioma':     {'precision': 0.75, 'recall': 0.53, 'f1': 0.62},
        'Meningioma': {'precision': 0.70, 'recall': 0.81, 'f1': 0.75},
        'No Tumor':   {'precision': 0.92, 'recall': 0.99, 'f1': 0.95},
        'Pituitary':  {'precision': 0.81, 'recall': 0.87, 'f1': 0.84},
        'accuracy':   0.7028,
    },
}

# ── Dataset statistics ────────────────────────────────────────────────────────
DATASET = {
    'Training': {'Glioma': 1733, 'Meningioma': 1699, 'No Tumor': 1847, 'Pituitary': 1705},
    'Testing':  {'Glioma': 504,  'Meningioma': 510,  'No Tumor': 545,  'Pituitary': 504},
}

# ── Ensemble build-up (accuracy as models are added 1→6) ─────────────────────
ENSEMBLE_BUILDUP = [
    {'models': 'CNN only',                             'accuracy': 91.18},
    {'models': '+ InceptionV3',                        'accuracy': 93.10},
    {'models': '+ Xception',                           'accuracy': 94.50},
    {'models': '+ CNN + Classical Ens.',               'accuracy': 96.20},
    {'models': '+ InceptionV3 + Classical Ens.',       'accuracy': 97.40},
    {'models': '+ Xception + Classical Ens. (final)',  'accuracy': 98.20},
]

# ── Literature comparison ─────────────────────────────────────────────────────
LITERATURE = [
    {
        'Citation': 'Kakon et al. Cancers 2025 (PMC12427295)',
        'Architecture': 'EfficientNetB7 + InceptionV3 + Xception',
        'Voting': 'Soft vote + LightGBM meta',
        'Per-model Grad-CAM': '✓',
        'Heatmap fusion': '✗',
        'Inter-model IoU': '✗',
        'Accuracy': '98.5%',
    },
    {
        'Citation': 'Advanced Dynamic Ensemble Sci Rep 2025',
        'Architecture': 'CNN + ResNet-50 + EfficientNet-B5',
        'Voting': 'Adaptive weights',
        'Per-model Grad-CAM': '✓ (+ SHAP + LIME)',
        'Heatmap fusion': '✗',
        'Inter-model IoU': '✗',
        'Accuracy': '99.1%',
    },
    {
        'Citation': 'Majority Voting MDPI Diagnostics 2025',
        'Architecture': '14 CNNs',
        'Voting': 'Majority vote',
        'Per-model Grad-CAM': '✓',
        'Heatmap fusion': '✗',
        'Inter-model IoU': '✗',
        'Accuracy': '99.8%',
    },
    {
        'Citation': 'TSO-Optimised Voting Informatica 2024',
        'Architecture': 'InceptionV3 + Xception',
        'Voting': 'Weighted soft vote (TSO)',
        'Per-model Grad-CAM': '✗',
        'Heatmap fusion': '✗',
        'Inter-model IoU': '✗',
        'Accuracy': '99.92%',
    },
    {
        'Citation': '**This Work** (Kavishka 2024)',
        'Architecture': 'CNN + InceptionV3 + Xception + 3×Classical ML',
        'Voting': 'Confidence-weighted',
        'Per-model Grad-CAM': '✓',
        'Heatmap fusion': '✓ (EAA-IoU)',
        'Inter-model IoU': '✓ (Novel)',
        'Accuracy': '98.20%',
    },
]

# ── Approximate model complexity ─────────────────────────────────────────────
MODEL_PARAMS = {
    'CNN Standalone':        0.90,   # million parameters (approx)
    'InceptionV3 Standalone': 21.8,
    'Xception Standalone':   22.9,
    'CNN + Classical Ens.':   0.90,
    'InceptionV3 + Classical Ens.': 21.8,
    'Xception + Classical Ens.': 22.9,
}
