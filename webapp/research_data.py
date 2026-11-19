"""
Pre-computed research results shown on the dashboard pages (no model files needed).
Every value comes from the logged result files of the training notebook:
metrics/results_master.csv, metrics/*_classification_report.json and confusion/*.csv
(1,600 held-out test images, 400 per class). ENSEMBLE_BUILDUP sums the probability vectors
of the predictors added so far (metrics/final_probs.npz).
"""

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_COLORS = {'Glioma': '#e74c3c', 'Meningioma': '#e67e22', 'No Tumor': '#27ae60', 'Pituitary': '#2980b9'}

FAMILY_COLORS = {'Final': '#f39c12', 'CNN': '#8e44ad', 'InceptionV3': '#2980b9', 'Xception': '#16a085', 'Classical': '#7f8c8d'}

FINAL_ACCURACY = 92.5          # six-model majority vote
BEST_SINGLE_ACCURACY = 90.44    # custom CNN
EAA_SUMMARY = {'mean_all': 0.0985, 'mean_correct': 0.1029, 'mean_misclassified': 0.0471, 'auroc': 0.7525,
               'confidence_auroc': 0.9127, 'flagged_pct_below_0.10': 50.81, 'errors_caught_pct_below_0.10': 82.26}

MODEL_ACCURACY = {'Final Ensemble (6-model)': 92.5,
 'Confidence-Weighted Ensemble': 92.25,
 'CNN Standalone': 90.44,
 'Xception Standalone': 87.62,
 'InceptionV3 Standalone': 86.62,
 'CNN + Classical Ensemble': 89.31,
 'Xception + Classical Ens.': 88.31,
 'InceptionV3 + Classical Ens.': 86.12,
 'Traditional ML (raw pixels)': 84.06}

MODEL_FAMILY = {'Final Ensemble (6-model)': 'Final',
 'Confidence-Weighted Ensemble': 'Final',
 'CNN Standalone': 'CNN',
 'CNN + Classical Ensemble': 'CNN',
 'InceptionV3 Standalone': 'InceptionV3',
 'InceptionV3 + Classical Ens.': 'InceptionV3',
 'Xception Standalone': 'Xception',
 'Xception + Classical Ens.': 'Xception',
 'Traditional ML (raw pixels)': 'Classical'}

RAW_PIXEL_ACCURACY = {'Random Forest': 87.0,
 'SVM': 82.25,
 'Logistic Regression': 79.87,
 'Decision Tree': 77.88,
 'Ensemble (DT+RF+SVM)': 84.06}

FEATURE_ML_ACCURACY = {'CNN features': {'RF': 89.94,
                  'DT': 86.5,
                  'SVM': 89.69,
                  'Ensemble': 89.31},
 'InceptionV3 features': {'RF': 86.75,
                          'DT': 83.69,
                          'SVM': 85.38,
                          'Ensemble': 86.12},
 'Xception features': {'RF': 88.38,
                       'DT': 86.0,
                       'SVM': 87.69,
                       'Ensemble': 88.31},
 'Raw pixels': {'RF': 87.0,
                'DT': 77.88,
                'SVM': 82.25,
                'Ensemble': 84.06}}

CONFUSION_MATRICES = {'Final Ensemble': [[310, 65, 23, 2], [6, 382, 2, 10], [0, 0, 400, 0], [2, 10, 0, 388]],
 'CNN + Classical Ens.': [[309, 46, 37, 8], [18, 353, 14, 15], [1, 1, 398, 0], [8, 23, 0, 369]],
 'Xception + Classical Ens.': [[290, 73, 30, 7], [17, 359, 3, 21], [1, 3, 396, 0], [6, 25, 1, 368]],
 'InceptionV3 + Classical Ens.': [[275, 86, 26, 13], [33, 342, 7, 18], [6, 2, 392, 0], [5, 18, 8, 369]]}

CLASSIFICATION_REPORTS = {'Final Ensemble': {'Glioma': {'precision': 0.975, 'recall': 0.775, 'f1': 0.864},
                    'Meningioma': {'precision': 0.836, 'recall': 0.955, 'f1': 0.891},
                    'No Tumor': {'precision': 0.941, 'recall': 1.0, 'f1': 0.97},
                    'Pituitary': {'precision': 0.97, 'recall': 0.97, 'f1': 0.97},
                    'accuracy': 0.925},
 'CNN + Classical Ens.': {'Glioma': {'precision': 0.92, 'recall': 0.772, 'f1': 0.84},
                          'Meningioma': {'precision': 0.835, 'recall': 0.882, 'f1': 0.858},
                          'No Tumor': {'precision': 0.886, 'recall': 0.995, 'f1': 0.938},
                          'Pituitary': {'precision': 0.941, 'recall': 0.922, 'f1': 0.932},
                          'accuracy': 0.8931},
 'InceptionV3 + Classical Ens.': {'Glioma': {'precision': 0.862, 'recall': 0.688, 'f1': 0.765},
                                  'Meningioma': {'precision': 0.763, 'recall': 0.855, 'f1': 0.807},
                                  'No Tumor': {'precision': 0.905, 'recall': 0.98, 'f1': 0.941},
                                  'Pituitary': {'precision': 0.922, 'recall': 0.922, 'f1': 0.922},
                                  'accuracy': 0.8612},
 'Xception + Classical Ens.': {'Glioma': {'precision': 0.924, 'recall': 0.725, 'f1': 0.812},
                               'Meningioma': {'precision': 0.78, 'recall': 0.897, 'f1': 0.835},
                               'No Tumor': {'precision': 0.921, 'recall': 0.99, 'f1': 0.954},
                               'Pituitary': {'precision': 0.929, 'recall': 0.92, 'f1': 0.925},
                               'accuracy': 0.8831}}

DATASET = {'Training': {'Glioma': 1400, 'Meningioma': 1400, 'No Tumor': 1400, 'Pituitary': 1400},
 'Testing': {'Glioma': 400, 'Meningioma': 400, 'No Tumor': 400, 'Pituitary': 400}}

ENSEMBLE_BUILDUP = [{'models': 'CNN only', 'accuracy': 90.5},
 {'models': '+ InceptionV3', 'accuracy': 91.38},
 {'models': '+ Xception', 'accuracy': 91.12},
 {'models': '+ CNN + Classical Ens.', 'accuracy': 91.94},
 {'models': '+ InceptionV3 + Classical Ens.', 'accuracy': 92.38},
 {'models': '+ Xception + Classical Ens. (all six)', 'accuracy': 92.25}]

LITERATURE = [{'Citation': 'Cheng et al. (2015)',
  'Architecture': 'Region augmentation + hand-crafted features',
  'Voting': 'None',
  'Per-model Grad-CAM': '✗',
  'Heatmap fusion': '✗',
  'Inter-model IoU': '✗',
  'Accuracy': '91.28%'},
 {'Citation': 'Zhou et al. (2019)',
  'Architecture': 'DenseNet + recurrent network',
  'Voting': 'None',
  'Per-model Grad-CAM': '✗',
  'Heatmap fusion': '✗',
  'Inter-model IoU': '✗',
  'Accuracy': '92.13%'},
 {'Citation': 'Deepak and Ameer (2019)',
  'Architecture': 'GoogLeNet features + SVM / KNN',
  'Voting': 'None',
  'Per-model Grad-CAM': '✗',
  'Heatmap fusion': '✗',
  'Inter-model IoU': '✗',
  'Accuracy': '98% (5-fold CV)'},
 {'Citation': 'Rehman et al. (2020)',
  'Architecture': 'Fine-tuned VGG16',
  'Voting': 'None',
  'Per-model Grad-CAM': '✗',
  'Heatmap fusion': '✗',
  'Inter-model IoU': '✗',
  'Accuracy': '98.69%'},
 {'Citation': 'Kang, Ullah and Gwak (2021)',
  'Architecture': 'Deep features from several CNNs + classical ML',
  'Voting': 'Feature combination',
  'Per-model Grad-CAM': '✗',
  'Heatmap fusion': '✗',
  'Inter-model IoU': '✗',
  'Accuracy': 'see paper'},
 {'Citation': '**This work** (Kavishka, 2026)',
  'Architecture': 'CNN + InceptionV3 + Xception + 3 × classical soft vote',
  'Voting': 'Majority and confidence-weighted',
  'Per-model Grad-CAM': '✓',
  'Heatmap fusion': '✓ (consensus map)',
  'Inter-model IoU': '✓ (EAA-IoU)',
  'Accuracy': '92.50%'}]

# million parameters (InceptionV3 / Xception: only 8,196 are trainable, the backbone is frozen)
MODEL_PARAMS = {'CNN Standalone': 3.31,
 'InceptionV3 Standalone': 21.81,
 'Xception Standalone': 20.87,
 'CNN + Classical Ens.': 3.31,
 'InceptionV3 + Classical Ens.': 21.81,
 'Xception + Classical Ens.': 20.87}

