#!/usr/bin/env python3
# Builds one Colab notebook per model from the tested all-in-one notebook, so the code is identical.
import ast, json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SRC = f'{ROOT}/Brain_Tumor_Train_And_Log.ipynb'
OUT = f'{ROOT}/individual_models'
os.makedirs(OUT, exist_ok=True)

code_cells = [c['source'] for c in json.load(open(SRC))['cells'] if c['cell_type'] == 'code']
def cell(prefix):
    hits = [c for c in code_cells if c.lstrip().startswith(prefix)]
    assert len(hits) == 1, prefix
    return hits[0]

C_MOUNT = cell('# 1.'); C_CONFIG = cell('# 2.'); C_LOG = cell('# 3.'); C_METRICS = cell('# 4.')
C_RESUME = cell('# 5.'); C_DATA = cell('# 6.'); C_BUILD = cell('# 7.'); C_HELP = cell('# 8.'); C_XAI = cell('# 9.')

def pick(src, names):
    # source of the named top-level functions / assignments, in their original order
    tree, out = ast.parse(src), []
    for node in tree.body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            out.append(ast.get_source_segment(src, node))
    missing = set(names) - {n for n in names if any(f'def {n}(' in s or s.startswith(n + ' ') for s in out)}
    assert not missing, missing
    return '\n\n'.join(out)

SHARED_HELPERS = ('# 7. Shared prediction, feature-extraction and classical-ML helpers (same in every notebook)\n'
                  + pick(C_HELP, ['predict_ds', 'extract_features', 'train_classical']))

def nb(cells):
    return {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []},
                         'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                         'language_info': {'name': 'python'}}}
md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s}
code = lambda s: {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s}

def show_results(prefixes, history=None, confusion=None):
    lines = ['# Results for this notebook (read back from Google Drive)',
             'if not is_done(STAGE):',
             '    raise SystemExit("This stage has not finished yet. Re-run the cell above; it resumes where it stopped.")',
             'import pandas as pd',
             'df = pd.read_csv(MASTER_CSV).drop_duplicates("model", keep="last")',
             f'mine = df[df.model.str.startswith({tuple(prefixes)!r})]',
             'print(mine[["model","accuracy","precision_macro","recall_macro","f1_macro","train_time_sec"]].to_string(index=False))']
    if confusion:
        lines += [f'print("\\nConfusion matrix ({confusion}):")',
                  f'print(pd.read_csv(os.path.join(CONF_DIR, "{confusion}_confusion_matrix.csv")).to_string(index=False))']
    if history:
        lines += [f'print("\\nTraining history ({history}), last epochs:")',
                  f'print(pd.read_csv(os.path.join(HISTORY_DIR, "{history}_history.csv")).tail().to_string(index=False))']
    return '\n'.join(lines)

def require(files, hint):
    return ('# Check prerequisites: the earlier notebooks must have produced these files\n'
            f'REQUIRED = {files!r}\n'
            'missing = [f for f in REQUIRED if not os.path.exists(os.path.join(OUTPUT_DIR, f))]\n'
            'if missing:\n'
            f'    raise RuntimeError("Missing: " + ", ".join(missing) + "\\nRun these notebooks first: {hint}")\n'
            'print("All prerequisites found.")')

SPECS = [
  dict(file='01_CNN.ipynb', stage='CNN_standalone', run='run_cnn', time='10-20 min',
       title='01 · Custom CNN (standalone classifier)',
       what='Trains the custom four-block CNN from scratch on 256 x 256 images (20 epochs, Adam) and evaluates it on the official test set.',
       model=['build_cnn'], runf=['run_cnn'], needs=None,
       outputs=['models/cnn_model.h5', 'history/cnn_model_history.csv', 'metrics/CNN_standalone_classification_report.json',
                'confusion/CNN_standalone_confusion_matrix.csv', 'row `CNN_standalone` in metrics/results_master.csv'],
       show=(['CNN_standalone'], 'cnn_model', 'CNN_standalone')),
  dict(file='02_InceptionV3.ipynb', stage='InceptionV3_standalone', run='run_inception', time='10-20 min',
       title='02 · InceptionV3 (transfer learning)',
       what='ImageNet-pretrained InceptionV3 with a frozen backbone and a new 4-class head (Global Average Pooling + Dense), 10 epochs at 256 x 256.',
       model=['_base', 'build_transfer_standalone'], runf=['run_inception'], needs=None,
       outputs=['models/inceptionv3_model.h5', 'history/inceptionv3_model_history.csv',
                'metrics/InceptionV3_standalone_classification_report.json', 'confusion/InceptionV3_standalone_confusion_matrix.csv'],
       show=(['InceptionV3_standalone'], 'inceptionv3_model', 'InceptionV3_standalone')),
  dict(file='03_Xception.ipynb', stage='Xception_standalone', run='run_xception', time='15-25 min',
       title='03 · Xception (transfer learning)',
       what='ImageNet-pretrained Xception with a frozen backbone and a new 4-class head, 10 epochs at 256 x 256.',
       model=['_base', 'build_transfer_standalone'], runf=['run_xception'], needs=None,
       outputs=['models/xception_model.h5', 'history/xception_model_history.csv',
                'metrics/Xception_standalone_classification_report.json', 'confusion/Xception_standalone_confusion_matrix.csv'],
       show=(['Xception_standalone'], 'xception_model', 'Xception_standalone')),
  dict(file='04_CNN_Feature_Ensemble.ipynb', stage='CNN_feature_ensemble', run='run_cnn_ensemble', time='10-20 min',
       title='04 · CNN feature extractor + classical ensemble',
       what='Trains a 128 x 128 CNN, reads its 256-value Dense layer (`dense`) as a feature vector for every image, then trains Random Forest, Decision Tree and SVM on those features and combines them by soft voting.',
       model=['build_cnn'], runf=['_run_feature_ensemble', 'run_cnn_ensemble'], needs=None,
       outputs=['models/cnn_ext.h5 (feature extractor)', 'models/cnn_feat_rf.pkl, cnn_feat_dt.pkl, cnn_feat_svm.pkl',
                'history/cnn_ext_history.csv', 'metrics + confusion files for cnn_feat_rf / _dt / _svm / _softvote_ensemble'],
       show=(['cnn_feat_'], 'cnn_ext', 'cnn_feat_softvote_ensemble')),
  dict(file='05_InceptionV3_Feature_Ensemble.ipynb', stage='InceptionV3_feature_ensemble', run='run_inception_ensemble', time='10-20 min',
       title='05 · InceptionV3 feature extractor + classical ensemble',
       what='Frozen InceptionV3 at 128 x 128 with a 256-value Dense layer (`dense_2`) used as the feature vector, then Random Forest, Decision Tree and SVM with soft voting.',
       model=['_base', 'build_transfer_extractor'], runf=['_run_feature_ensemble', 'run_inception_ensemble'], needs=None,
       outputs=['models/inceptionv3_ext.h5', 'models/inceptionv3_feat_rf.pkl, _dt.pkl, _svm.pkl', 'history/inceptionv3_ext_history.csv',
                'metrics + confusion files for inceptionv3_feat_*'],
       show=(['inceptionv3_feat_'], 'inceptionv3_ext', 'inceptionv3_feat_softvote_ensemble')),
  dict(file='06_Xception_Feature_Ensemble.ipynb', stage='Xception_feature_ensemble', run='run_xception_ensemble', time='10-20 min',
       title='06 · Xception feature extractor + classical ensemble',
       what='Frozen Xception at 128 x 128 with a 256-value Dense layer (`dense_3`) used as the feature vector, then Random Forest, Decision Tree and SVM with soft voting.',
       model=['_base', 'build_transfer_extractor'], runf=['_run_feature_ensemble', 'run_xception_ensemble'], needs=None,
       outputs=['models/xception_ext.h5', 'models/xception_feat_rf.pkl, _dt.pkl, _svm.pkl', 'history/xception_ext_history.csv',
                'metrics + confusion files for xception_feat_*'],
       show=(['xception_feat_'], 'xception_ext', 'xception_feat_softvote_ensemble')),
  dict(file='07_Traditional_ML_Raw_Pixels.ipynb', stage='Traditional_raw_pixels', run='run_traditional_raw', time='30-90 min (the SVM is slow)',
       title='07 · Classical machine learning on raw pixels',
       what='Each greyscale 256 x 256 image is flattened to 65,536 pixel values. Random Forest, Decision Tree, SVM and Logistic Regression are trained on them, plus a soft-vote ensemble. Each classifier is saved the moment it finishes.',
       model=[], runf=['run_traditional_raw'], needs=None,
       outputs=['models/raw_rf.pkl, raw_dt.pkl, raw_svm.pkl, raw_lr.pkl', 'metrics + confusion files for raw_*'],
       show=(['raw_'], None, 'raw_softvote_ensemble')),
  dict(file='08_Final_Ensemble.ipynb', stage='FINAL_ensemble', run='run_final_ensemble', time='about 5 min',
       title='08 · Final six-model ensemble',
       what='Loads the three deep models and the three feature ensembles, collects their class probabilities for every test image and combines them two ways: majority vote and confidence-weighted vote (sum of probabilities). Also saves the per-image predictions.',
       model=[], runf=['run_final_ensemble'],
       needs=(['models/cnn_model.h5', 'models/inceptionv3_model.h5', 'models/xception_model.h5', 'models/cnn_ext.h5',
               'models/inceptionv3_ext.h5', 'models/xception_ext.h5', 'models/cnn_feat_svm.pkl',
               'models/inceptionv3_feat_svm.pkl', 'models/xception_feat_svm.pkl'], '01 to 06'),
       outputs=['metrics/final_per_image_predictions.csv', 'metrics/final_probs.npz',
                'metrics + confusion files for FINAL_majority_vote and FINAL_confidence_weighted'],
       show=(['FINAL_'], None, 'FINAL_confidence_weighted')),
  dict(file='09_EAA_IoU_Analysis.ipynb', stage='EAA_IoU_analysis', run='run_eaa_iou', time='10-20 min',
       title='09 · Explainable AI: EAA-IoU agreement analysis',
       what='For every test image, builds attention maps (HiResCAM, a Grad-CAM-family method) from the CNN, Xception and InceptionV3 for the class the ensemble chose, measures the pairwise overlap (IoU) and averages it into the EAA-IoU score. Then tests whether low agreement signals a wrong prediction. Resumable image by image.',
       model=['_GRAD_MODELS', '_last_conv_index', '_last_4d_layer', '_final_dense', '_logits', 'gradcam_batch', 'iou'],
       runf=['run_eaa_iou', 'summarise_eaa'],
       needs=(['models/cnn_model.h5', 'models/xception_model.h5', 'models/inceptionv3_model.h5',
               'metrics/final_per_image_predictions.csv'], '01, 02, 03 and 08'),
       outputs=['metrics/eaa_iou_per_image.csv', 'metrics/eaa_iou_summary.json', 'metrics/eaa_iou_threshold_table.csv'],
       show=None),
  dict(file='10_Upload_to_HuggingFace.ipynb', stage='HF_upload', run='run_hf_upload', time='5-10 min',
       title='10 · Upload models and results to Hugging Face',
       what='Packs each feature ensemble into one file, writes manifest.json, and uploads the files the web app needs (models/) plus every result file (results_2026-10-22/) to shehank98/brain-tumor-mri-models. Needs a Hugging Face WRITE token saved as the Colab secret HF_TOKEN.',
       model=[], runf=['WEBAPP_FILES', '_hf_token', 'run_hf_upload'], needs=None,
       outputs=['Hugging Face: models/*.h5, models/*.pkl, models/manifest.json', 'Hugging Face: results_2026-10-22/...'],
       show=None),
]

for s in SPECS:
    src_model = pick(C_BUILD if s['stage'] not in ('EAA_IoU_analysis',) else C_XAI, s['model']) if s['model'] else ''
    run_src = C_XAI if s['stage'] in ('EAA_IoU_analysis', 'HF_upload') else C_HELP
    src_run = pick(run_src, s['runf'])
    header = (f"# {s['title']}\n**Final code set · 2026-10-22 · individual notebook**\n\n"
              f"**What this notebook does.** {s['what']}\n\n"
              f"**Run first:** {('notebooks ' + s['needs'][1]) if s['needs'] else 'nothing, this notebook is independent'}.  \n"
              f"**Approximate time on a Colab T4 GPU:** {s['time']}.\n\n"
              "**Saves (on Google Drive, in `brain_tumor_results_2026-10-22/`):**\n"
              + '\n'.join(f'- `{o}`' if not o.startswith('row') and not o.startswith('metrics +') and not o.startswith('Hugging') else f'- {o}'
                          for o in s['outputs'])
              + "\n\n**How to run:** `Runtime > Change runtime type > T4 GPU`, then run every cell from top to bottom. "
                "If Colab disconnects, reconnect and run all cells again: training resumes from the last finished epoch and "
                "finished work is skipped.\n\n"
                "Cells 1 to 7 are the same in every notebook (Drive, settings, logging, data loading). "
                "The cells after them are specific to this model.")
    cells = [md(header), code(C_MOUNT), code(C_CONFIG), code(C_LOG), code(C_METRICS), code(C_RESUME), code(C_DATA),
             code(SHARED_HELPERS), md('## This notebook: ' + s['title'].split('· ', 1)[1])]
    if src_model:
        label = 'Explainability functions' if s['stage'] == 'EAA_IoU_analysis' else 'Model definition'
        cells.append(code(f'# 8. {label}\n' + src_model))
    cells.append(code(('# 9. ' if src_model else '# 8. ') + 'Training, evaluation and logging\n' + src_run))
    if s['needs']:
        cells.append(code(require(*s['needs'])))
    cells.append(code(f"# Run it (safe to re-run: resumes or skips finished work)\nSTAGE = {s['stage']!r}\nstage(STAGE, {s['run']})"))
    if s['stage'] == 'EAA_IoU_analysis':
        cells.append(code('# Results for this notebook\nimport pandas as pd\n'
                          'summary = json.load(open(os.path.join(METRICS_DIR, "eaa_iou_summary.json")))\n'
                          'print(json.dumps({k: v for k, v in summary.items() if k != "threshold_table"}, indent=2))\n'
                          'print(pd.read_csv(os.path.join(METRICS_DIR, "eaa_iou_threshold_table.csv")).to_string(index=False))'))
    elif s['stage'] == 'HF_upload':
        cells.append(code('# Check what is now on Hugging Face\nfrom huggingface_hub import list_repo_files\n'
                          'files = list_repo_files(HF_REPO_ID)\n'
                          'print("Web app files present:", [f for f in WEBAPP_FILES if "models/" + f in files])\n'
                          'print("Web app files missing:", [f for f in WEBAPP_FILES if "models/" + f not in files])'))
    else:
        cells.append(code(show_results(*s['show'])))
    for c in cells:
        if c['cell_type'] == 'code':
            compile(c['source'], s['file'], 'exec')
    json.dump(nb(cells), open(f"{OUT}/{s['file']}", 'w'), indent=1)
    print(f"wrote {s['file']:42s} {len(cells)} cells")
