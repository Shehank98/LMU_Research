#!/usr/bin/env python3
# Builder: emits the resumable Colab training+logging notebook.
import json, os

C = []  # cells
def md(s):  C.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s):C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s})

md("""# Brain Tumour MRI — Full Training & Result Logging
**Final code set · 2026-10-22**

Run this notebook **top to bottom** in Google Colab (`Runtime > Change runtime type > GPU`).

**It is built so it cannot lose your work:**
- Everything (models, metrics, logs) is saved to **Google Drive**, so it survives disconnects.
- Deep models **checkpoint after every epoch** — if Colab crashes, just re-run the cell and it **resumes** from the last epoch, it does not start over.
- Each model runs in its **own cell**; if one fails, the others are already saved.
- **Finished stages are skipped** automatically on a re-run.
- All results are saved as **data, not images**: `results_master.csv`, per-model `classification_report.json`, `confusion_matrix.csv`, and per-epoch `history.csv`.

After it finishes, open the `metrics/results_master.csv` on your Drive — that is your results table for the final report.""")

# 1 mount
code("""# 1. Mount Google Drive (results are saved here so they survive disconnects)
from google.colab import drive
drive.mount('/content/drive')""")

# 2 config
code("""# 2. Configuration  --  EDIT THE TWO PATHS BELOW IF NEEDED
import os

DRIVE_DATA_DIR = '/content/drive/MyDrive/MRI_DATASET'                # your dataset on Drive (Training/ + Testing/)
OUTPUT_DIR     = '/content/drive/MyDrive/brain_tumor_results_2026-10-22'  # all results are written here

# Hugging Face: trained models are uploaded here and the web app downloads them from here
HF_REPO_ID     = 'shehank98/brain-tumor-mri-models'
RESULTS_PREFIX = 'results_2026-10-22'

# The final ensemble refuses to run unless all 6 predictors exist (keeps the report honest)
ALLOW_PARTIAL_ENSEMBLE = False

# Explainability method for the EAA-IoU analysis: 'hirescam' (default, correct for every model) or 'gradcam'
CAM_METHOD = 'hirescam'

# Reading thousands of images from Drive every epoch is very slow and is a common reason
# long Colab runs stall. The dataset is copied ONCE to Colab's fast local disk and read from there.
import shutil
LOCAL_DATA_DIR = '/content/MRI_DATASET'
if not os.path.isdir(os.path.join(LOCAL_DATA_DIR, 'Testing')):
    print('Copying dataset from Drive to local disk (one time, about 1 to 3 minutes) ...')
    shutil.copytree(DRIVE_DATA_DIR, LOCAL_DATA_DIR, dirs_exist_ok=True)
DATA_DIR = LOCAL_DATA_DIR

# Hyperparameters (kept identical to the project so results are comparable)
EPOCHS_CNN       = 20
EPOCHS_TRANSFER  = 10
EPOCHS_EXTRACTOR = 10
BATCH    = 32
SIZE_STD = 256      # standalone CNN / InceptionV3 / Xception
SIZE_EXT = 128      # feature extractors that feed classical ML
SEED     = 42
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_CLASSES   = 4

MODELS_DIR  = os.path.join(OUTPUT_DIR, 'models')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
HISTORY_DIR = os.path.join(OUTPUT_DIR, 'history')
CONF_DIR    = os.path.join(OUTPUT_DIR, 'confusion')
LOG_DIR     = os.path.join(OUTPUT_DIR, 'logs')
for d in [MODELS_DIR, METRICS_DIR, HISTORY_DIR, CONF_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
MASTER_CSV  = os.path.join(METRICS_DIR, 'results_master.csv')
RUN_LOG     = os.path.join(LOG_DIR, 'run_log.txt')
PROGRESS    = os.path.join(OUTPUT_DIR, 'progress.json')

# sanity check: every class folder must exist and contain images; the real counts are saved for the report
import json
counts = {}
for split in ['Training', 'Testing']:
    counts[split] = {}
    for c in CLASS_NAMES:
        pth = os.path.join(DATA_DIR, split, c)
        n = len(os.listdir(pth)) if os.path.isdir(pth) else 0
        counts[split][c] = n
        print('{:8s} {:11s} {:5d} images {}'.format(split, c, n, '' if n else '  <-- MISSING, check DRIVE_DATA_DIR'))
    print('{:8s} {:11s} {:5d} images in total'.format(split, '', sum(counts[split].values())))
json.dump(counts, open(os.path.join(METRICS_DIR, 'dataset_counts.json'), 'w'), indent=2)
print('Results ->', OUTPUT_DIR)""")

# 3 imports + logging + progress
code("""# 3. Imports, seeds, logging and progress tracking
import json, time, csv, random, datetime, traceback
import numpy as np
random.seed(SEED); np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
import joblib

print('TensorFlow', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU') or 'NONE  (Runtime > Change runtime type > GPU)')

def log(msg):
    line = '[{:%Y-%m-%d %H:%M:%S}] {}'.format(datetime.datetime.now(), msg)
    print(line)
    with open(RUN_LOG, 'a') as f:
        f.write(line + '\\n')

def _progress():
    return json.load(open(PROGRESS)) if os.path.exists(PROGRESS) else {}

def mark(stage, status, info=''):
    data = _progress()
    data[stage] = {'status': status, 'info': str(info)[:300], 'time': str(datetime.datetime.now())}
    json.dump(data, open(PROGRESS, 'w'), indent=2)

def is_done(stage):
    return _progress().get(stage, {}).get('status') == 'done'

def stage(name, fn):
    # Runs fn() guarded: logs any error and keeps going so one failure never loses the rest.
    if is_done(name):
        log('SKIP ' + name + ' (already done)'); return
    mark(name, 'running')
    try:
        fn(); mark(name, 'done'); log('DONE ' + name)
    except Exception as e:
        mark(name, 'error', e)
        log('ERROR in ' + name + ': ' + repr(e) + '\\n' + traceback.format_exc())
        print('\\n>>> ' + name + ' failed but was isolated. Fix and re-run THIS cell; other stages are safe.')""")

# 4 metrics logging
code("""# 4. Metric logging  --  saves accuracy and every other number as DATA (csv/json), not images
def save_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    with open(os.path.join(CONF_DIR, name + '_confusion_matrix.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['actual_vs_predicted'] + CLASS_NAMES)
        for i, row in enumerate(cm):
            w.writerow([CLASS_NAMES[i]] + [int(v) for v in row])
    return cm

def log_metrics(name, y_true, y_pred, train_time=None, note=''):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _  = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    pc, rc, fc, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0)
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                zero_division=0, output_dict=True)
    json.dump(rep, open(os.path.join(METRICS_DIR, name + '_classification_report.json'), 'w'), indent=2)
    save_confusion(name, y_true, y_pred)
    header = (['model','accuracy','precision_macro','recall_macro','f1_macro']
              + ['f1_' + c for c in CLASS_NAMES]
              + ['n_test','train_time_sec','timestamp','note'])
    row = ([name, round(acc,4), round(p,4), round(r,4), round(f,4)]
           + [round(float(x),4) for x in fc]
           + [int(len(y_true)), round(float(train_time or 0),1),
              str(datetime.datetime.now()), note])
    new = not os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, 'a', newline='') as fcsv:
        w = csv.writer(fcsv)
        if new: w.writerow(header)
        w.writerow(row)
    log('{}: accuracy={:.2f}%   macro-F1={:.3f}'.format(name, acc*100, f))
    return acc""")

# 5 resumable fit
code("""# 5. Resumable training  --  checkpoints every epoch; a crash resumes, it does not restart
class EpochTracker(Callback):
    def __init__(self, path): super().__init__(); self.path = path
    def on_epoch_end(self, epoch, logs=None):
        json.dump({'epoch': int(epoch) + 1}, open(self.path, 'w'))

def resumable_fit(name, build_fn, train_ds, val_ds, epochs):
    final_path = os.path.join(MODELS_DIR, name + '.h5')
    ckpt_path  = os.path.join(MODELS_DIR, name + '_ckpt.keras')
    epoch_path = os.path.join(MODELS_DIR, name + '_epoch.json')
    if os.path.exists(final_path):
        log(name + ': final model already exists, loading')
        return load_model(final_path)
    init_epoch = 0
    if os.path.exists(ckpt_path) and os.path.exists(epoch_path):
        model = load_model(ckpt_path)
        init_epoch = json.load(open(epoch_path)).get('epoch', 0)
        log('{}: resuming from epoch {}'.format(name, init_epoch))
    else:
        model = build_fn()
    cbs = [ModelCheckpoint(ckpt_path, save_freq='epoch', save_best_only=False),
           CSVLogger(os.path.join(HISTORY_DIR, name + '_history.csv'), append=True),
           EpochTracker(epoch_path)]
    if init_epoch < epochs:
        model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  initial_epoch=init_epoch, callbacks=cbs, verbose=1)
    model.save(final_path)                       # permanent copy on Drive
    for p in (ckpt_path, epoch_path):
        try: os.remove(p)
        except OSError: pass
    return model""")

# 6 data loaders
code("""# 6. Data loaders
def tf_datasets(size):
    common = dict(image_size=(size, size), batch_size=BATCH, label_mode='int', seed=SEED)
    tr = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'Training'), validation_split=0.1, subset='training',
        shuffle=True, **common)
    va = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'Training'), validation_split=0.1, subset='validation',
        shuffle=True, **common)
    te = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'Testing'), shuffle=False,
        image_size=(size, size), batch_size=BATCH, label_mode='int')
    norm = lambda x, y: (x / 255.0, y)
    AT = tf.data.AUTOTUNE
    return tr.map(norm).prefetch(AT), va.map(norm).prefetch(AT), te.map(norm).prefetch(AT)

def load_numpy(split, size, grayscale=False):
    import cv2
    X, y = [], []
    base = os.path.join(DATA_DIR, split)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    for idx, c in enumerate(CLASS_NAMES):
        folder = os.path.join(base, c)
        for fn in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fn), flag)
            if img is None: continue
            img = cv2.resize(img, (size, size))
            X.append(img); y.append(idx)
    X = np.asarray(X, dtype='float32') / 255.0
    if not grayscale:
        X = X[..., ::-1]           # BGR(cv2) -> RGB to match the Keras loaders
    return X, np.asarray(y)""")

# 7 builders
code("""# 7. Model builders (same architectures as the project)
def build_cnn(size, feat_name='dense'):
    m = models.Sequential([
        layers.Input(shape=(size, size, 3)),   # explicit Input: required by Keras 3 to read inner layers
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'), layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'), layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'), layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu', name=feat_name),
        layers.Dropout(0.5, name='feature_dropout'),
        layers.Dense(N_CLASSES, activation='softmax', name='predictions'),   # explicit names: no clashes
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def _base(kind, size):
    App = tf.keras.applications.InceptionV3 if kind == 'inception' else tf.keras.applications.Xception
    b = App(input_shape=(size, size, 3), weights='imagenet', include_top=False)
    b.trainable = False
    return b

def build_transfer_standalone(kind, size=256):
    base = _base(kind, size)
    x = layers.GlobalAveragePooling2D()(base.output)
    out = layers.Dense(N_CLASSES, activation='softmax', name='dense')(x)
    m = Model(base.input, out)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def build_transfer_extractor(kind, feat_name, size=128):
    base = _base(kind, size)
    x = layers.Flatten()(base.output)
    x = layers.Dense(256, activation='relu', name=feat_name)(x)
    x = layers.Dropout(0.5, name='feature_dropout')(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='predictions')(x)
    m = Model(base.input, out)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m""")

# 8 helpers: predict/extract/classical + run_* defs
code("""# 8. Prediction / feature-extraction / classical helpers, and the stage functions
def predict_ds(model, ds):
    yt, yp = [], []
    for X, y in ds:
        p = model.predict(X, verbose=0)
        yp.extend(np.argmax(p, axis=1)); yt.extend(y.numpy())
    return np.asarray(yt), np.asarray(yp)

def extract_features(model, layer_name, X, batch=64):
    try:
        ex = Model(model.inputs, model.get_layer(layer_name).output)
        return ex.predict(X, batch_size=batch, verbose=1)
    except (AttributeError, ValueError):
        # Keras 3 Sequential without a defined input: run the layers by hand up to the feature layer
        out = []
        for s in range(0, len(X), batch):
            h = tf.convert_to_tensor(X[s:s + batch], tf.float32)
            for layer in model.layers:
                h = layer(h, training=False)
                if layer.name == layer_name:
                    break
            out.append(h.numpy())
        return np.concatenate(out)

def train_classical(tag, Xtr, ytr, Xte, yte, include_lr=False):
    # Each classifier is trained and SAVED separately, so a crash in one keeps the others.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    specs = {'rf':  RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
             'dt':  DecisionTreeClassifier(random_state=SEED),
             'svm': SVC(probability=True, random_state=SEED, cache_size=1000)}
    if include_lr:
        specs['lr'] = LogisticRegression(max_iter=1000, n_jobs=-1)
    fitted = {}
    for key, clf in specs.items():
        mp = os.path.join(MODELS_DIR, tag + '_' + key + '.pkl')
        if os.path.exists(mp):
            fitted[key] = joblib.load(mp); log(tag + '_' + key + ': loaded (skip)'); continue
        t = time.time(); log(tag + '_' + key + ': training ...')
        clf.fit(Xtr, ytr); joblib.dump(clf, mp)
        log_metrics(tag + '_' + key, yte, clf.predict(Xte), time.time() - t)
        fitted[key] = clf
    vote_keys = [k for k in ['rf','dt','svm'] if k in fitted]
    probs = sum(fitted[k].predict_proba(Xte) for k in vote_keys) / len(vote_keys)
    log_metrics(tag + '_softvote_ensemble', yte, np.argmax(probs, axis=1))
    return fitted

# ---------- deep standalone models ----------
def run_cnn():
    tr, va, te = tf_datasets(SIZE_STD)
    m = resumable_fit('cnn_model', lambda: build_cnn(SIZE_STD, 'dense'), tr, va, EPOCHS_CNN)
    yt, yp = predict_ds(m, te); log_metrics('CNN_standalone', yt, yp)

def run_inception():
    tr, va, te = tf_datasets(SIZE_STD)
    m = resumable_fit('inceptionv3_model', lambda: build_transfer_standalone('inception'),
                      tr, va, EPOCHS_TRANSFER)
    yt, yp = predict_ds(m, te); log_metrics('InceptionV3_standalone', yt, yp)

def run_xception():
    tr, va, te = tf_datasets(SIZE_STD)
    m = resumable_fit('xception_model', lambda: build_transfer_standalone('xception'),
                      tr, va, EPOCHS_TRANSFER)
    yt, yp = predict_ds(m, te); log_metrics('Xception_standalone', yt, yp)

# ---------- deep feature extractor + classical ensemble ----------
def _run_feature_ensemble(ext_name, build_ext, layer, tag):
    tr, va, _ = tf_datasets(SIZE_EXT)
    ext = resumable_fit(ext_name, build_ext, tr, va, EPOCHS_EXTRACTOR)
    Xtr, ytr = load_numpy('Training', SIZE_EXT)
    Xte, yte = load_numpy('Testing',  SIZE_EXT)
    ftr = extract_features(ext, layer, Xtr)
    fte = extract_features(ext, layer, Xte)
    del Xtr, Xte
    train_classical(tag, ftr, ytr, fte, yte)

def run_cnn_ensemble():
    _run_feature_ensemble('cnn_ext', lambda: build_cnn(SIZE_EXT, 'dense'), 'dense', 'cnn_feat')

def run_inception_ensemble():
    _run_feature_ensemble('inceptionv3_ext',
        lambda: build_transfer_extractor('inception', 'dense_2'), 'dense_2', 'inceptionv3_feat')

def run_xception_ensemble():
    _run_feature_ensemble('xception_ext',
        lambda: build_transfer_extractor('xception', 'dense_3'), 'dense_3', 'xception_feat')

# ---------- classical ML on raw pixels ----------
def run_traditional_raw():
    Xtr, ytr = load_numpy('Training', SIZE_STD, grayscale=True)
    Xte, yte = load_numpy('Testing',  SIZE_STD, grayscale=True)
    Xtr = Xtr.reshape(len(Xtr), -1); Xte = Xte.reshape(len(Xte), -1)
    train_classical('raw', Xtr, ytr, Xte, yte, include_lr=True)

# ---------- final 6-model ensemble ----------
def run_final_ensemble():
    X256, yte = load_numpy('Testing', SIZE_STD)
    X128, _   = load_numpy('Testing', SIZE_EXT)
    probs, names = [], []
    for fn, nm in [('cnn_model.h5','CNN'), ('inceptionv3_model.h5','InceptionV3'),
                   ('xception_model.h5','Xception')]:
        mp = os.path.join(MODELS_DIR, fn)
        if os.path.exists(mp):
            probs.append(load_model(mp).predict(X256, verbose=0)); names.append(nm)
            log('final: used ' + fn)
    for ext_fn, layer, tag, nm in [('cnn_ext.h5','dense','cnn_feat','CNN_ensemble'),
                                   ('inceptionv3_ext.h5','dense_2','inceptionv3_feat','InceptionV3_ensemble'),
                                   ('xception_ext.h5','dense_3','xception_feat','Xception_ensemble')]:
        mp = os.path.join(MODELS_DIR, ext_fn)
        if not os.path.exists(mp): continue
        feats = extract_features(load_model(mp), layer, X128)
        clfs = [joblib.load(os.path.join(MODELS_DIR, tag + '_' + k + '.pkl'))
                for k in ['rf','dt','svm']
                if os.path.exists(os.path.join(MODELS_DIR, tag + '_' + k + '.pkl'))]
        if clfs:
            probs.append(sum(c.predict_proba(feats) for c in clfs) / len(clfs)); names.append(nm)
            log('final: used ' + ext_fn + ' + ' + tag)
    if len(probs) < 6 and not ALLOW_PARTIAL_ENSEMBLE:
        raise RuntimeError('The final ensemble needs all 6 predictors but found {} ({}). Train the missing '
                           'models first, or set ALLOW_PARTIAL_ENSEMBLE = True in cell 2.'.format(
                               len(probs), ', '.join(names) or 'none'))
    if not probs:
        raise RuntimeError('No trained models found yet.')
    P = np.stack(probs)                                  # (n_models, N, 4)
    summed = P.sum(axis=0)
    pred_conf = np.argmax(summed, axis=1)
    log_metrics('FINAL_confidence_weighted', yte, pred_conf, note=str(P.shape[0]) + ' models')
    votes = np.argmax(P, axis=2)                          # (n_models, N)
    maj = np.array([np.bincount(votes[:, i], minlength=N_CLASSES).argmax()
                    for i in range(votes.shape[1])])
    log_metrics('FINAL_majority_vote', yte, maj, note=str(P.shape[0]) + ' models')
    # per-image predictions as DATA (used by the EAA-IoU analysis and for the report)
    import pandas as pd
    df = pd.DataFrame({'idx': np.arange(len(yte)), 'true': yte,
                       'pred_confidence_weighted': pred_conf, 'pred_majority': maj,
                       'ensemble_confidence': summed.max(axis=1) / summed.sum(axis=1)})
    for nm, p in zip(names, probs):
        df['pred_' + nm] = p.argmax(axis=1); df['conf_' + nm] = p.max(axis=1)
    df.to_csv(os.path.join(METRICS_DIR, 'final_per_image_predictions.csv'), index=False)
    np.savez(os.path.join(METRICS_DIR, 'final_probs.npz'), P=P, y=yte, names=np.array(names))
    log('final: per-image predictions saved')

print('Stage functions ready.')""")

code(r"""# 9. Explainable AI (Grad-CAM family + EAA-IoU) and Hugging Face upload
# ---------- batched CAM that works for Sequential (CNN) and Functional (Inception/Xception) ----------
_GRAD_MODELS = {}

def _last_conv_index(model):
    idx = None
    for i, l in enumerate(model.layers):
        if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
            idx = i
    return idx

def _last_4d_layer(model):
    for l in reversed(model.layers):
        try:
            if len(l.output.shape) == 4:
                return l.name
        except Exception:
            continue
    raise ValueError('no spatial layer found in ' + model.name)

def _final_dense(model):
    last = model.layers[-1]
    return last if isinstance(last, tf.keras.layers.Dense) else None

def _logits(dense, h):
    # Pre-softmax class scores. Using the softmax output instead makes confident models give
    # vanishing gradients and EMPTY heatmaps, so CAMs are always computed from the logits.
    z = tf.matmul(h, dense.kernel)
    return z + dense.bias if dense.bias is not None else z

def gradcam_batch(model, X, class_idx, out_size=256):
    # X: (B,H,W,3) in [0,1]; class_idx: (B,) explained class per image. Returns (B,out,out) in [0,1].
    X = tf.convert_to_tensor(X, tf.float32)
    cls = tf.convert_to_tensor(np.asarray(class_idx), tf.int32)
    dense = _final_dense(model)
    if isinstance(model, tf.keras.Sequential):
        ci = _last_conv_index(model)
        body = model.layers[:-1] if dense is not None else model.layers
        with tf.GradientTape() as tape:
            x, conv = X, None
            for i, l in enumerate(body):
                x = l(x, training=False)
                if i == ci:
                    conv = x
                    tape.watch(conv)
            scores = _logits(dense, x) if dense is not None else x
            score = tf.gather(scores, cls, axis=1, batch_dims=1)
        grads = tape.gradient(score, conv)
    else:
        key = id(model)
        if key not in _GRAD_MODELS:
            head_in = dense.input if dense is not None else model.output
            _GRAD_MODELS[key] = Model(model.inputs, [model.get_layer(_last_4d_layer(model)).output, head_in])
        with tf.GradientTape() as tape:
            conv, h = _GRAD_MODELS[key](X, training=False)
            scores = _logits(dense, h) if dense is not None else h
            score = tf.gather(scores, cls, axis=1, batch_dims=1)
        grads = tape.gradient(score, conv)
    if CAM_METHOD == 'gradcam':
        cam = tf.einsum('bhwc,bc->bhw', conv, tf.reduce_mean(grads, axis=(1, 2)))
    else:
        # HiResCAM (Draelos and Carin, 2020): element-wise gradient x activation. Identical to
        # Grad-CAM for Global-Average-Pooling heads (InceptionV3, Xception) and correct for the
        # CNN's Flatten + Dense head, where averaging the gradients cancels out to an empty map.
        cam = tf.reduce_sum(grads * conv, axis=-1)
    cam = tf.nn.relu(cam)
    cam = tf.math.divide_no_nan(cam, tf.reduce_max(cam, axis=(1, 2), keepdims=True))
    return tf.image.resize(cam[..., None], (out_size, out_size))[..., 0].numpy()

def iou(a, b, t=0.5):
    A, B = a >= t, b >= t
    union = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum() / union) if union > 0 else 0.0

# ---------- EAA-IoU over the whole test set (resumable, writes rows as it goes) ----------
def run_eaa_iou(batch=16):
    import pandas as pd
    pred_csv = os.path.join(METRICS_DIR, 'final_per_image_predictions.csv')
    if not os.path.exists(pred_csv):
        raise RuntimeError('Run the FINAL_ensemble stage first.')
    pred = pd.read_csv(pred_csv)
    X, y = load_numpy('Testing', SIZE_STD)
    mdl = {nm: load_model(os.path.join(MODELS_DIR, fn)) for nm, fn in
           [('CNN','cnn_model.h5'), ('Xception','xception_model.h5'), ('InceptionV3','inceptionv3_model.h5')]}
    out_csv = os.path.join(METRICS_DIR, 'eaa_iou_per_image.csv')
    start = len(pd.read_csv(out_csv)) if os.path.exists(out_csv) else 0
    if start:
        log('EAA-IoU: resuming at image {}'.format(start))
    cols = ['idx','true','pred_final','correct','iou_cnn_xcp','iou_cnn_inc','iou_xcp_inc','eaa_iou']
    for s in range(start, len(X), batch):
        e = min(s + batch, len(X))
        cls = pred['pred_confidence_weighted'].values[s:e]          # explain the ensemble decision
        maps = {nm: gradcam_batch(m, X[s:e], cls) for nm, m in mdl.items()}
        rows = []
        for k in range(e - s):
            a, b, c = maps['CNN'][k], maps['Xception'][k], maps['InceptionV3'][k]
            p = [iou(a, b), iou(a, c), iou(b, c)]
            i = s + k
            rows.append([i, int(y[i]), int(cls[k]), int(y[i] == cls[k])] + [round(v, 4) for v in p]
                        + [round(float(np.mean(p)), 4)])
        pd.DataFrame(rows, columns=cols).to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
        if (s // batch) % 20 == 0:
            log('EAA-IoU: {}/{} images'.format(e, len(X)))
    summarise_eaa()

def summarise_eaa():
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    d = pd.read_csv(os.path.join(METRICS_DIR, 'eaa_iou_per_image.csv')).drop_duplicates('idx')
    wrong = d[d.correct == 0]; right = d[d.correct == 1]
    s = {'cam_method': CAM_METHOD, 'n_images': int(len(d)), 'n_misclassified': int(len(wrong)),
         'mean_eaa_all': round(float(d.eaa_iou.mean()), 4),
         'mean_eaa_correct': round(float(right.eaa_iou.mean()), 4) if len(right) else None,
         'mean_eaa_misclassified': round(float(wrong.eaa_iou.mean()), 4) if len(wrong) else None,
         'mean_eaa_by_true_class': {CLASS_NAMES[int(k)]: round(float(v), 4)
                                    for k, v in d.groupby('true').eaa_iou.mean().items()}}
    if len(wrong) and len(right):
        s['auroc_low_eaa_predicts_error'] = round(float(roc_auc_score(1 - d.correct, -d.eaa_iou)), 4)
        try:
            from scipy.stats import mannwhitneyu
            s['mann_whitney_p_value'] = float(mannwhitneyu(right.eaa_iou, wrong.eaa_iou,
                                                           alternative='greater').pvalue)
        except Exception as e:
            s['mann_whitney_p_value'] = 'not computed: ' + str(e)
    table = []
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7]:
        flag = d.eaa_iou < t
        table.append({'threshold': t,
                      'flagged_pct': round(100 * flag.mean(), 2),
                      'errors_caught_pct': round(100 * (flag & (d.correct == 0)).sum() / max(len(wrong), 1), 2),
                      'error_rate_in_flagged_pct': round(100 * (1 - d[flag].correct.mean()), 2) if flag.any() else None,
                      'error_rate_in_unflagged_pct': round(100 * (1 - d[~flag].correct.mean()), 2) if (~flag).any() else None})
    s['threshold_table'] = table
    json.dump(s, open(os.path.join(METRICS_DIR, 'eaa_iou_summary.json'), 'w'), indent=2)
    pd.DataFrame(table).to_csv(os.path.join(METRICS_DIR, 'eaa_iou_threshold_table.csv'), index=False)
    log('EAA-IoU summary: ' + json.dumps({k: v for k, v in s.items() if k != 'threshold_table'}))

# ---------- upload models (names the web app expects) and all result files to Hugging Face ----------
WEBAPP_FILES = ['cnn_model.h5', 'inceptionv3_model.h5', 'xception_model.h5',
                'cnn_ext.h5', 'inceptionv3_ext.h5', 'xception_ext.h5',
                'cnn_ensemble_model.pkl', 'inceptionv3_ensemble_model.pkl', 'xception_ensemble_model.pkl',
                'manifest.json']

def _hf_token():
    try:
        from google.colab import userdata
        tok = userdata.get('HF_TOKEN')
        if tok: return tok
    except Exception:
        pass
    import getpass
    return getpass.getpass('Paste your Hugging Face WRITE token: ')

def run_hf_upload():
    from huggingface_hub import HfApi, CommitOperationAdd
    # one file per feature ensemble: a list [RandomForest, DecisionTree, SVM]; the app averages their probabilities
    for tag, out in [('cnn_feat','cnn_ensemble_model.pkl'), ('inceptionv3_feat','inceptionv3_ensemble_model.pkl'),
                     ('xception_feat','xception_ensemble_model.pkl')]:
        parts = [joblib.load(os.path.join(MODELS_DIR, tag + '_' + k + '.pkl')) for k in ['rf','dt','svm']
                 if os.path.exists(os.path.join(MODELS_DIR, tag + '_' + k + '.pkl'))]
        if parts:
            joblib.dump(parts, os.path.join(MODELS_DIR, out))
    manifest = {'created': str(datetime.datetime.now()), 'class_names': CLASS_NAMES,
                'standalone': {'cnn': 'cnn_model.h5', 'inception': 'inceptionv3_model.h5',
                               'xception': 'xception_model.h5', 'input_size': SIZE_STD},
                'extractors': {'cnn':       {'file': 'cnn_ext.h5',         'layer': 'dense',   'input_size': SIZE_EXT},
                               'inception': {'file': 'inceptionv3_ext.h5', 'layer': 'dense_2', 'input_size': SIZE_EXT},
                               'xception':  {'file': 'xception_ext.h5',    'layer': 'dense_3', 'input_size': SIZE_EXT}},
                'ensembles':  {'cnn': 'cnn_ensemble_model.pkl', 'inception': 'inceptionv3_ensemble_model.pkl',
                               'xception': 'xception_ensemble_model.pkl'},
                'normalisation': 'RGB / 255.0', 'results_prefix': RESULTS_PREFIX,
                'versions': {'tensorflow': tf.__version__, 'keras': tf.keras.__version__ if hasattr(tf.keras, '__version__') else '',
                             'scikit_learn': __import__('sklearn').__version__}}
    json.dump(manifest, open(os.path.join(MODELS_DIR, 'manifest.json'), 'w'), indent=2)

    api = HfApi(token=_hf_token())
    api.create_repo(HF_REPO_ID, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo='models/' + f, path_or_fileobj=os.path.join(MODELS_DIR, f))
           for f in WEBAPP_FILES if os.path.exists(os.path.join(MODELS_DIR, f))]
    if not ops:
        raise RuntimeError('No model files to upload yet. Run the training stages first.')
    log('HF: uploading {} model files (large files take a few minutes) ...'.format(len(ops)))
    api.create_commit(repo_id=HF_REPO_ID, operations=ops, commit_message='Models from final run ' + RESULTS_PREFIX)
    res = []
    for sub in ['metrics', 'confusion', 'history', 'logs']:
        folder = os.path.join(OUTPUT_DIR, sub)
        for fn in sorted(os.listdir(folder)):
            if fn.endswith('.npz'): continue
            res.append(CommitOperationAdd(path_in_repo=RESULTS_PREFIX + '/' + sub + '/' + fn,
                                          path_or_fileobj=os.path.join(folder, fn)))
    api.create_commit(repo_id=HF_REPO_ID, operations=res, commit_message='Results ' + RESULTS_PREFIX)
    log('HF: uploaded {} model files and {} result files to https://huggingface.co/{}'.format(
        len(ops), len(res), HF_REPO_ID))

print('XAI and upload functions ready.')""")

md("""## Run the stages below one at a time
Run each cell and wait for it to finish. If a cell crashes or Colab disconnects, just **re-run that same cell** — it resumes from the last epoch (deep models) or skips what is already saved (classical models). Already-finished stages print `SKIP`.""")

code("stage('CNN_standalone',        run_cnn)")
code("stage('InceptionV3_standalone', run_inception)")
code("stage('Xception_standalone',    run_xception)")
code("stage('CNN_feature_ensemble',       run_cnn_ensemble)")
code("stage('InceptionV3_feature_ensemble', run_inception_ensemble)")
code("stage('Xception_feature_ensemble',    run_xception_ensemble)")
code("# Classical ML on raw pixels. The SVM here is the slowest step (large feature vectors).\n# It is saved the moment it finishes, and skipped on any re-run.\nstage('Traditional_raw_pixels', run_traditional_raw)")
code("# Run this only after the stages above have finished.\nstage('FINAL_ensemble', run_final_ensemble)")
code("# Explainable AI: Grad-CAM from CNN, Xception and InceptionV3 on every test image,\n# pairwise IoU and the EAA-IoU score. Resumable: if it stops, re-run and it continues.\nstage('EAA_IoU_analysis', run_eaa_iou)")
code("# Upload the trained models (with the names the web app expects) and every result file\n# to Hugging Face. Needs a WRITE token: add it once as a Colab secret named HF_TOKEN\n# (key icon on the left), or paste it when asked.\nstage('HF_upload', run_hf_upload)")

md("## Results summary")
code("""# Show the master results table and package everything for download
import pandas as pd
if os.path.exists(MASTER_CSV):
    df = pd.read_csv(MASTER_CSV).drop_duplicates('model', keep='last')
    pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
    print(df[['model','accuracy','precision_macro','recall_macro','f1_macro','train_time_sec']].to_string(index=False))
else:
    print('No results yet.')
print('\\nProgress:'); print(json.dumps(_progress(), indent=2))

# Zip the whole results folder so you can download it in one click
import shutil
zip_path = '/content/brain_tumor_results_2026-10-22'
shutil.make_archive(zip_path, 'zip', OUTPUT_DIR)
print('\\nEverything is also on your Drive at:', OUTPUT_DIR)
print('Zip for download:', zip_path + '.zip')
try:
    from google.colab import files; files.download(zip_path + '.zip')
except Exception as e:
    print('(download skipped:', e, ')')""")

md("""## Optional — download the dataset
Skip this if your dataset is already on Drive at `DRIVE_DATA_DIR` with `Training/` and `Testing/` folders.

To pull it from Kaggle, upload your `kaggle.json` (Kaggle > Account > Create New API Token) when prompted, then run:
```python
from google.colab import files; files.upload()          # choose kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!pip -q install kaggle
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p /content
!unzip -q /content/brain-tumor-mri-dataset.zip -d /content/drive/MyDrive/MRI_DATASET
```
After it unzips, check `DRIVE_DATA_DIR = '/content/drive/MyDrive/MRI_DATASET'` in cell 2 and re-run from the top.

## Troubleshooting
- **"MISSING" in the cell 2 check** — fix `DRIVE_DATA_DIR`; it must contain `Training/` and `Testing/`, each with the four class folders. Then run `!rm -rf /content/MRI_DATASET` and re-run cell 2.
- **Session crashed / disconnected** — reconnect, run cells 1–9, then re-run the stage cell that was interrupted. It resumes.
- **Out of memory on the final ensemble** — run it on its own after restarting the runtime (`Runtime > Restart`), then run cells 1–9 and the final cell.
- **Where are my numbers?** — `MyDrive/brain_tumor_results_2026-10-22/metrics/results_master.csv` (one row per model), plus `confusion/` and `history/`.""")

nb = {"cells": C,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

# syntax-check every code cell
bad = 0
for i, c in enumerate(C):
    if c["cell_type"] == "code":
        try: compile(c["source"], "cell_%d" % i, "exec")
        except SyntaxError as e:
            bad += 1; print("SYNTAX ERROR in cell", i, ":", e)
print("code cells checked, syntax errors:", bad)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Brain_Tumor_Train_And_Log.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "| cells:", len(C))
