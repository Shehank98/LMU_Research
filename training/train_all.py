"""
train_all.py — Run the full training pipeline end-to-end.

Produces all 9 model files in MODELS/:
  cnn_standalone.h5, InceptionV3.h5, Xception.h5
  cnn_ensemble.h5, inc_ensemble.h5, xcp_ensemble.h5
  cnn_ensemble_model.pkl, inc_ensemble_model.pkl, xcp_ensemble_model.pkl

Run from repo root or from training/:
    python training/train_all.py
    -- or --
    cd training && python train_all.py

GPU strongly recommended — expected total time: 8–16 h on a T4 GPU.
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(__file__))

SCRIPTS = [
    ('train_01_cnn.py',          'Custom CNN (standalone + extractor + classical ensemble)'),
    ('train_02_inception.py',    'InceptionV3 (standalone + extractor + classical ensemble)'),
    ('train_03_xception.py',     'Xception (standalone + extractor + classical ensemble)'),
    ('train_04_classical_raw.py','Classical ML on raw pixels (research baseline)'),
]


def run_script(filename):
    script_path = os.path.join(os.path.dirname(__file__), filename)
    namespace = {'__name__': '__main__', '__file__': script_path}
    with open(script_path) as f:
        exec(compile(f.read(), script_path, 'exec'), namespace)


if __name__ == '__main__':
    total_start = time.time()
    print('=' * 60)
    print('  NeuroScan AI — Full Training Pipeline')
    print('  Seed: 42  |  Batch: 32  |  Norm: /255.0')
    print('=' * 60)

    for i, (script, description) in enumerate(SCRIPTS, 1):
        print(f'\n[{i}/{len(SCRIPTS)}] {description}')
        print('-' * 60)
        t0 = time.time()
        run_script(script)
        elapsed = time.time() - t0
        print(f'\n  Completed in {elapsed/60:.1f} min')

    total = time.time() - total_start
    print('\n' + '=' * 60)
    print(f'  All training complete — {total/3600:.1f} h total')
    print('  Run upload_to_hf.py to push models to HuggingFace Hub.')
    print('=' * 60)
