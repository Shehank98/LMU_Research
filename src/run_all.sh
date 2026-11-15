#!/bin/sh
# Full pipeline: three deep models, the classical stage on their features, then the ensemble.
# Finished steps are skipped, so the script can simply be run again after an interruption.
set -e
cd "$(dirname "$0")"
OUT=${RUN_DIR:-../runs/v2}
for m in cnn inceptionv3 xception; do
    [ -f "$OUT/probs/$m.npz" ] || python train.py $m
    [ -f "$OUT/probs/${m}_feat_softvote.npz" ] || python classical.py $m
done
python ensemble.py
