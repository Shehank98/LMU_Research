#!/bin/sh
# Run every analysis script on the improved models (after src/run_all.sh has finished).
set -e
cd "$(dirname "$0")"
python 14_clean_subset.py
export MODEL_SET=improved
for s in 02_xai_examples 16_eaa_iou 12_shortcut_investigation 03_xai_quantitative 03b_xai_summary_sanity 07_shap_rf 06_xai_example_figures 10_xai_quant_figures; do
    echo "== $s"; python $s.py
done
unset MODEL_SET
python 17_improved_figures.py
echo "ANALYSIS DONE"
