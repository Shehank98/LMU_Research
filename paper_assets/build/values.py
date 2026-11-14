"""Collect every number the paper uses from the result files, so the text cannot drift from the data."""
import os, json, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
M, CF, RES = (os.path.join(ROOT, d) for d in ('metrics', 'confusion', 'paper_assets/xai_results'))
def pct(x, nd=2): return f'{100 * float(x):.{nd}f}'
def j(path): return json.load(open(path))

def load():
    V = {}
    master = pd.read_csv(os.path.join(M, 'results_master.csv')).set_index('model')
    for m, r in master.iterrows():
        V[f'acc.{m}'] = pct(r.accuracy); V[f'f1.{m}'] = f'{r.f1_macro:.3f}'; V[f'prec.{m}'] = f'{r.precision_macro:.3f}'; V[f'rec.{m}'] = f'{r.recall_macro:.3f}'
        for c in ['glioma', 'meningioma', 'notumor', 'pituitary']: V[f'f1c.{m}.{c}'] = f'{r["f1_" + c]:.3f}'
    for f in os.listdir(M):
        if f.endswith('_classification_report.json'):
            m = f.replace('_classification_report.json', ''); d = j(os.path.join(M, f))
            for c in ['glioma', 'meningioma', 'notumor', 'pituitary']:
                V[f'recall.{m}.{c}'] = pct(d[c]['recall']); V[f'precision.{m}.{c}'] = pct(d[c]['precision'])
    cnt = j(os.path.join(M, 'dataset_counts.json'))
    V['n.train'] = f"{sum(cnt['Training'].values()):,}"; V['n.test'] = f"{sum(cnt['Testing'].values()):,}"
    V['n.train.class'] = f"{cnt['Training']['glioma']:,}"; V['n.test.class'] = f"{cnt['Testing']['glioma']:,}"
    for name in ['FINAL_majority_vote', 'CNN_standalone', 'Xception_standalone', 'InceptionV3_standalone']:
        cm = pd.read_csv(os.path.join(CF, name + '_confusion_matrix.csv'), index_col=0)
        for a in cm.index:
            for b in cm.columns: V[f'cm.{name}.{a}.{b}'] = str(int(cm.loc[a, b]))
    eaa = j(os.path.join(M, 'eaa_iou_summary.json'))
    for k in ['mean_eaa_all', 'mean_eaa_correct', 'mean_eaa_misclassified', 'auroc_low_eaa_predicts_error', 'n_misclassified']: V['eaa.' + k] = str(eaa[k])
    for c, v in eaa['mean_eaa_by_true_class'].items(): V['eaa.class.' + c] = f'{v:.4f}'
    V['eaa.p'] = f"{eaa['mann_whitney_p_value']:.1e}".replace('e-', ' × 10<sup>-').replace('e+', ' × 10<sup>') + '</sup>'
    V['eaa.t02.flag'] = str(eaa['threshold_table'][0]['flagged_pct']); V['eaa.t02.caught'] = str(eaa['threshold_table'][0]['errors_caught_pct'])
    st = j(os.path.join(RES, 'results_stats.json'))
    def flat(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict): flat(prefix + k + '.', v)
            elif isinstance(v, list): V[prefix + k] = '–'.join((f'{x:.2f}' if isinstance(x, float) else str(x)) for x in v)
            elif isinstance(v, float) and k in ('acc', 'majority_acc_pct'): V[prefix + k] = f'{v:.2f}'
            else: V[prefix + k] = str(v)
    flat('st.', {k: v for k, v in st.items() if k not in ('mcnemar',)})
    for r in pd.read_csv(os.path.join(RES, 'mcnemar.csv')).to_dict('records'):
        key = f"mc.{r['a']}.{r['b']}"
        V[key + '.n01'] = str(r['a_right_b_wrong']); V[key + '.n10'] = str(r['a_wrong_b_right'])
        V[key + '.p'] = fmt_p(r['p_exact']); V[key + '.ph'] = fmt_p(r['p_holm'])
    for f in ['verification.json', 'final_ensemble_rerun.json', 'shap_rf.json', 'examples.json', 'xai_quantitative.json', 'method_agreement_examples.json']:
        p = os.path.join(RES, f)
        if os.path.exists(p): flat(f.split('.')[0] + '.', j(p) if f != 'examples.json' else {'n': len(j(p)['examples'])})
    q = os.path.join(RES, 'xai_quantitative.json')
    if os.path.exists(q):
        for mdl, lst in j(q).get('sanity_check', {}).items():
            if isinstance(lst, list):
                for i, d in enumerate(lst):
                    V[f'san.{mdl}.{i}.layer'] = (d['randomised_up_to'].split('_')[0] + ' (all layers)') if d['randomised_up_to'].startswith('block') else d['randomised_up_to'] + (' (output layer)' if i == 0 else ''); V[f'san.{mdl}.{i}.rho'] = f"{d['spearman']:.3f}"; V[f'san.{mdl}.{i}.ssim'] = f"{d['ssim']:.3f}"
                V[f'san.{mdl}.last.rho'] = f"{lst[-1]['spearman']:.3f}"; V[f'san.{mdl}.last.ssim'] = f"{lst[-1]['ssim']:.3f}"; V[f'san.{mdl}.first.rho'] = f"{lst[0]['spearman']:.3f}"
    if 'xai_quantitative.consistency.mannwhitney_p_rho' in V: V['cons.p'] = fmt_p(float(V['xai_quantitative.consistency.mannwhitney_p_rho']))
    for m in ['CNN', 'Xception', 'InceptionV3']:
        k = f'xai_quantitative.shortcut.{m}.'
        if k + 'mean_energy_in_rim_pct' in V:
            V[f'sc.{m}.mass'] = f"{float(V[k + 'mean_energy_outside_head_pct']) + float(V[k + 'mean_energy_in_rim_pct']):.2f}"
            V['sc.area'] = f"{float(V[k + 'mean_area_outside_head_pct']) + float(V[k + 'mean_area_rim_pct']):.2f}"
    p = os.path.join(RES, 'deletion_insertion.csv')
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            for k in ['deletion_auc', 'insertion_auc', 'deletion_sd', 'insertion_sd']: V[f'di.{r.model}.{r.method}.{k}'] = f'{r[k]:.3f}'
    return V

def fmt_p(p):
    if p >= 0.001: return f'{p:.3f}'
    s = f'{p:.1e}'; m, e = s.split('e'); return f'{m} × 10<sup>{int(e)}</sup>'

if __name__ == '__main__':
    V = load(); print(len(V))
    for k in sorted(V): print(k, '=', V[k])
