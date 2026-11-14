"""Editable Excel version of every data chart in the paper (native Excel charts linked to the cells).
Blue cells are inputs you can change; black cells are formulas; every chart redraws from its table."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import BarChart, LineChart, ScatterChart, Reference, Series
from openpyxl.chart.error_bar import ErrorBars
from openpyxl.chart.data_source import NumDataSource, NumRef
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter as L
from sklearn.metrics import roc_curve, precision_recall_curve

M, CF, H = [os.path.join(ROOT, d) for d in ('metrics', 'confusion', 'history')]
OUT = os.path.join(ROOT, 'output', 'Paper_Charts_Editable.xlsx')
F = 'Arial'; BLUE = Font(name=F, color='0000FF', size=10); BLACK = Font(name=F, size=10); BOLD = Font(name=F, bold=True, size=10)
TITLE = Font(name=F, bold=True, size=12); HDR_FILL = PatternFill('solid', fgColor='D9E2F3'); NOTE = Font(name=F, italic=True, size=9, color='555555')
thin = Side(style='thin', color='BFBFBF'); BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
wb = Workbook(); ws0 = wb.active; ws0.title = 'README'
# figure numbers exactly as in the built paper
sys.path.insert(0, os.path.join(ROOT, 'paper_assets', 'build'))
import build_paper, engine, values as _v, ch_front, appendix as _ap
_r = engine.Renderer(None, False, {}); _r.number(ch_front.blocks() + build_paper.body() + _ap.appendices(_v.load()))
FN = _r.num['fig']
def fname(*keys): return 'Fig' + '-'.join(str(FN[k]) for k in keys)

def sheet(name, title, note):
    ws = wb.create_sheet(name); ws['A1'] = title; ws['A1'].font = TITLE; ws['A2'] = note; ws['A2'].font = NOTE; return ws
def table(ws, r0, c0, header, rows, inputs=True, fmt=None):
    for j, h in enumerate(header):
        c = ws.cell(r0, c0 + j, h); c.font = BOLD; c.fill = HDR_FILL; c.border = BOX; c.alignment = Alignment(wrap_text=True, vertical='center')
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = ws.cell(r0 + i, c0 + j, v); c.border = BOX
            isf = isinstance(v, str) and v.startswith('=')
            c.font = BLACK if (isf or j == 0 or not inputs) else BLUE
            if fmt and j > 0 and fmt.get(j): c.number_format = fmt[j]
    return r0 + len(rows)
def tidy(ch, title, xt=None, yt=None, w=16, h=8.5):
    ch.title = title; ch.roundedCorners = False; ch.width = w; ch.height = h
    if ch.legend is not None: ch.legend.overlay = False; ch.legend.position = 'b'
    if hasattr(ch, 'x_axis'):
        ch.x_axis.delete = False; ch.y_axis.delete = False
        if xt: ch.x_axis.title = xt
        if yt: ch.y_axis.title = yt
    return ch
def widths(ws, w):
    for i, x in enumerate(w, 1): ws.column_dimensions[L(i)].width = x

# 1 dataset
cnt = json.load(open(os.path.join(M, 'dataset_counts.json')))
ws = sheet(fname('dist') + '_Dataset', 'Figure %d –' % FN['dist'] + ' Images per class', 'Source: metrics/dataset_counts.json. Edit the blue counts; totals and the chart update.')
end = table(ws, 4, 1, ['Class', 'Training', 'Testing'], [[LABELS[i], cnt['Training'][c], cnt['Testing'][c]] for i, c in enumerate(CLASSES)])
ws.cell(end + 1, 1, 'Total').font = BOLD
for j in (2, 3): ws.cell(end + 1, j, f'=SUM({L(j)}5:{L(j)}{end})').font = BOLD
ch = BarChart(); ch.add_data(Reference(ws, min_col=2, max_col=3, min_row=4, max_row=end), titles_from_data=True); ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=end))
ws.add_chart(tidy(ch, 'Images per class', 'Class', 'Images'), 'E4'); widths(ws, [16, 12, 12])

# 2 training curves
for sname, fig, names in [(fname('train_sa') + '_Training', 'Figure %d –' % FN['train_sa'] + ' Training curves of the standalone deep models', ['cnn_model', 'inceptionv3_model', 'xception_model']),
                          (fname('train_ext') + '_Training_Extractors', 'Figure %d –' % FN['train_ext'] + ' Training curves of the 128 px feature extractors', ['cnn_ext', 'inceptionv3_ext', 'xception_ext'])]:
    ws = sheet(sname, fig, 'Source: history/<model>_history.csv. Each block holds one model; edit values to redraw its accuracy and loss charts.')
    col = 1
    for k, nm in enumerate(names):
        h = pd.read_csv(os.path.join(H, nm + '_history.csv'))
        ws.cell(4, col, nm).font = TITLE
        end = table(ws, 5, col, ['Epoch', 'Train acc', 'Val acc', 'Train loss', 'Val loss'],
                    [[int(r.epoch) + 1, round(r.accuracy, 4), round(r.val_accuracy, 4), round(r.loss, 4), round(r.val_loss, 4)] for _, r in h.iterrows()], fmt={1: '0.000', 2: '0.000', 3: '0.000', 4: '0.000'})
        for which, (a, b), yt in [('accuracy', (1, 2), 'Accuracy'), ('loss', (3, 4), 'Loss')]:
            ch = LineChart(); ch.add_data(Reference(ws, min_col=col + a, max_col=col + b, min_row=5, max_row=end), titles_from_data=True)
            ch.set_categories(Reference(ws, min_col=col, min_row=6, max_row=end))
            ws.add_chart(tidy(ch, f'{nm} – {which}', 'Epoch', yt, w=12, h=7), f'{L(col)}{28 if which == "accuracy" else 43}')
        col += 7
    for c in range(1, col): ws.column_dimensions[L(c)].width = 10

# 3 confusion matrices (counts are inputs; row-normalised matrix is formulas + colour scale)
ws = sheet(fname('cm_deep', 'cm_feat', 'cm_raw') + '_Confusion', 'Figures %d, %d, %d –' % (FN['cm_deep'], FN['cm_feat'], FN['cm_raw']) + ' Confusion matrices (counts and row-normalised)', 'Source: confusion/*.csv. Edit the blue counts; the normalised block (formulas) and its colours update.')
models = ['CNN_standalone', 'InceptionV3_standalone', 'Xception_standalone', 'FINAL_majority_vote', 'FINAL_confidence_weighted'] + \
         [f'{a}_feat_{b}' for a in ['cnn', 'xception', 'inceptionv3'] for b in ['rf', 'svm', 'dt', 'softvote_ensemble']] + ['raw_rf', 'raw_svm', 'raw_lr', 'raw_dt', 'raw_softvote_ensemble']
r = 4
for m in models:
    cm = pd.read_csv(os.path.join(CF, m + '_confusion_matrix.csv'), index_col=0)
    ws.cell(r, 1, m).font = TITLE
    table(ws, r + 1, 1, ['True \\ Pred'] + LABELS, [[LABELS[i]] + [int(v) for v in cm.iloc[i].values] for i in range(4)])
    table(ws, r + 1, 7, ['Normalised'] + LABELS, [[LABELS[i]] + [f'=IF(SUM($B{r + 2 + i}:$E{r + 2 + i})=0,0,{L(2 + j)}{r + 2 + i}/SUM($B{r + 2 + i}:$E{r + 2 + i}))' for j in range(4)] for i in range(4)],
          fmt={1: '0.00', 2: '0.00', 3: '0.00', 4: '0.00'})
    ws.conditional_formatting.add(f'H{r + 2}:K{r + 5}', ColorScaleRule(start_type='num', start_value=0, start_color='FFFFFF', end_type='num', end_value=1, end_color='2F5597'))
    ws.cell(r + 6, 1, 'Accuracy').font = BOLD; ws.cell(r + 6, 2, f'=(B{r + 2}+C{r + 3}+D{r + 4}+E{r + 5})/SUM(B{r + 2}:E{r + 5})').number_format = '0.00%'
    r += 9
widths(ws, [24, 11, 12, 11, 11, 3, 12, 11, 12, 11, 11])

# 4 accuracy of all models with Wilson CI (formulas)
master = pd.read_csv(os.path.join(M, 'results_master.csv')).sort_values('accuracy')
ws = sheet(fname('acc_all') + '_Accuracy', 'Figure %d –' % FN['acc_all'] + ' Test accuracy of all configurations with Wilson 95% CI', 'Source: metrics/results_master.csv. Edit Accuracy or n; the CI columns are formulas (Wilson score interval, z = 1.96).')
ws['H4'] = 'z'; ws['I4'] = 1.96; ws['I4'].font = BLUE
rows = []
for i, (_, x) in enumerate(master.iterrows()):
    rr = 5 + i + 1
    lo = f'=((B{rr}+$I$4^2/(2*C{rr}))/(1+$I$4^2/C{rr})-$I$4*SQRT(B{rr}*(1-B{rr})/C{rr}+$I$4^2/(4*C{rr}^2))/(1+$I$4^2/C{rr}))'
    hi = lo.replace('-$I$4*SQRT', '+$I$4*SQRT')
    rows.append([x.model, round(float(x.accuracy), 4), int(x.n_test), lo, hi, f'=B{rr}-D{rr}', f'=E{rr}-B{rr}'])
end = table(ws, 5, 1, ['Model', 'Accuracy', 'n test', 'CI low', 'CI high', 'minus', 'plus'], rows, fmt={1: '0.00%', 3: '0.00%', 4: '0.00%', 5: '0.00%', 6: '0.00%'})
ch = BarChart(); ch.type = 'bar'; ch.add_data(Reference(ws, min_col=2, min_row=5, max_row=end), titles_from_data=True); ch.set_categories(Reference(ws, min_col=1, min_row=6, max_row=end))
s = ch.series[0]; s.errBars = ErrorBars(errDir='x', errBarType='both', errValType='cust', noEndCap=False,
    plus=NumDataSource(numRef=NumRef(f=f"'{ws.title}'!$G$6:$G${end}")), minus=NumDataSource(numRef=NumRef(f=f"'{ws.title}'!$F$6:$F${end}")))
ch.y_axis.scaling.min = 0.7; ch.y_axis.scaling.max = 1.0; ch.y_axis.number_format = '0%'
ws.add_chart(tidy(ch, 'Test accuracy with Wilson 95% CI', None, 'Accuracy', w=20, h=13), 'J4'); widths(ws, [34, 10, 8, 10, 10, 9, 9, 3, 6])

# 5 F1 heatmap
ws = sheet(fname('f1') + '_F1_Heatmap', 'Figure %d –' % FN['f1'] + ' Per-class F1-score of every configuration', 'Source: metrics/results_master.csv. Edit values; the colour scale updates.')
m2 = pd.read_csv(os.path.join(M, 'results_master.csv'))
end = table(ws, 4, 1, ['Model'] + LABELS + ['Macro F1 (formula)'], [[x.model, round(x.f1_glioma, 4), round(x.f1_meningioma, 4), round(x.f1_notumor, 4), round(x.f1_pituitary, 4), f'=AVERAGE(B{5 + i}:E{5 + i})'] for i, (_, x) in enumerate(m2.iterrows())],
            fmt={1: '0.000', 2: '0.000', 3: '0.000', 4: '0.000', 5: '0.000'})
ws.conditional_formatting.add(f'B5:E{end}', ColorScaleRule(start_type='num', start_value=0.65, start_color='440154', mid_type='num', mid_value=0.85, mid_color='21918C', end_type='num', end_value=1, end_color='FDE725'))
widths(ws, [34, 11, 12, 11, 11, 16])

# 6 ROC / PR
P = np.load(os.path.join(RES, 'standalone_test_probs.npz')); yt = P['y']
ws = sheet(fname('roc') + '_ROC_PR', 'Figure %d –' % FN['roc'] + ' ROC and precision–recall curves (one-vs-rest)', 'Computed from the saved models’ test probabilities (paper_assets/xai_results/standalone_test_probs.npz). Curves are thinned to ≤ 120 points for Excel.')
col = 1; anchors = []
for nm in ['CNN', 'InceptionV3', 'Xception']:
    roc_c = ScatterChart(); pr_c = ScatterChart()
    for c in range(4):
        for kind, chart in [('ROC', roc_c), ('PR', pr_c)]:
            if kind == 'ROC': x, y_, _ = roc_curve(yt == c, P[nm][:, c])
            else: y_, x, _ = precision_recall_curve(yt == c, P[nm][:, c])
            idx = np.unique(np.linspace(0, len(x) - 1, min(120, len(x))).astype(int)); x, y_ = x[idx], y_[idx]
            ws.cell(4, col, f'{nm} {LABELS[c]} {kind}').font = BOLD
            table(ws, 5, col, ['FPR' if kind == 'ROC' else 'Recall', 'TPR' if kind == 'ROC' else 'Precision'], [[round(float(a), 4), round(float(b), 4)] for a, b in zip(x, y_)])
            s = Series(Reference(ws, min_col=col + 1, min_row=6, max_row=5 + len(x)), Reference(ws, min_col=col, min_row=6, max_row=5 + len(x)), title=LABELS[c])
            s.marker.symbol = 'none'; s.smooth = False; chart.series.append(s); col += 3
    anchors.append((nm, roc_c, pr_c))
for k, (nm, rc, pc) in enumerate(anchors):
    for c in (rc, pc): c.style = 13; c.x_axis.scaling.min = 0; c.x_axis.scaling.max = 1; c.y_axis.scaling.min = 0; c.y_axis.scaling.max = 1
    ws.add_chart(tidy(rc, f'{nm} ROC', 'False positive rate', 'True positive rate', w=11, h=8), f'A{130 + 17 * k}')
    ws.add_chart(tidy(pc, f'{nm} precision–recall', 'Recall', 'Precision', w=11, h=8), f'J{130 + 17 * k}')

# 7 old vs new
ws = sheet(fname('oldnew') + '_Old_vs_New', 'Figure %d –' % FN['oldnew'] + ' HND (2024) accuracies vs standardised re-run', 'HND values from Final_Research_Report.pdf (the 98.20% was on training images); re-run values from results_master.csv.')
on = [['Custom CNN', 0.94, 0.9044], ['Xception', 0.8691, 0.8762], ['InceptionV3', 0.8463, 0.8662], ['CNN feat. ens.', 0.8371, 0.8931], ['Xception feat. ens.', 0.7028, 0.8831],
      ['InceptionV3 feat. ens.', 0.712, 0.8612], ['Raw-pixel ens.', 0.83, 0.8406], ['Final ensemble', 0.982, 0.925]]
end = table(ws, 4, 1, ['Model', 'HND report (2024)', 'Re-run (2026), held-out', 'Change (formula)'], [r_ + [f'=C{5 + i}-B{5 + i}'] for i, r_ in enumerate(on)], fmt={1: '0.00%', 2: '0.00%', 3: '+0.00%;-0.00%'})
ch = BarChart(); ch.add_data(Reference(ws, min_col=2, max_col=3, min_row=4, max_row=end), titles_from_data=True); ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=end))
ch.y_axis.scaling.min = 0.6; ch.y_axis.number_format = '0%'; ws.add_chart(tidy(ch, 'HND vs re-run accuracy', None, 'Accuracy', w=18), 'F4'); widths(ws, [24, 16, 20, 14])

# 8 EAA-IoU (raw data + formulas)
e = pd.read_csv(os.path.join(M, 'eaa_iou_per_image.csv')).drop_duplicates('idx')
ws = sheet(fname('eaa') + '_EAA_IoU', 'Figure %d –' % FN['eaa'] + ' EAA-IoU on all 1,600 test images', 'Source: metrics/eaa_iou_per_image.csv (columns A–H). Summary cells are formulas; edit the threshold in K15 to test an alert rule.')
table(ws, 4, 1, list(e.columns), e.values.tolist(), inputs=False)
ws['J4'] = 'Summary by true class'; ws['J4'].font = TITLE
table(ws, 5, 10, ['Class', 'Mean EAA-IoU', 'Median', 'n'], [[LABELS[c], f'=AVERAGEIFS($H$5:$H$1604,$B$5:$B$1604,{c})', f'=MEDIAN(IF($B$5:$B$1604={c},$H$5:$H$1604))', f'=COUNTIF($B$5:$B$1604,{c})'] for c in range(4)] +
      [['Correct', '=AVERAGEIFS($H$5:$H$1604,$D$5:$D$1604,1)', '', '=COUNTIF($D$5:$D$1604,1)'], ['Misclassified', '=AVERAGEIFS($H$5:$H$1604,$D$5:$D$1604,0)', '', '=COUNTIF($D$5:$D$1604,0)']], fmt={1: '0.0000', 2: '0.0000'})
for rr in range(6, 10):   # MEDIAN(IF()) must be an array formula
    from openpyxl.worksheet.formula import ArrayFormula
    ws[f'L{rr}'] = ArrayFormula(f'L{rr}', ws[f'L{rr}'].value)
ws['J15'] = 'Alert threshold'; ws['K15'] = 0.10; ws['K15'].font = BLUE
ws['J16'] = 'Images flagged (%)'; ws['K16'] = '=COUNTIF($H$5:$H$1604,"<"&K15)/COUNT($H$5:$H$1604)'; ws['K16'].number_format = '0.00%'
ws['J17'] = 'Errors caught (%)'; ws['K17'] = '=COUNTIFS($H$5:$H$1604,"<"&K15,$D$5:$D$1604,0)/COUNTIF($D$5:$D$1604,0)'; ws['K17'].number_format = '0.00%'
ch = BarChart(); ch.add_data(Reference(ws, min_col=11, min_row=5, max_row=11), titles_from_data=True); ch.set_categories(Reference(ws, min_col=10, min_row=6, max_row=11))
ws.add_chart(tidy(ch, 'Mean EAA-IoU by class and correctness', None, 'Mean EAA-IoU', w=14), 'N4'); widths(ws, [6, 6, 9, 8, 10, 10, 10, 9, 2, 18, 13, 10, 8])

# 9 SHAP on RF
sh = json.load(open(os.path.join(RES, 'shap_rf.json')))
ws = sheet(fname('shap_rf') + '_SHAP_RF', 'Figure %d –' % FN['shap_rf'] + ' TreeSHAP for the Random Forest on CNN features', 'Top-15 units as reported in paper_assets/xai_results/shap_rf.json and fig_shap_rf.png; rerun 07_shap_rf.py for all 256 units.')
ws['A4'] = 'Features carrying 80% of attribution'; ws['B4'] = sh['features_for_80pct']; ws['A5'] = 'Inactive units on test set'; ws['B5'] = sh['dead_or_inactive_features']
ws['A6'] = 'Top-5 units share of attribution (%)'; ws['B6'] = sh['top5_share_pct']
imp = pd.read_csv(os.path.join(RES, 'shap_rf_importance.csv'), index_col=0); imp['tot'] = imp.sum(1); top = imp.sort_values('tot', ascending=False).head(15).iloc[::-1]
end = table(ws, 9, 1, ['Unit'] + LABELS, [[f'unit {u}'] + [round(float(top.loc[u, c]), 5) for c in LABELS] for u in top.index], fmt={j: '0.0000' for j in range(1, 5)})
ch = BarChart(); ch.type = 'bar'; ch.grouping = 'stacked'; ch.overlap = 100
ch.add_data(Reference(ws, min_col=2, max_col=5, min_row=9, max_row=end), titles_from_data=True); ch.set_categories(Reference(ws, min_col=1, min_row=10, max_row=end))
ws.add_chart(tidy(ch, 'Top 15 CNN features for the Random Forest (mean |SHAP|)', None, 'mean |SHAP value|', w=16, h=10), 'H4')
srt = imp.sort_values('tot', ascending=False)
end2 = table(ws, 9, 20, ['Rank', 'Unit', 'Total |SHAP|', 'Cumulative share (formula)'],
             [[i + 1, int(u), round(float(srt.loc[u, 'tot']), 6), f'=SUM($V$10:V{10 + i})/SUM($V$10:$V$265)'] for i, u in enumerate(srt.index)], fmt={2: '0.000000', 3: '0.0%'})
ch = LineChart(); ch.add_data(Reference(ws, min_col=23, min_row=9, max_row=end2), titles_from_data=True); ch.set_categories(Reference(ws, min_col=20, min_row=10, max_row=end2))
ws.add_chart(tidy(ch, 'Cumulative share of attribution', 'Features (ranked)', 'Share', w=14, h=8), 'H26')
widths(ws, [38, 12, 12, 12, 12])

# 10 quantitative XAI (if available)
qpath = os.path.join(RES, 'xai_quantitative.json')
if os.path.exists(qpath):
    q = json.load(open(qpath)); di = pd.read_csv(os.path.join(RES, 'deletion_insertion.csv')); cur = np.load(os.path.join(RES, 'deletion_insertion_curves.npz'))
    ws = sheet(fname('di') + '_Deletion_Insertion', 'Figure %d –' % FN['di'] + ' Deletion and insertion curves', 'Source: deletion_insertion_curves.npz. AUC row = trapezoid rule formula over the 21 points (step 0.05).')
    col = 1
    for m in ['CNN', 'Xception', 'InceptionV3']:
        for kind in ['deletion', 'insertion']:
            ws.cell(4, col, f'{m} {kind}').font = TITLE
            meths = ['gradcam', 'gradcampp', 'hirescam', 'random']
            end = table(ws, 5, col, ['Fraction'] + [f'{x}' for x in meths], [[round(t / 20, 2)] + [round(float(cur[f'{m}_{x}_{kind}'][t]), 4) for x in meths] for t in range(21)], fmt={j: '0.000' for j in range(1, 5)})
            ws.cell(end + 1, col, 'AUC').font = BOLD
            for j in range(1, 5):
                cl = L(col + j); c = ws.cell(end + 1, col + j, f'=0.05*(SUM({cl}6:{cl}{end})-({cl}6+{cl}{end})/2)'); c.number_format = '0.000'; c.font = BOLD
            ch = LineChart(); ch.add_data(Reference(ws, min_col=col + 1, max_col=col + 4, min_row=5, max_row=end), titles_from_data=True); ch.set_categories(Reference(ws, min_col=col, min_row=6, max_row=end))
            ws.add_chart(tidy(ch, f'{m} {kind}', 'Fraction of pixels', 'Class probability', w=11, h=7), f'{L(col)}30')
            col += 7
    ws = sheet(fname('san') + '_Sanity_Check', 'Figure %d –' % FN['san'] + ' Cascading weight randomisation', 'Source: xai_quantitative.json → sanity_check.')
    r0 = 4
    for m in ['CNN', 'Xception']:
        s = q['sanity_check'][m]
        end = table(ws, r0, 1, [f'{m}: randomised down to', 'Spearman rho', 'SSIM'], [[d['randomised_up_to'], d['spearman'], d['ssim']] for d in s], fmt={1: '0.000', 2: '0.000'})
        ch = LineChart(); ch.add_data(Reference(ws, min_col=2, max_col=3, min_row=r0, max_row=end), titles_from_data=True); ch.set_categories(Reference(ws, min_col=1, min_row=r0 + 1, max_row=end))
        ws.add_chart(tidy(ch, f'{m}: similarity to original Grad-CAM', 'Layers randomised', 'Similarity', w=13, h=7), f'F{r0}'); r0 = max(end + 3, r0 + 16)
    widths(ws, [30, 13, 10])
    ws = sheet(fname('cons') + '_Consistency', 'Figure %d –' % FN['cons'] + ' Cross-model agreement and attention outside the brain', 'Source: xai_quantitative.json.')
    cs = q['consistency']
    end = table(ws, 4, 1, ['Class', 'Mean Spearman rho', 'Mean top-20% IoU'], [[c, cs['by_true_class_mean_rho'][c], cs['by_true_class_mean_iou20'][c]] for c in CLASSES], fmt={1: '0.000', 2: '0.000'})
    ch = BarChart(); ch.add_data(Reference(ws, min_col=2, max_col=3, min_row=4, max_row=end), titles_from_data=True); ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=end))
    ws.add_chart(tidy(ch, 'Agreement between models by true class', None, 'Agreement', w=13, h=7), 'F4')
    sc = q['shortcut']
    end = table(ws, 20, 1, ['Model', 'Grad-CAM mass outside head or rim (%)', 'Image area outside head or rim (%)'],
                [[m, round(sc[m]['mean_energy_outside_head_pct'] + sc[m]['mean_energy_in_rim_pct'], 2), round(sc[m]['mean_area_outside_head_pct'] + sc[m]['mean_area_rim_pct'], 2)] for m in ['CNN', 'Xception', 'InceptionV3']], fmt={1: '0.00', 2: '0.00'})
    ch = BarChart(); ch.add_data(Reference(ws, min_col=2, max_col=3, min_row=20, max_row=end), titles_from_data=True); ch.set_categories(Reference(ws, min_col=1, min_row=21, max_row=end))
    ws.add_chart(tidy(ch, 'Attention outside the brain', None, '%', w=13, h=7), 'F20'); widths(ws, [16, 22, 22])

# 11 McNemar with formulas
mc = pd.read_csv(os.path.join(RES, 'mcnemar.csv'))
ws = sheet('Table_McNemar', 'Exact McNemar tests (Chapter 4)', 'Edit n01 / n10; the exact two-sided p-value is recomputed with BINOMDIST. Holm adjustment is in the paper (Table 9).')
table(ws, 4, 1, ['Model A', 'Model B', 'n01 (A right, B wrong)', 'n10 (A wrong, B right)', 'Exact p (formula)'],
      [[x.a, x.b, int(x.a_right_b_wrong), int(x.a_wrong_b_right), f'=MIN(1,2*BINOMDIST(MIN(C{5 + i},D{5 + i}),C{5 + i}+D{5 + i},0.5,TRUE))'] for i, (_, x) in enumerate(mc.iterrows())], fmt={4: '0.00E+00'})
for rr in range(5, 5 + len(mc)): ws.cell(rr, 2).font = BLACK
widths(ws, [18, 26, 20, 20, 16])

# README
ws0['A1'] = 'Editable charts for the research paper'; ws0['A1'].font = TITLE
lines = ['Each sheet holds the data behind one chart of the paper and a native Excel chart drawn from it.',
         'Blue text = input values you can change. Black text = formulas or labels. Charts redraw automatically.',
         'Sheet names start with the figure number used in the paper (numbers refer to the final paper build).',
         'The figures inside the Word document are 300-dpi images from paper_assets/figures/; changing this workbook does not change them.',
         'To put an edited chart in Word: select the chart → Copy → in Word, Paste (or Paste Special → Picture).',
         'Architecture and pipeline diagrams (CNN, ensemble, XAI pipeline, conceptual framework) and the MRI heat-map panels are not data charts, so they are not included here.',
         'Sources: metrics/, confusion/, history/ (logged training run) and paper_assets/xai_results/ (analysis for this report).']
for i, t in enumerate(lines): ws0.cell(3 + i, 1, t).font = BLACK
ws0.cell(11, 1, 'Sheet').font = BOLD; ws0.cell(11, 2, 'Contents').font = BOLD
for i, s in enumerate(wb.sheetnames[1:]): ws0.cell(12 + i, 1, s).font = BLACK; ws0.cell(12 + i, 2, wb[s]['A1'].value).font = BLACK
ws0.column_dimensions['A'].width = 28; ws0.column_dimensions['B'].width = 90
for s in wb.worksheets: s.sheet_view.showGridLines = True
import re as _re
def _k(ws): return (0, 0) if ws.title == 'README' else (2, 0) if ws.title.startswith('Table') else (1, int(_re.findall(r'\d+', ws.title)[0]))
wb._sheets.sort(key=_k)
for i, s_ in enumerate(wb.sheetnames[1:]): ws0.cell(12 + i, 1, s_); ws0.cell(12 + i, 2, wb[s_]['A1'].value)
wb.save(OUT); print('saved', OUT, wb.sheetnames)
