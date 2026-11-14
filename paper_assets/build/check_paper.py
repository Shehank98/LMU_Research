"""Final checks on the built paper:
 1. every decimal / percentage in the text is backed by a result file, the old report, or a cited source;
 2. every figure, table and listing is referred to in the text;
 3. every in-text citation is in the reference list and every reference is cited;
 4. no placeholders or TODOs remain;
 5. body word count.
Usage: python paper_assets/build/check_paper.py output/Brain_Tumor_XAI_Research_Paper_clean.docx
"""
import os, sys, re, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from docx import Document
import values, refs

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(values.ROOT, 'output', 'Brain_Tumor_XAI_Research_Paper_clean.docx')
doc = Document(path)
paras = [p.text for p in doc.paragraphs]
cells = [c.text for t in doc.tables for row in t.rows for c in row.cells]
num_cells = [c.text for t in doc.tables if 'Section' not in ' '.join(x.text for x in t.rows[0].cells) for row in t.rows for c in row.cells]
alltext = '\n'.join(paras + cells)
ok = True
def fail(msg):
    global ok; ok = False; print('  FAIL', msg)

# ---------- 1. numbers
allowed = set()
def add_num(x):
    try: f = float(str(x).replace(',', ''))
    except ValueError: return
    for v in (f, 100 * f):
        for nd in range(0, 5): allowed.add(f'{round(v, nd):.{nd}f}')
for k, v in values.load().items():
    for tok in re.findall(r'\d+(?:\.\d+)?', str(v).replace(',', '')): add_num(tok)
for f in glob.glob(os.path.join(values.ROOT, 'metrics', '*')) + glob.glob(os.path.join(values.ROOT, 'confusion', '*')) + \
         glob.glob(os.path.join(values.ROOT, 'history', '*')) + glob.glob(os.path.join(values.RES, '*.json')) + glob.glob(os.path.join(values.RES, '*.csv')):
    for tok in re.findall(r'-?\d+\.\d+(?:e-?\d+)?|\b\d+\b', open(f, errors='ignore').read()): add_num(tok)
OLD_PDF = ['98.20', '94', '86.91', '84.63', '83.71', '70.28', '71.20', '83', '88.26', '80.02', '79.52', '85.45', '80.99', '66.60', '56.71', '68.20',
           '73.67', '57.34', '68.05', '74.64', '9,047', '6,984', '2,063', '6,985', '1,733', '1,699', '1,847', '1,705', '1,706', '1688', '1636', '1837', '1698']
LITERATURE = ['89.90', '91.28', '92.13', '98', '98.69', '100', '94.68']        # values stated in the cited papers (Cheng 2015 91.28; Deepak 98; Rehman 98.69)
DERIVED = {'5.70': '98.20 - 92.50', '7.44': '89.69 - 82.25', '2.06': '92.50 - 90.44', '0.24': '86.62 - 86.38', '22.5': '90 / 400 gliomas',
           '25.75': '103 / 400', '0.40': 'planned EAA threshold', '0.65': 'planned EAA threshold', '0.05': 'threshold', '0.10': 'threshold', '0.5': 'binarisation / dropout',
           '0.001': 'Adam default', '1.5': 'line spacing', '5812.1': 'results_master', '2.7': 'results_master', '0.2': 'LIME ratio', '0.8': 'quantile', '0.9': 'split', '65,536': '256 x 256 pixels'}
for x in OLD_PDF + LITERATURE + list(DERIVED): add_num(x)
ref_i = [i for i, p in enumerate(paras) if p.strip() == 'References'][-1]
apx_i = [i for i, p in enumerate(paras) if p.strip() == 'Appendices'][-1]
numtext = '\n'.join(p for p in paras[:ref_i] + paras[apx_i:] if not re.match(r'^(Chapter \d|\d+(\.\d+)+ |Appendix [A-H])', p)) + '\n' + '\n'.join(num_cells)
numtext = re.sub(r'(Section|Sections|Chapter|Ch\.|§)\s?\d+(\.\d+)*(–\d+(\.\d+)*)?', ' ', numtext)
numtext = re.sub(r'\b\d+\.\d+(\.\d+)?(, \d+\.\d+)*(–\d+\.\d+)?(?= \(|,| and)', lambda m: m.group(0) if '%' in m.group(0) else m.group(0), numtext)
numtext = re.sub(r'(https?://\S+|doi:\S+|arXiv:\S+)', ' ', numtext)
body_numbers = re.findall(r'(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+)(?!\d)', numtext)
bad = sorted({n for n in body_numbers if n.replace(',', '') not in allowed and n not in allowed and f'{float(n.replace(",", "")):.2f}' not in allowed})
bad = [b for b in bad if not re.fullmatch(r'\d\.\d', b) or b not in ('1.0', '2.0')]
print(f'[1] numbers: {len(set(body_numbers))} distinct decimals / thousands checked')
for b in bad:
    ctx = re.search(r'.{0,60}' + re.escape(b) + r'.{0,40}', numtext)
    fail(f'unbacked number {b}: …{ctx.group(0) if ctx else ""}…')

# ---------- 2. figures, tables, listings referenced
caps = [p.text for p in doc.paragraphs if p.style.name == 'Caption']
noncap = '\n'.join([p.text for p in doc.paragraphs if p.style.name != 'Caption'] + cells)
for kind in ('Figure', 'Table', 'Listing'):
    nums = [int(m.group(1)) for c in caps for m in [re.match(kind + r' (\d+)\.', c)] if m]
    unref = [n for n in nums if not re.search(rf'\b{kind} {n}\b', noncap)]
    print(f'[2] {kind}s: {len(nums)} captions, {len(nums) - len(unref)} referenced in text')
    for n in unref: fail(f'{kind} {n} is not referenced in the text')

# ---------- 3. citations
body_end = alltext.find('\nReferences\n')
body = '\n'.join(paras[:[i for i, p in enumerate(paras) if p.strip() == 'References'][-1]]) + '\n' + '\n'.join(cells)
cited = set()
for key, r in refs.REF.items():
    sur = re.escape(r['surname']); yr = re.escape(str(r['year']))
    if re.search(sur + r"[^()\n]{0,80}?[ (]" + yr, body): cited.add(key)
    else: fail(f'reference not cited in text: {key}')
pat = re.compile(r"([A-Z][A-Za-zÀ-ž'\-]+(?: (?:der|van|de) [A-Z][\w-]+)?)(?: et al\.|,? and [A-Z][\w\-]+(?: [A-Z][\w\-]+)?|, [A-Z][\w\-]+ and [A-Z][\w\-]+)?(?:,| \()\s?(\d{4}|n\.d\.)\)?")
surnames = {r['surname'].split()[-1] for r in refs.REF.values()} | {r['surname'] for r in refs.REF.values()}
IGNORE = {'Re-run', 'HND', 'Unit', 'Pearson', 'Report', 'November', 'October', 'Colab', 'Submission', 'Brain', 'Figure', 'Table', 'Section', 'Signature', 'Date', 'Kaggle'}
unknown = set()
for m in pat.finditer(body):
    s = m.group(1)
    if s in IGNORE: continue
    if s not in surnames and s.split()[-1] not in surnames: unknown.add(m.group(0)[:60])
print(f'[3] references: {len(refs.REF)} in list, {len(cited)} cited; unmatched citation-like strings: {len(unknown)}')
for u in sorted(unknown): fail(f'citation without reference? {u}')

# ---------- 4. placeholders
for tok in ['TODO', 'TBD', 'XXX', 'lorem', '<<', '>>', '{fig:', '{tab:', '{lst:', 'stub', '[citation']:
    if tok.lower() in alltext.lower(): fail(f'placeholder text found: {tok}')
print('[4] placeholders checked')

# ---------- 5. word count (Chapter 1 .. before References)
start = [i for i, p in enumerate(paras) if p.startswith('Chapter 1')][0]
end = [i for i, p in enumerate(paras) if p.strip() == 'References'][-1]
words = sum(len(p.split()) for p in paras[start:end] if not re.match(r'^(def |    |#|\s)', p))
abstract = next(paras[i + 1] for i, p in enumerate(paras) if p.strip().startswith('Abstract'))
print(f'[5] body words (Ch.1–8, excluding code and tables): {words}; abstract words: {len(abstract.split())}')
if words < 6000: fail('body shorter than 6,000 words')
if not 250 <= len(abstract.split()) <= 300: fail('abstract not 250–300 words')
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
