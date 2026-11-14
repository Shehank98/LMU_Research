"""Harvard reference list. DOI entries are generated from the verified Crossref records; others were verified by web search."""
import json, os, re, html
META = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'references', 'crossref_metadata.json')))
ACCESSED = '14 November 2026'
def _clean(s): return re.sub(r'\s+', ' ', html.unescape(re.sub('<[^>]+>', '', s or ''))).strip()
def _initials(given):
    parts = re.split(r'[\s]+', given.replace('.', ' ').strip()); out = []
    for p in parts:
        if not p: continue
        out.append('-'.join(q[0].upper() + '.' for q in p.split('-') if q))
    return ''.join(out)
def _authors(auth):
    names = []
    for a in auth:
        fam = _clean(a.get('family') or a.get('name') or ''); giv = a.get('given') or ''
        names.append(f"{fam}, {_initials(giv)}" if giv else fam)
    if len(names) > 6: return names[0] + ' et al.'
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]
def from_doi(doi, title_case=None, container=None):
    m = META[doi]; y = m['issued']['date-parts'][0][0]; t = _clean(m['title'][0]) if not title_case else title_case
    cont = container or _clean((m.get('container-title') or [''])[0]); typ = m['type']
    vol, iss, pg, art = m.get('volume'), m.get('issue'), m.get('page'), m.get('article-number')
    s = f"{_authors(m['author'])} ({y}) '{t}', "
    if typ in ('journal-article',):
        s += f"<i>{cont}</i>"
        if vol: s += f", {vol}"
        if iss: s += f"({iss})"
        if pg and '-' in pg: s += f", pp. {pg.replace('-', '–')}"
        elif pg or art: s += f", {pg or art}"
    else:
        s += f"in <i>{cont}</i>"
        if pg: s += f", pp. {pg.replace('-', '–')}"
    return s + f". doi:{doi}.", m, y
REF = {}   # key -> dict(surname, year, text)
def add(key, surname, year, text): REF[key] = dict(surname=surname, year=str(year), text=text)
DOIS = {
 'Abiwinanda2018': ('10.1007/978-981-10-9035-6_33', 'World Congress on Medical Physics and Biomedical Engineering 2018 (IFMBE Proceedings, vol. 68/1)'),
 'Cheng2015': ('10.1371/journal.pone.0140381', None), 'Cortes1995': ('10.1007/BF00994018', None), 'Ismael2018': ('10.1109/EIT.2018.8500308', None),
 'Talo2019': ('10.1016/j.cogsys.2018.12.007', None), 'Minz2017': ('10.1109/IACC.2017.0146', None), 'Patel2019': ('10.1109/ACCESS.2019.2927792', None),
 'Sultan2019': ('10.1109/ACCESS.2019.2919122', None), 'Tripathi2019': ('10.1007/978-981-13-9042-5_45', 'Computational Intelligence in Pattern Recognition (Advances in Intelligent Systems and Computing, vol. 999)'),
 'Yousaf2023': ('10.1016/j.bspc.2023.104875', None), 'Zhou2019': ('10.1007/978-3-030-11723-8_21', 'Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2018), Lecture Notes in Computer Science, vol. 11383'),
 'Selvaraju2017': ('10.1109/ICCV.2017.74', None), 'Chattopadhay2018': ('10.1109/WACV.2018.00097', None), 'Ribeiro2016': ('10.1145/2939672.2939778', None),
 'Chollet2017': ('10.1109/CVPR.2017.195', None), 'Szegedy2016': ('10.1109/CVPR.2016.308', None),
 'Dietterich2000': ('10.1007/3-540-45014-9_1', 'Multiple Classifier Systems (MCS 2000), Lecture Notes in Computer Science, vol. 1857'),
 'Rudin2019': ('10.1038/s42256-019-0048-x', None), 'Zeiler2014': ('10.1007/978-3-319-10590-1_53', 'Computer Vision – ECCV 2014, Lecture Notes in Computer Science, vol. 8689'),
 'Breiman2001': ('10.1023/A:1010933404324', None), 'Quinlan1986': ('10.1007/BF00116251', None), 'McNemar1947': ('10.1007/BF02295996', None),
 'Dietterich1998': ('10.1162/089976698300017197', None), 'Efron1979': ('10.1214/aos/1176344552', None), 'LeCun2015': ('10.1038/nature14539', None),
 'Litjens2017': ('10.1016/j.media.2017.07.005', None), 'Deng2009': ('10.1109/CVPR.2009.5206848', None), 'Geirhos2020': ('10.1038/s42256-020-00257-z', None),
 'Zech2018': ('10.1371/journal.pmed.1002683', None), 'Louis2021': ('10.1093/neuonc/noab106', None), 'BarredoArrieta2020': ('10.1016/j.inffus.2019.12.012', None),
 'vanderVelden2022': ('10.1016/j.media.2022.102470', None), 'Salahuddin2022': ('10.1016/j.compbiomed.2021.105111', None), 'Tjoa2021': ('10.1109/TNNLS.2020.3027314', None),
 'Ghassemi2021': ('10.1016/S2589-7500(21)00208-9', None), 'Saporta2022': ('10.1038/s42256-022-00536-x', None), 'Arun2021': ('10.1148/ryai.2021200267', None),
 'Swati2019': ('10.1016/j.compmedimag.2019.05.001', None), 'Deepak2019': ('10.1016/j.compbiomed.2019.103345', None), 'Rehman2020': ('10.1007/s00034-019-01246-3', None),
 'Badza2020': ('10.3390/app10061999', None), 'Nhlapho2024': ('10.3390/info15040182', None), 'Zeineldin2022': ('10.1007/s11548-022-02619-x', None),
 'Esmaeili2021': ('10.3390/jpm11111213', None), 'Ganaie2022': ('10.1016/j.engappai.2022.105151', None), 'Kang2021': ('10.3390/s21062222', None),
 'Saeedi2023': ('10.1186/s12911-023-02114-6', None), 'Topol2019': ('10.1038/s41591-018-0300-7', None), 'Kelly2019': ('10.1186/s12916-019-1426-2', None),
 'Montavon2017': ('10.1016/j.patcog.2016.11.008', None), 'Shin2016': ('10.1109/TMI.2016.2528162', None), 'Kakon2025': ('10.3390/cancers17172853', None),
 'Aasem2024': ('10.3389/fdata.2024.1366415', None)}
for key, (doi, cont) in DOIS.items():
    text, m, y = from_doi(doi, container=cont)
    fam = _clean(m['author'][0].get('family'))
    if key == 'BarredoArrieta2020': fam = 'Barredo Arrieta'
    if key == 'vanderVelden2022': fam = 'van der Velden'
    if key == 'Rehman2020': y = 2020; text = text.replace('(2019)', '(2020)').replace("Systems, and Signal Processing</i>", "Systems, and Signal Processing</i>, 39(2), pp. 757–775") if '39' not in text else text.replace('(2019)', '(2020)')
    text = text.replace("'\"Why Should I Trust You?\"'", "'\"Why should I trust you?\": explaining the predictions of any classifier'")
    text = text.replace('Kai Li and Li Fei-Fei', 'Li, K. and Fei-Fei, L.').replace('R. Niakan Kalhori, S.', 'Niakan Kalhori, S.R.')
    add(key, fam, y, text)
add('Lundberg2017', 'Lundberg', 2017, "Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', in <i>Advances in Neural Information Processing Systems 30 (NIPS 2017)</i>, pp. 4765–4774. Available at: https://papers.nips.cc/paper/7062 (Accessed: " + ACCESSED + ").")
add('Adebayo2018', 'Adebayo', 2018, "Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M. and Kim, B. (2018) 'Sanity checks for saliency maps', in <i>Advances in Neural Information Processing Systems 31 (NeurIPS 2018)</i>, pp. 9505–9515. Available at: https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps (Accessed: " + ACCESSED + ").")
add('Petsiuk2018', 'Petsiuk', 2018, "Petsiuk, V., Das, A. and Saenko, K. (2018) 'RISE: randomized input sampling for explanation of black-box models', in <i>Proceedings of the British Machine Vision Conference (BMVC 2018)</i>. Available at: https://arxiv.org/abs/1806.07421 (Accessed: " + ACCESSED + ").")
add('Draelos2020', 'Draelos', 2020, "Draelos, R.L. and Carin, L. (2020) 'Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks', <i>arXiv preprint</i>, arXiv:2011.08891. Available at: https://arxiv.org/abs/2011.08891 (Accessed: " + ACCESSED + ").")
add('Pedregosa2011', 'Pedregosa', 2011, "Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', <i>Journal of Machine Learning Research</i>, 12(85), pp. 2825–2830. Available at: https://jmlr.org/papers/v12/pedregosa11a.html (Accessed: " + ACCESSED + ").")
add('Abadi2016', 'Abadi', 2016, "Abadi, M. et al. (2016) 'TensorFlow: a system for large-scale machine learning', in <i>12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)</i>. Savannah, GA: USENIX Association, pp. 265–283. Available at: https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi (Accessed: " + ACCESSED + ").")
add('Nickparvar', 'Nickparvar', 'n.d.', "Nickparvar, M. (n.d.) <i>Brain Tumor MRI Dataset</i> [Dataset]. Kaggle. Available at: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset (Accessed: " + ACCESSED + ").")
add('BJyotibrat2026', 'BJyotibrat', 2026, "BJyotibrat (2026) <i>Masoud-Nickparvar-Brain-Tumor-MRI-Dataset</i> [Dataset mirror]. Hugging Face. Available at: https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset (Accessed: " + ACCESSED + ").")
add('Kavishka2026', 'Kavishka', 2026, "Kavishka, P.S. (2026) <i>brain-tumor-mri-models: trained models and logged results</i> [Model repository]. Hugging Face. Available at: https://huggingface.co/shehank98/brain-tumor-mri-models (Accessed: " + ACCESSED + ").")
def sorted_refs(): return sorted(REF.items(), key=lambda kv: (kv[1]['surname'].lower().replace('van der ', 'v'), str(kv[1]['year'])))
if __name__ == '__main__':
    for k, v in sorted_refs(): print(k, '|', re.sub('<[^>]+>', '', v['text']))
    print(len(REF))
