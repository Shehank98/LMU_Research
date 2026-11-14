"""Look up every DOI on Crossref and store the authoritative metadata used for the reference list."""
import json, urllib.request, urllib.parse, sys, os, time
DOIS = """10.1007/978-981-10-9035-6_33
10.1371/journal.pone.0140381
10.1007/BF00994018
10.1109/EIT.2018.8500308
10.1016/j.cogsys.2018.12.007
10.1109/IACC.2017.0146
10.1109/ACCESS.2019.2927792
10.1109/ACCESS.2019.2919122
10.1016/j.compbiomed.2019.103551
10.1016/j.bspc.2023.104875
10.1007/978-3-030-11723-8_21
10.1109/ICCV.2017.74
10.1109/WACV.2018.00097
10.1145/2939672.2939778
10.1109/CVPR.2017.195
10.1109/CVPR.2016.308
10.1007/3-540-45014-9_1
10.1038/s42256-019-0048-x
10.34740/KAGGLE/DSV/2645886
10.1007/978-3-319-10590-1_53
10.1023/A:1010933404324
10.1007/BF00116251
10.1007/BF02295996
10.1162/089976698300017197
10.1214/aos/1176344552
10.1038/nature14539
10.1016/j.media.2017.07.005
10.1109/CVPR.2009.5206848
10.1038/s42256-020-00257-z
10.1371/journal.pmed.1002683
10.1093/neuonc/noab106
10.1016/j.inffus.2019.12.012
10.1016/j.media.2022.102470
10.1016/j.compbiomed.2021.105111
10.1109/TNNLS.2020.3027314
10.1016/S2589-7500(21)00208-9
10.1038/s42256-022-00536-x
10.1148/ryai.2021200267
10.1016/j.compmedimag.2019.05.001
10.1016/j.compbiomed.2019.103345
10.1007/s00034-019-01246-3
10.3390/app10061999
10.3390/info15040182
10.1007/s11548-022-02619-x
10.3390/jpm11111213
10.1016/j.engappai.2022.105151
10.3390/s21062222
10.1186/s12911-023-02114-6
10.1038/s41591-018-0300-7
10.1186/s12916-019-1426-2
10.1016/j.patcog.2016.11.008
10.1109/TMI.2016.2528162""".split()
out = {}
for d in DOIS:
    try:
        with urllib.request.urlopen('https://api.crossref.org/works/' + urllib.parse.quote(d), timeout=30) as r:
            m = json.load(r)['message']
        out[d] = {k: m.get(k) for k in ['author', 'title', 'container-title', 'volume', 'issue', 'page', 'issued', 'publisher', 'type', 'article-number', 'event']}
        a = m.get('author') or []
        print('OK ', d, '|', (a[0].get('family') if a else m.get('publisher')), m['issued']['date-parts'][0][0], '|', (m.get('title') or [''])[0][:90], '|', (m.get('container-title') or [''])[0][:60])
    except Exception as e:
        out[d] = None; print('ERR', d, e)
    time.sleep(0.2)
json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'references', 'crossref_metadata.json'), 'w'), indent=1)
