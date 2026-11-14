"""Build the highlighted and clean versions of the paper.  Run from the repository root:
    python paper_assets/build/build_paper.py
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(__file__))
from engine import *
import values, refs, ch_front, ch1_2, ch3, ch4, ch5, ch6_8, appendix

ROOT = values.ROOT; os.chdir(ROOT)
OUT = os.path.join(ROOT, 'output'); os.makedirs(OUT, exist_ok=True)
FOOTER = 'P. S. Kavishka (E187041)  |  CS6P05ES Final Project Report  |  Page'

def references(doc, r):
    for key, ref in refs.sorted_refs():
        p = doc.add_paragraph(); pf = p.paragraph_format
        pf.left_indent = Cm(1.0); pf.first_line_indent = Cm(-1.0); pf.line_spacing = 1.15; pf.space_after = Pt(6)
        add_runs(p, ref['text'], size=11)

def body():
    return ch1_2.ch1() + ch1_2.ch2() + ch3.ch3() + ch4.ch4() + ch5.ch5() + ch6_8.ch6() + ch6_8.ch7() + ch6_8.ch8()

def build(highlight, path):
    V = values.load()
    doc = new_document(FOOTER)
    front = ch_front.blocks()
    rest = [RAW(lambda d, r: start_main_section(d))] + body() + [H(1, 'References'), RAW(references)] + appendix.appendices(V)
    r = Renderer(doc, highlight, V)
    r.render(front + rest)
    doc.core_properties.title = ch_front.TITLE; doc.core_properties.author = ch_front.STUDENT
    finalize(doc); doc.save(path); return r

if __name__ == '__main__':
    build(True, os.path.join(OUT, 'Brain_Tumor_XAI_Research_Paper.docx'))
    build(False, os.path.join(OUT, 'Brain_Tumor_XAI_Research_Paper_clean.docx'))
    print('built')
