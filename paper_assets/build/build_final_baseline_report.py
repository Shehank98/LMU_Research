"""Build the final report (baseline scope) from output/final_baseline_report/*.md:
references from refs.py (only the cited ones), pandoc to .docx, then house style
(Times New Roman 12, 1.5 spacing, bordered tables, page numbers, title page)."""

import glob, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(__file__))
import refs
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "output", "final_baseline_report")
OUT = os.path.join(SRC, "Final_Project_Report_E187041.docx")

md = "\n\n".join(open(f).read() for f in sorted(glob.glob(os.path.join(SRC, "0*.md"))))
cited = []
for k, r in refs.REF.items():
    if re.search(re.escape(r["surname"]) + r"[^()\n]{0,80}?[ (,]" + re.escape(str(r["year"])), md):
        cited.append(r["text"])


def to_md(t):
    t = t.replace("<i>", "*").replace("</i>", "*")
    return re.sub(r"(doi:)(\S+?)\.$", r"\1\2.", t)


ref_md = "\n\n# References {.unnumbered}\n\n" + "\n\n".join(
    to_md(t) for t in sorted(cited, key=lambda s: s.lower())
)
full = os.path.join(SRC, "_full.md")
open(full, "w").write(md + ref_md)
subprocess.run(
    [
        "pandoc",
        full,
        "-o",
        OUT,
        "--toc",
        "--toc-depth=2",
        "--resource-path",
        ROOT,
        "-f",
        "markdown+tex_math_dollars",
    ],
    check=True,
)

doc = Document(OUT)
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.54)
    s.top_margin = s.bottom_margin = Cm(2.54)

BLACK = RGBColor(0, 0, 0)
STYLES = {st.name: st for st in doc.styles}
for name in [
    "Normal",
    "Body Text",
    "First Paragraph",
    "Compact",
    "Table Caption",
    "Image Caption",
    "Caption",
    "Title",
    "Subtitle",
    "Author",
    "Date",
    "Abstract",
    "Bibliography",
    "TOC Heading",
]:
    st = STYLES.get(name)
    if st is None:
        continue
    st.font.name = "Times New Roman"
    (
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        if st.element.rPr is not None and st.element.rPr.rFonts is not None
        else None
    )
    st.font.color.rgb = BLACK
    if name in ("Normal", "Body Text", "First Paragraph"):
        st.font.size = Pt(12)
        st.paragraph_format.line_spacing = 1.5
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if name == "Compact":
        st.font.size = Pt(10)
    if name in ("Table Caption", "Image Caption", "Caption"):
        st.font.size = Pt(10)
        st.font.italic = True
for lvl, size in [(1, 16), (2, 13), (3, 12)]:
    st = STYLES[f"Heading {lvl}"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = BLACK
    rpr = st.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(a), "Times New Roman")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if fonts.get(qn(a)) is not None:
            del fonts.attrib[qn(a)]
    st.paragraph_format.space_before = Pt(12)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True

sc = STYLES.get("Source Code")
if sc is not None:
    sc.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    sc.paragraph_format.line_spacing = 1.0
    sc.font.size = Pt(8.5)
# chapters, abstract, abbreviations, references start on a new page
for p in doc.paragraphs:
    if p.style.name == "Heading 1":
        p.paragraph_format.page_break_before = True
    if p.style.name == "Source Code":
        p.paragraph_format.line_spacing = 1.0
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.keep_together = True
        for r in p.runs:
            r.font.size = Pt(8.5)
            r.font.name = "Consolas"

# title page: centre and add a page break after the date
for p in doc.paragraphs[:8]:
    if p.style.name in ("Title", "Subtitle", "Author", "Date"):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.color.rgb = BLACK
        if p.style.name == "Title":
            p.paragraph_format.space_before = Pt(120)
            for r in p.runs:
                r.font.size = Pt(18)
                r.bold = True
    if p.style.name == "Date":
        p.add_run().add_break(WD_BREAK.PAGE)

# tables: borders, 10 pt, single spacing
for t in doc.tables:
    tblPr = t._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "808080")
        b.append(e)
    tblPr.append(b)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_after = Pt(2)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.size = Pt(10)
                    r.font.name = "Times New Roman"

# page numbers in the footer
for s in doc.sections:
    fp = s.footer.paragraphs[0] if s.footer.paragraphs else s.footer.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.add_run("P. S. Kavishka (E187041) | CS6P05ES Final Project Report | Page ")
    r = fp.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            f = OxmlElement("w:fldChar")
            f.set(qn("w:fldCharType"), kind)
            r._r.append(f)
        else:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = text
            r._r.append(it)
    for rr in fp.runs:
        rr.font.size = Pt(9)
        rr.font.name = "Times New Roman"
doc.save(OUT)
os.remove(full)
print("written", OUT, "references:", len(cited))
