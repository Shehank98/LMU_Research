"""Small document engine on top of python-docx.

Blocks are plain tuples produced by the chapter modules. The engine numbers figures, tables and listings,
resolves {fig:key}/{tab:key}/{lst:key} cross-references and <<path>> result placeholders, and writes either a
highlighted version (new material in yellow, "[NEW]" on headings) or a clean version.
"""
import os, re, copy
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_BREAK, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FONT = 'Times New Roman'; MONO = 'Courier New'

# ---------------------------------------------------------------- block constructors
def H(level, text, new=False): return ('h', level, text, new)
def P(text, new=False, style=None, align=None): return ('p', text, new, style, align)
def B(items, new=False): return ('bul', items, new)
def NL(items, new=False): return ('num', items, new)
def FIG(key, path, caption, new=False, width=6.0): return ('fig', key, path, caption, new, width)
def TAB(key, caption, header, rows, new=False, widths=None, note=None, font=10): return ('tab', key, caption, header, rows, new, widths, note, font)
def CODE(key, caption, code, new=True): return ('code', key, caption, code, new)
def PB(): return ('pb',)
def RAW(fn): return ('raw', fn)          # callable(doc, ctx) for front matter specials

# ---------------------------------------------------------------- low level helpers
def _field(run, instr, cached=''):
    r = run._r
    b = OxmlElement('w:fldChar'); b.set(qn('w:fldCharType'), 'begin'); r.append(b)
    i = OxmlElement('w:instrText'); i.set(qn('xml:space'), 'preserve'); i.text = f' {instr} '; r.append(i)
    s = OxmlElement('w:fldChar'); s.set(qn('w:fldCharType'), 'separate'); r.append(s)
    t = OxmlElement('w:t'); t.set(qn('xml:space'), 'preserve'); t.text = cached; r.append(t)
    e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), 'end'); r.append(e)

def _shade(cell_or_par, fill):
    el = cell_or_par._tc if hasattr(cell_or_par, '_tc') else cell_or_par._p
    pr = el.get_or_add_tcPr() if hasattr(cell_or_par, '_tc') else el.get_or_add_pPr()
    sh = OxmlElement('w:shd'); sh.set(qn('w:val'), 'clear'); sh.set(qn('w:color'), 'auto'); sh.set(qn('w:fill'), fill); pr.append(sh)

def _left_border(par, color='E0A800'):
    pPr = par._p.get_or_add_pPr(); bd = OxmlElement('w:pBdr'); l = OxmlElement('w:left')
    for k, v in (('w:val', 'single'), ('w:sz', '24'), ('w:space', '8'), ('w:color', color)): l.set(qn(k), v)
    bd.append(l); pPr.append(bd)

def _keep_next(par): par.paragraph_format.keep_with_next = True

INLINE = re.compile(r'(\*\*.+?\*\*|<i>.+?</i>|<sub>.+?</sub>|<sup>.+?</sup>)')
def add_runs(par, text, highlight=False, size=None, font=FONT, color=None):
    for part in INLINE.split(text):
        if not part: continue
        bold = part.startswith('**'); ital = part.startswith('<i>'); sub = part.startswith('<sub>'); sup = part.startswith('<sup>')
        txt = re.sub(r'^\*\*|\*\*$|^<i>|</i>$|^<sub>|</sub>$|^<sup>|</sup>$', '', part)
        r = par.add_run(txt); r.bold = bold or None; r.italic = ital or None
        if sub: r.font.subscript = True
        if sup: r.font.superscript = True
        r.font.name = font; r._element.rPr.rFonts.set(qn('w:eastAsia'), font)
        if size: r.font.size = Pt(size)
        if color: r.font.color.rgb = RGBColor.from_string(color)
        if highlight: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return par

# ---------------------------------------------------------------- document set-up
def new_document(footer_text):
    doc = Document()
    st = doc.styles['Normal']; st.font.name = FONT; st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
    pf = st.paragraph_format; pf.line_spacing = 1.5; pf.space_after = Pt(6); pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for lvl, size, before, after in ((1, 16, 18, 10), (2, 13, 14, 6), (3, 12, 10, 4)):
        h = doc.styles[f'Heading {lvl}']; h.font.name = FONT; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = RGBColor(0, 0, 0)
        h.element.rPr.rFonts.set(qn('w:eastAsia'), FONT); h.element.rPr.rFonts.set(qn('w:ascii'), FONT); h.element.rPr.rFonts.set(qn('w:hAnsi'), FONT)
        h.paragraph_format.space_before = Pt(before); h.paragraph_format.space_after = Pt(after); h.paragraph_format.keep_with_next = True
        h.paragraph_format.line_spacing = 1.15; h.font.italic = False
    cap = doc.styles['Caption']; cap.font.name = FONT; cap.font.size = Pt(10); cap.font.bold = False; cap.font.italic = True
    cap.font.color.rgb = RGBColor(0, 0, 0); cap.paragraph_format.line_spacing = 1.0; cap.paragraph_format.space_after = Pt(8)
    sec = doc.sections[0]
    sec.page_height = Cm(29.7); sec.page_width = Cm(21.0)
    sec.left_margin = Inches(1.25); sec.right_margin = Inches(1); sec.top_margin = Inches(1); sec.bottom_margin = Inches(1)
    # ask Word to refresh TOC / list fields when the file is opened
    settings = doc.settings.element; uf = OxmlElement('w:updateFields'); uf.set(qn('w:val'), 'true'); settings.append(uf)
    doc._footer_text = footer_text
    _footer(sec, footer_text, roman=True)
    return doc

def _footer(sec, text, roman=False, restart=False):
    sec.footer.is_linked_to_previous = False
    f = sec.footer.paragraphs[0]; f.text = ''
    f.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_runs(f, text + ' ', size=9); r = f.add_run(); r.font.size = Pt(9); r.font.name = FONT; _field(r, 'PAGE', '1')
    sectPr = sec._sectPr; pg = sectPr.find(qn('w:pgNumType'))
    if pg is None: pg = OxmlElement('w:pgNumType'); sectPr.append(pg)
    pg.set(qn('w:fmt'), 'lowerRoman' if roman else 'decimal')
    if restart or roman: pg.set(qn('w:start'), '1')

def start_main_section(doc):
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    sec.left_margin = Inches(1.25); sec.right_margin = Inches(1); sec.top_margin = Inches(1); sec.bottom_margin = Inches(1)
    _footer(sec, doc._footer_text, roman=False, restart=True)

def toc(doc, kind='toc', cached='Right-click and choose "Update Field" to refresh this list.'):
    p = doc.add_paragraph(); r = p.add_run()
    instr = {'toc': 'TOC \\o "1-3" \\h \\z \\u', 'fig': 'TOC \\h \\z \\c "Figure"', 'tab': 'TOC \\h \\z \\c "Table"', 'lst': 'TOC \\h \\z \\c "Listing"'}[kind]
    _field(r, instr, cached)

# ---------------------------------------------------------------- rendering
class Renderer:
    def __init__(self, doc, highlight, values):
        self.doc, self.hl, self.values = doc, highlight, values
        self.num = {'fig': {}, 'tab': {}, 'lst': {}}; self.index = {'fig': [], 'tab': [], 'lst': []}; self.figpaths = {}
    def number(self, blocks):
        c = {'fig': 0, 'tab': 0, 'lst': 0}
        for b in blocks:
            kind = {'fig': 'fig', 'tab': 'tab', 'code': 'lst'}.get(b[0])
            if kind:
                c[kind] += 1; self.num[kind][b[1]] = c[kind]
    def resolve(self, text):
        def ref(m):
            kind, key = m.group(1), m.group(2)
            if key not in self.num[kind]: raise KeyError(f'unknown {kind} ref {key}')
            return {'fig': 'Figure', 'tab': 'Table', 'lst': 'Listing'}[kind] + ' ' + str(self.num[kind][key])
        text = re.sub(r'\{(fig|tab|lst):([\w-]+)\}', ref, text)
        def val(m):
            k = m.group(1)
            if k not in self.values: raise KeyError('missing value ' + k)
            return str(self.values[k])
        return re.sub(r'<<([\w.\-]+)>>', val, text)
    def caption(self, kind, key, text, new, before=False):
        label = {'fig': 'Figure', 'tab': 'Table', 'lst': 'Listing'}[kind]
        p = self.doc.add_paragraph(style='Caption')
        if before: _keep_next(p)
        r = p.add_run(label + ' '); r.font.name = FONT
        r2 = p.add_run(); r2.font.name = FONT; _field(r2, f'SEQ {label} \\* ARABIC', str(self.num[kind][key]))
        add_runs(p, '. ' + self.resolve(text) + (' [NEW]' if (new and self.hl) else ''), highlight=new and self.hl)
        if new and self.hl:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW; r2.font.highlight_color = WD_COLOR_INDEX.YELLOW
        self.index[kind].append((self.num[kind][key], key, text))
        return p
    def render(self, blocks):
        self.number(blocks)
        for b in blocks:
            getattr(self, '_' + b[0])(*b[1:])
    def _raw(self, fn): fn(self.doc, self)
    def _pb(self): self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    def _h(self, level, text, new):
        t = self.resolve(text) + (' [NEW]' if (new and self.hl) else '')
        p = self.doc.add_heading(level=level); add_runs(p, t, highlight=new and self.hl)
        for r in p.runs: r.font.color.rgb = RGBColor(0, 0, 0)
        if level == 1: p.paragraph_format.page_break_before = True
    def _p(self, text, new, style, align):
        p = self.doc.add_paragraph(style=style) if style else self.doc.add_paragraph()
        if align == 'center': p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(p, self.resolve(text), highlight=new and self.hl)
    def _list(self, items, new, style):
        for it in items:
            p = self.doc.add_paragraph(style=style); p.paragraph_format.space_after = Pt(3)
            add_runs(p, self.resolve(it), highlight=new and self.hl)
    def _bul(self, items, new): self._list(items, new, 'List Bullet')
    def _num(self, items, new): self._list(items, new, 'List Number')
    def _fig(self, key, path, caption, new, width):
        self.figpaths[key] = path
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; _keep_next(p)
        p.paragraph_format.line_spacing = 1.0
        p.add_run().add_picture(path, width=Inches(width))
        if new and self.hl: _left_border(p)
        self.caption('fig', key, caption, new)
    def _tab(self, key, caption, header, rows, new, widths, note, font):
        self.caption('tab', key, caption, new, before=True)
        t = self.doc.add_table(rows=1, cols=len(header)); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        def fill(cell, txt, bold=False, shade=None):
            cell.text = ''; pp = cell.paragraphs[0]; pp.paragraph_format.line_spacing = 1.0; pp.paragraph_format.space_after = Pt(0)
            add_runs(pp, ('**' + txt + '**') if bold and txt else self.resolve(str(txt)), size=font, highlight=False)
            if shade: _shade(cell, shade)
        for i, h in enumerate(header): fill(t.rows[0].cells[i], h, bold=True, shade='FFF2B3' if (new and self.hl) else 'D9E2F3')
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row): fill(cells[i], v)
        tr = t.rows[0]._tr; trPr = tr.get_or_add_trPr(); th = OxmlElement('w:tblHeader'); th.set(qn('w:val'), 'true'); trPr.append(th)
        if widths:
            t.autofit = False
            tblPr = t._tbl.tblPr; lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
            tw = OxmlElement('w:tblW'); tw.set(qn('w:w'), str(int(sum(widths) * 567))); tw.set(qn('w:type'), 'dxa')
            old = tblPr.find(qn('w:tblW'))
            if old is not None: tblPr.remove(old)
            tblPr.append(tw)
            grid = t._tbl.tblGrid
            for i, gc in enumerate(grid.findall(qn('w:gridCol'))): gc.set(qn('w:w'), str(int(widths[i] * 567)))
            for row in t.rows:
                for i, w in enumerate(widths): row.cells[i].width = Cm(w)
        if note:
            p = self.doc.add_paragraph(); p.paragraph_format.line_spacing = 1.0
            add_runs(p, self.resolve(note), size=9, highlight=new and self.hl)
        else:
            sp = self.doc.add_paragraph(); sp.paragraph_format.space_after = Pt(2); sp.paragraph_format.line_spacing = 1.0
    def _code(self, key, caption, code, new):
        self.caption('lst', key, caption, new, before=True)
        lines = code.strip('\n').split('\n')
        for i, line in enumerate(lines):
            p = self.doc.add_paragraph(); pf = p.paragraph_format
            pf.line_spacing = 1.0; pf.space_after = Pt(0); pf.space_before = Pt(0); pf.left_indent = Cm(0.3)
            if i < len(lines) - 1: pf.keep_with_next = True
            r = p.add_run(line if line else ' '); r.font.name = MONO; r._element.rPr.rFonts.set(qn('w:eastAsia'), MONO); r.font.size = Pt(8.5)
            _shade(p, 'F2F2F2')
            if new and self.hl: _left_border(p)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
