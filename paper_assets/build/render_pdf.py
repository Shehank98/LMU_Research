"""Open a .docx in LibreOffice, refresh the table of contents / figure lists / fields, export PDF and (optionally) save the refreshed .docx.
Usage: /usr/bin/python3 render_pdf.py in.docx out.pdf [refreshed.docx]"""
import sys, os, time, subprocess, uno
from com.sun.star.beans import PropertyValue
def prop(n, v): p = PropertyValue(); p.Name = n; p.Value = v; return p
src, pdf = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); out_docx = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None
port = 2002 + os.getpid() % 500
proc = subprocess.Popen(['soffice', '--headless', '--invisible', '--nologo', '--norestore', f'-env:UserInstallation=file:///tmp/lo_{port}',
                         f'--accept=socket,host=localhost,port={port};urp;'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ctx = None
for _ in range(60):
    try:
        local = uno.getComponentContext(); res = local.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', local)
        ctx = res.resolve(f'uno:socket,host=localhost,port={port};urp;StarOffice.ComponentContext'); break
    except Exception: time.sleep(1)
desktop = ctx.ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop', ctx)
doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(src), '_blank', 0, (prop('Hidden', True),))
for _ in range(2):
    idx = doc.getDocumentIndexes()
    for i in range(idx.getCount()): idx.getByIndex(i).update()
    doc.getTextFields().refresh(); doc.refresh()
doc.storeToURL(uno.systemPathToFileUrl(pdf), (prop('FilterName', 'writer_pdf_Export'),))
if out_docx: doc.storeToURL(uno.systemPathToFileUrl(out_docx), (prop('FilterName', 'MS Word 2007 XML'),))
doc.close(True)
try: desktop.terminate()
except Exception: pass
proc.wait(timeout=60); print('ok', pdf)
