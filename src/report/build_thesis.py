"""Phase 8: build the thesis (Word + PDF) from the project chapters in ``docs/``.

Why generate instead of writing in Word by hand?  Every number in the thesis then comes from the same chapter files
that were written next to the executed notebooks, so the report cannot drift away from the results.  Re-running this
script after any change rebuilds the whole document in ~1 minute.

Input   docs/00_abstract.md, docs/01..09_*.md (chapters), docs/99_improvement_ideas.md (future work),
        docs/decision_log.md (appendix), reports/figures/*.png (figures referenced in the chapters)
Output  docs/report/TransitPulse_Thesis.docx and .pdf

Markdown support (only what the chapters use): headings, paragraphs, **bold**, *italic*, `code`, [links](url),
bullet / numbered lists, pipe tables, fenced code, GitHub alerts (> [!NOTE]) and mermaid diagrams (rendered to PNG via
the public mermaid.ink service; if offline the diagram source is shown instead).  Figure files mentioned in the text
(e.g. ``p3_roc_pr.png``) are inserted right after the paragraph or table that first mentions them, with a numbered
caption.  The PDF is produced with Microsoft Word (COM) so the table of contents gets real page numbers; fallbacks are
docx2pdf and LibreOffice.

Run: ``python -m src.report.build_thesis``
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401  (kept for landscape tables if needed)
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FIG = ROOT / "reports" / "figures"
OUT = DOCS / "report"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "_diagrams"; CACHE.mkdir(exist_ok=True)

TITLE = "TransitPulse: Predicting and Explaining Bus Delays for Transit Agencies"
SUBTITLE = "A machine-learning and explainable-AI system for MBTA bus routes in Boston"
META = ["Graduate Final Project Report", "[Student name]", "[Degree programme] · [University]",
        "Supervisor: [name]", "November 2026"]

# chapter order: (file, chapter title shown in the thesis); files that do not exist are skipped with a warning
CHAPTERS = [
    ("01_introduction.md", "Introduction"),
    ("01_data_source_selection.md", "Data Source Selection"),
    ("02_data_preparation_and_eda.md", "Data Preparation and Exploratory Analysis"),
    ("03_external_data_integration.md", "External Data Integration"),
    ("04_feature_engineering.md", "Feature Engineering"),
    ("05_models_and_evaluation.md", "Models and Evaluation"),
    ("06_explainability.md", "Explainability"),
    ("07_extensions.md", "Extensions: Severity, Delay Minutes, Propagation, Bunching, Chronic Segments"),
    ("08_dashboard.md", "The TransitPulse Website"),
    ("08_live_mode.md", "Live Mode"),
    ("09_conclusion.md", "Conclusions"),
    ("99_improvement_ideas.md", "Future Work"),
]
APPENDICES = [("decision_log.md", "Decision Log")]
# figures of a phase that the chapter text does not name explicitly are added at the end of the chapter
CHAPTER_FIGS = {"02_data_preparation_and_eda.md": "eda_", "03_external_data_integration.md": "p1_",
                "04_feature_engineering.md": "p2_", "05_models_and_evaluation.md": "p3_", "06_explainability.md": "p4_",
                "07_extensions.md": "p5_"}

CAPTIONS = {  # nicer captions for the most important figures (fallback: derived from the file name)
    "eda_target.png": "Distribution of delay and share of events more than 5 minutes late",
    "eda_by_hour.png": "Late rate by hour of day, weekday vs weekend vs holiday",
    "eda_by_route.png": "Late rate by route",
    "eda_stop_position.png": "Delay along the trip (position of the timepoint on the route)",
    "eda_propagation.png": "Delay propagation: start delay vs end delay",
    "eda_bunching.png": "Headway ratio distribution (bunching and big gaps)",
    "p2_splits.png": "Out-of-time split: train Jan-Aug, validation Sep-Oct, test Nov-Dec 2025",
    "p3_roc_pr.png": "ROC and precision-recall curves on the test period",
    "p3_model_comparison.png": "Model comparison (test AUC / PR-AUC), planning and operational modes",
    "p3_calibration.png": "Calibration of the predicted probabilities",
    "p3_ablation.png": "Feature-group ablation",
    "p3_walk_forward.png": "Walk-forward evaluation by month",
    "p3_learning_curve.png": "Learning curve (training-set size vs test AUC)",
    "p3_threshold.png": "Precision, recall and F1 as a function of the decision threshold",
    "p4_global_importance.png": "Global SHAP importance (top features)",
    "p4_group_importance.png": "SHAP importance by feature group",
    "p4_shap_by_period.png": "Mean SHAP contribution of each group by time of day",
    "p4_what_if_planning.png": "What-if analysis: adding schedule time (planning mode)",
    "p4_what_if_operational.png": "What-if analysis: delay tipping point (operational mode)",
}

_fig_no = 0
_tab_no = 0
_placed: set[str] = set()


# ------------------------------------------------------------------ helpers
def _font(run, name="Calibri", size=None, bold=None, italic=None, color=None, mono=False):
    run.font.name = "Consolas" if mono else name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), run.font.name)
    if size: run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    if italic is not None: run.italic = italic
    if color: run.font.color.rgb = RGBColor.from_string(color)


INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|<br>)")


def add_inline(par, text: str, size=None, bold=False):
    """Write markdown inline formatting into a paragraph."""
    text = text.replace("\\|", "|").replace("&nbsp;", " ")
    for part in INLINE.split(text):
        if not part:
            continue
        if part == "<br>":
            par.add_run().add_break(); continue
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2]); _font(r, size=size, bold=True)
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1]); _font(r, size=(size or 11) - 1, mono=True, color="1F3864", bold=bold)
        elif part.startswith("[") and "](" in part:
            label = part[1:part.index("](")]
            r = par.add_run(label.strip("`")); _font(r, size=size, color="1F4E79", bold=bold)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = par.add_run(part[1:-1]); _font(r, size=size, italic=True, bold=bold)
        else:
            r = par.add_run(part); _font(r, size=size, bold=bold)


def shade(cell, hex_color: str):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def caption(doc, kind: str, text: str):
    global _fig_no, _tab_no
    if kind == "Figure":
        _fig_no += 1; n = _fig_no
    else:
        _tab_no += 1; n = _tab_no
    p = doc.add_paragraph(style="Caption")
    r = p.add_run(f"{kind} {n}. "); r.bold = True
    p.add_run(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_figure(doc, path: Path, text: str, width_cm=15.5):
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, "Figure", text)


def place_figures(doc, text: str):
    """Insert every existing figure whose file name appears in ``text`` and has not been placed yet."""
    for name in re.findall(r"([A-Za-z0-9_]+\.png)", text):
        if name in _placed or not (FIG / name).exists():
            continue
        _placed.add(name)
        add_figure(doc, FIG / name, CAPTIONS.get(name, name[:-4].split("_", 1)[-1].replace("_", " ").capitalize()))


def add_table(doc, rows: list[list[str]], title: str | None = None):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    t = doc.add_table(rows=1, cols=ncol); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    size = 9 if ncol <= 4 else 8 if ncol <= 7 else 7
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; add_inline(c.paragraphs[0], h, size=size, bold=True); shade(c, "D9E2F3")
    for r in body:
        cells = t.add_row().cells
        for i in range(ncol):
            cells[i].text = ""
            add_inline(cells[i].paragraphs[0], r[i] if i < len(r) else "", size=size)
    # column widths proportional to the text they hold (equal widths squeeze long "reason" columns)
    avg = [sum(len(rw[i]) if i < len(rw) else 0 for rw in rows) / len(rows) for i in range(ncol)]
    mx = [max(len(rw[i]) if i < len(rw) else 0 for rw in rows) for i in range(ncol)]
    # short columns (ids, phases, numbers) get room for their longest entry so they never wrap letter by letter
    w = [min(max(a, 8.0, len(header[i]) + 2, mx[i] + 3 if mx[i] <= 12 else 0), 70.0) for i, a in enumerate(avg)]
    total = 16.4                                     # usable A4 width in cm (21 - 2 x 2.3 margins)
    t.autofit = False
    for row in t.rows:
        for i, c in enumerate(row.cells):
            c.width = Cm(total * w[i] / sum(w))
    doc.add_paragraph()


def mermaid_png(src: str) -> Path | None:
    key = base64.urlsafe_b64encode(src.encode()).decode()
    import hashlib
    out = CACHE / f"m_{hashlib.md5(src.encode()).hexdigest()[:12]}.png"
    if out.exists():
        return out
    try:
        import requests
        r = requests.get(f"https://mermaid.ink/img/{key}?type=png&bgColor=FFFFFF", timeout=30)
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            out.write_bytes(r.content); return out
    except Exception as e:  # noqa: BLE001
        print("mermaid render failed:", e)
    return None


def split_row(line: str) -> list[str]:
    s = line.strip().strip("|")
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]


# ------------------------------------------------------------------ markdown -> docx
def render_markdown(doc, md: str, top_level: int):
    """Render one chapter. ``top_level``: the docx heading level for the chapter's own '#' heading (it is skipped)."""
    lines = md.splitlines()
    i, para = 0, []

    def flush():
        nonlocal para
        if para:
            text = " ".join(x.strip() for x in para)
            p = doc.add_paragraph(); add_inline(p, text); p.paragraph_format.space_after = Pt(6)
            place_figures(doc, text)
            para = []

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            lang = s[3:].strip(); block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i]); i += 1
            i += 1
            src = "\n".join(block)
            if lang == "mermaid":
                png = mermaid_png(src)
                if png:
                    add_figure(doc, png, "Architecture / flow diagram", width_cm=14); continue
            p = doc.add_paragraph()
            r = p.add_run(src); _font(r, size=8, mono=True)
            p.paragraph_format.left_indent = Cm(0.5)
            continue
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            title = s.lstrip("#").strip()
            if level == 1:   # chapter title comes from CHAPTERS
                i += 1; continue
            title = re.sub(r"^(\d+(\.\d+)*\.?\s+)", "", title)          # drop manual numbering ("8.1 What")
            title = re.sub(r"^Step \d+\s*[-–]\s*", "", title)
            doc.add_heading(title.replace("**", "").replace("`", ""), level=min(top_level + level - 1, 4))
            i += 1; continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            flush()
            rows = [split_row(s)]; i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i])); i += 1
            add_table(doc, rows)
            place_figures(doc, "\n".join("|".join(r) for r in rows))
            continue
        if s.startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip()); i += 1
            kind = None
            if quote and re.match(r"\[!(\w+)\]", quote[0]):
                kind = re.match(r"\[!(\w+)\]", quote[0]).group(1).capitalize(); quote[0] = re.sub(r"\[!\w+\]\s*", "", quote[0])
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.8)
            if kind:
                r = p.add_run(f"{kind}: "); _font(r, bold=True, color="7F6000")
            add_inline(p, " ".join(q for q in quote if q))
            continue
        m_b = re.match(r"^(\s*)[*\-+]\s+(.*)", ln)
        m_n = re.match(r"^(\s*)(\d+[a-z]?)[.)]\s+(.*)", ln)
        if m_b or m_n:
            flush()
            indent = len((m_b or m_n).group(1)) // 2
            text = m_b.group(2) if m_b else m_n.group(3)
            i += 1
            # continuation lines: indented by >= 2 spaces, not a nested list item and not an indented table
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() and \
                    not lines[i].strip().startswith("|") and \
                    not re.match(r"^\s*([*\-+]|\d+[a-z]?[.)])\s+", lines[i]):
                text += " " + lines[i].strip(); i += 1
            style = "List Bullet" if m_b else "List Number"
            if indent and m_b:
                style = "List Bullet 2"
            p = doc.add_paragraph(style=style)
            if m_n and style == "List Number":
                # keep the author's numbering visible (Word restarts are unreliable across tables)
                p.style = doc.styles["List Paragraph"] if "List Paragraph" in doc.styles else p.style
                r = p.add_run(f"{m_n.group(2)}. "); _font(r, bold=True)
                p.paragraph_format.left_indent = Cm(0.6)
            add_inline(p, text)
            place_figures(doc, text)
            continue
        if not s:
            flush(); i += 1; continue
        if re.match(r"^-{3,}$", s):
            flush(); i += 1; continue
        para.append(ln); i += 1
    flush()


def toc(doc):
    p = doc.add_paragraph()
    r = p.add_run()
    for tag, text in (("begin", None), (None, 'TOC \\o "1-2" \\h \\z \\u'), ("separate", None), (None, None), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag); r._r.append(el)
        elif text:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = text; r._r.append(it)
        else:
            t = OxmlElement("w:t"); t.text = "Table of contents - updated automatically when the PDF is built."; r._r.append(t)


def page_numbers(section):
    p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag); r._r.append(el)
        else:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = text; r._r.append(it)


def build() -> Path:
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
    for lvl, size in ((1, 18), (2, 14), (3, 12), (4, 11)):
        h = doc.styles[f"Heading {lvl}"]; h.font.size = Pt(size); h.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.3); sec.top_margin = sec.bottom_margin = Cm(2.2)
    page_numbers(sec)

    # title page
    for _ in range(5):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITLE); _font(r, size=24, bold=True, color="1F3864")
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(SUBTITLE); _font(r, size=14, italic=True)
    for _ in range(4):
        doc.add_paragraph()
    for m in META:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(m); _font(r, size=12)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    abstract = DOCS / "00_abstract.md"
    if abstract.exists():
        doc.add_heading("Abstract", level=1)
        render_markdown(doc, abstract.read_text(encoding="utf-8"), 1)
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.add_heading("Contents", level=1)
    toc(doc)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    n = 0
    for fname, title in CHAPTERS:
        f = DOCS / fname
        if not f.exists():
            print("skip (missing):", fname); continue
        n += 1
        doc.add_heading(f"{n}. {title}", level=1)
        render_markdown(doc, f.read_text(encoding="utf-8"), 2)
        rest = [g.name for g in sorted(FIG.glob(f"{CHAPTER_FIGS[fname]}*.png"))] if fname in CHAPTER_FIGS else []
        rest = [g for g in rest if g not in _placed]
        if rest:
            doc.add_heading("Additional figures", level=2)
            place_figures(doc, " ".join(rest))
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    for k, (fname, title) in enumerate(APPENDICES):
        f = DOCS / fname
        if f.exists():
            doc.add_heading(f"Appendix {chr(65 + k)}. {title}", level=1)
            render_markdown(doc, f.read_text(encoding="utf-8"), 2)
    path = OUT / "TransitPulse_Thesis.docx"
    doc.save(path)
    print(f"docx: {path}  ({n} chapters, {_fig_no} figures)")
    return path


def to_pdf(docx_path: Path) -> Path | None:
    pdf = docx_path.with_suffix(".pdf")
    try:   # Word: update TOC/page fields, save both files
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        word = win32com.client.DispatchEx("Word.Application"); word.Visible = False
        try:
            d = word.Documents.Open(str(docx_path))
            for t in d.TablesOfContents:
                t.Update()
            d.Fields.Update()
            d.Save()
            d.SaveAs2(str(pdf), FileFormat=17)
            d.Close(False)
        finally:
            word.Quit()
        print("pdf (Word):", pdf); return pdf
    except Exception as e:  # noqa: BLE001
        print("Word COM failed:", e)
    try:
        from docx2pdf import convert
        convert(str(docx_path), str(pdf)); print("pdf (docx2pdf):", pdf); return pdf
    except Exception as e:  # noqa: BLE001
        print("docx2pdf failed:", e)
    import shutil
    import subprocess
    soffice = shutil.which("soffice")
    if soffice:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(docx_path)], check=False)
        if pdf.exists():
            print("pdf (LibreOffice):", pdf); return pdf
    print("PDF not produced: open the .docx in Word and use File > Save As > PDF.")
    return None


if __name__ == "__main__":
    p = build()
    if "--no-pdf" not in sys.argv:
        to_pdf(p)
