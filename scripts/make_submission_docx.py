"""Create the LMS submission document: <Name>_<RegNum>_FGAI_Assignment3.docx

  .venv/bin/python scripts/make_submission_docx.py --name "Ryan Gupta" --reg 2XXXXXXXX \
      --repo https://github.com/<user>/<repo>
"""
import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
INK = RGBColor(0x16, 0x21, 0x3E)


def add_link(par, url, text=None):
    part = par.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                          is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color"); color.set(qn("w:val"), "0F7C8C"); rpr.append(color)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rpr.append(u)
    run.append(rpr)
    t = OxmlElement("w:t"); t.text = text or url; run.append(t)
    link.append(run)
    par._p.append(link)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Ryan Gupta")
    ap.add_argument("--reg", default="RegNum")
    ap.add_argument("--repo", default="https://github.com/RyanGupta16/anuvaad-lab-en-hi-mt")
    args = ap.parse_args()

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11.5)

    h = doc.add_paragraph(); h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = h.add_run("Foundations of Generative AI (ICT4403)"); r.bold = True; r.font.size = Pt(17); r.font.color.rgb = INK
    h2 = doc.add_paragraph(); h2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = h2.add_run("Assignment 3 — Submission Details"); r.font.size = Pt(13); r.font.color.rgb = INK

    rows = [
        ("Student name", args.name, None),
        ("Registration number", args.reg, None),
        ("Assignment title", "Assignment 3: Machine Translation with Encoder–Decoder and Decoder-only "
                             "Models (Transformer, mT5, LLaMA, Gemma)", None),
        ("Title of the developed autoregressive text-generation system",
         "Anuvaad Lab — an English→Hindi neural machine-translation benchmark comparing a from-scratch "
         "Transformer, fine-tuned mT5-small, and prompted / LoRA-fine-tuned LLaMA-3.2-1B and Gemma-3-1B, "
         "with responsible-AI guardrails and an interactive dashboard", None),
        ("Dataset name", "IIT Bombay English–Hindi Parallel Corpus (cfilt/iitb-english-hindi)", None),
        ("Dataset link", None, "https://huggingface.co/datasets/cfilt/iitb-english-hindi"),
        ("Dataset homepage", None, "https://www.cfilt.iitb.ac.in/iitb_parallel/"),
        ("GitHub repository link", None, args.repo),
    ]
    table = doc.add_table(rows=0, cols=2)
    table.style = "Light Grid Accent 1"
    for k, v, link in rows:
        cells = table.add_row().cells
        cells[0].text = ""
        kr = cells[0].paragraphs[0].add_run(k); kr.bold = True
        if link:
            add_link(cells[1].paragraphs[0], link)
        else:
            cells[1].text = v
    for row in table.rows:
        row.cells[0].width = Pt(150); row.cells[1].width = Pt(330)

    summ = ROOT / "results" / "summary.json"
    if summ.exists():
        S = json.loads(summ.read_text())
        doc.add_paragraph()
        p = doc.add_paragraph(); p.add_run("Headline results (500-sentence IITB test subset)").bold = True
        t = doc.add_table(rows=1, cols=5); t.style = "Light List Accent 1"
        for c, txt in zip(t.rows[0].cells, ["System", "BLEU", "chrF", "Params", "ms / sentence"]):
            c.text = txt
        for k in sorted(S["systems"], key=lambda k: -S["systems"][k]["chrf"]):
            s = S["systems"][k]
            cells = t.add_row().cells
            cells[0].text = s["label"]; cells[1].text = f"{s['bleu']:.2f}"; cells[2].text = f"{s['chrf']:.2f}"
            cells[3].text = f"{s['params']/1e6:.0f} M"; cells[4].text = f"{s['ms_per_sentence']:.0f}"

    out = ROOT / "submission" / f"{args.name.replace(' ', '')}_{args.reg}_FGAI_Assignment3.docx"
    out.parent.mkdir(exist_ok=True)
    doc.save(out)
    print("wrote", out)
    write_pdf(args.name, args.reg, args.repo)




def write_pdf(name: str, reg: str, repo: str) -> Path:
    """Same content as the .docx, rendered directly to PDF (for the LMS upload)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    ink = colors.HexColor("#16213E")
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=ink, fontSize=17, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Normal"], textColor=ink, fontSize=13, alignment=1, spaceAfter=14)
    cell = ParagraphStyle("cell", parent=ss["Normal"], fontSize=10.5, leading=14)
    bold = ParagraphStyle("bold", parent=cell, fontName="Helvetica-Bold")
    link = lambda u: Paragraph(f'<link href="{u}" color="#0F7C8C"><u>{u}</u></link>', cell)

    rows = [
        ("Student name", Paragraph(name, cell)),
        ("Registration number", Paragraph(reg, cell)),
        ("Assignment title", Paragraph("Assignment 3: Machine Translation with Encoder–Decoder and Decoder-only "
                                       "Models (Transformer, mT5, LLaMA, Gemma)", cell)),
        ("Title of the developed autoregressive text-generation system", Paragraph(
            "Anuvaad Lab — an English→Hindi neural machine-translation benchmark comparing a from-scratch "
            "Transformer, fine-tuned mT5-small, and prompted / LoRA-fine-tuned LLaMA-3.2-1B and Gemma-3-1B, "
            "with responsible-AI guardrails and an interactive dashboard", cell)),
        ("Dataset name", Paragraph("IIT Bombay English–Hindi Parallel Corpus (cfilt/iitb-english-hindi)", cell)),
        ("Dataset link", link("https://huggingface.co/datasets/cfilt/iitb-english-hindi")),
        ("Dataset homepage", link("https://www.cfilt.iitb.ac.in/iitb_parallel/")),
        ("GitHub repository link", link(repo)),
    ]
    data = [[Paragraph(k, bold), v] for k, v in rows]
    t = Table(data, colWidths=[55 * mm, 115 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9DB0D0")),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF1F6")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story = [Paragraph("Foundations of Generative AI (ICT4403)", h1),
             Paragraph("Assignment 3 — Submission Details", h2), t]

    summ = ROOT / "results" / "summary.json"
    if summ.exists():
        S = json.loads(summ.read_text())
        story += [Spacer(1, 14), Paragraph("<b>Headline results (500-sentence IITB test subset)</b>", cell),
                  Spacer(1, 6)]
        res = [["System", "BLEU", "chrF", "Params", "ms / sentence"]]
        for k in sorted(S["systems"], key=lambda k: -S["systems"][k]["chrf"]):
            s = S["systems"][k]
            res.append([s["label"], f"{s['bleu']:.2f}", f"{s['chrf']:.2f}", f"{s['params']/1e6:.0f} M",
                        f"{s['ms_per_sentence']:.0f}"])
        rt = Table(res, colWidths=[70 * mm, 22 * mm, 22 * mm, 25 * mm, 31 * mm])
        rt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9.5),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("LINEBELOW", (0, 0), (-1, 0), 0.8, ink),
                                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#CDD4E1")),
                                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#E3F2EF")),
                                ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
        story.append(rt)

    out = ROOT / "submission" / f"{name.replace(' ', '')}_{reg}_FGAI_Assignment3.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=22 * mm, bottomMargin=20 * mm,
                      title="FGAI Assignment 3 – Submission", author=name).build(story)
    print("wrote", out)
    return out


if __name__ == "__main__":
    main()
