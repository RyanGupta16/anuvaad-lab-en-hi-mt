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


if __name__ == "__main__":
    main()
