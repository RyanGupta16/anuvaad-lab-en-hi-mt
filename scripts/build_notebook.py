"""Builds notebooks/EN_HI_Machine_Translation.ipynb (then executed with nbclient).

The notebook is the narrative entry point for evaluators: it walks through the data,
the four model families, the evaluation and the responsible-AI analysis, reading the
artefacts produced by the src/ pipeline (set RUN_TRAINING=True to retrain end to end).
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md("""# English → Hindi Machine Translation: Transformer vs mT5 vs LLaMA vs Gemma
**ICT4403 Foundations of Generative AI — Assignment 3**

This notebook is the guided tour of the project. All heavy lifting lives in the `src/` package
(so it can run headless with `./run_all.sh`); here we load the artefacts it produced, show the
key code paths and discuss the results.

| Part | What it covers |
|---|---|
| i | dataset description, preprocessing, splits; training the scratch Transformer, fine-tuning mT5, prompting + LoRA for LLaMA and Gemma |
| ii | BLEU, chrF, chrF++, TER with bootstrap CIs, training time, inference time, parameter counts, tokenizer fertility; architecture comparison |
| iii | responsible-AI analysis: gender bias (WinoMT-style), hallucination, cultural misrepresentation, sensitive content; mitigations |
"""),
code("""import sys, json, os
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option("display.max_colwidth", 120)
RUN_TRAINING = False   # True → retrain every model (≈3–4 h on Apple M4)"""),
md("## Part i — Dataset\n**IIT Bombay English–Hindi Parallel Corpus** (Kunchukuttan, Mehta & Bhattacharyya, LREC 2018) — "
   "[Hugging Face: cfilt/iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi), "
   "[official page](https://www.cfilt.iitb.ac.in/iitb_parallel/). 1.66 M training pairs drawn from news, "
   "Wikipedia, government sites, TED talks, religious texts and software localisation files; "
   "520-sentence dev set and 2,507-sentence news test set (WMT14 domain)."),
code("""from src.data import build_splits, keep_pair, normalise_hi
from src.config import DataConfig
splits = build_splits()            # cached after the first run
stats = json.loads(Path("results/data_stats.json").read_text())
pd.DataFrame({k: v for k, v in stats.items() if k in ("train", "val", "test")}).T"""),
code("""print("raw corpus sizes:", stats["raw_counts"])
print(f"train pool after filtering: {stats['after_filtering_train_pool']:,} pairs ({stats['filter_retention_pct']}% kept)")
splits["train"].sample(6, random_state=7)"""),
md("""### Preprocessing
1. Unicode **NFC** normalisation and whitespace collapsing; Hindi sentence-final `.`/`|` mapped to the danda `।` (IITB is inconsistent).
2. Noise filters: 3–40 words per side; Hindi/English word-length ratio ≤ 2.2; ≥ 60 % Devanagari letters on the Hindi side and no Devanagari on the English side;
   removal of software-localisation artefacts (`%s`, `_File`, `<tags>`, URLs, `{0}`); untranslated copies and duplicate sources dropped.
3. **Splits**: a seeded 120 k-pair training subset; official dev set (520) for validation/early stopping; a seeded 500-sentence sample of the official news test set for the final evaluation.
   A leakage guard removes any training pair whose English side appears in validation or test."""),
code("""demo = [("Open _File", "_फ़ाइल खोलें"), ("Accerciser", "एक्सेर्सिसर"),
        ("The meeting was postponed due to heavy rain.", "भारी बारिश के कारण बैठक स्थगित कर दी गई.")]
for en, hi in demo:
    print(f"keep={keep_pair(en, normalise_hi(hi), DataConfig())!s:5}  {en!r}")
display(Image("results/plots/data_lengths.png"))"""),
md("""## Part i — Models

| | Architecture | Size used | Pre-training | Adaptation |
|---|---|---|---|---|
| **Transformer** | encoder–decoder (4+4 layers, d=384, 6 heads) | 22.7 M | none | trained from scratch, joint 16 k SentencePiece unigram vocab |
| **mT5-small** | encoder–decoder (8+8 layers) | 300 M | mC4, 101 languages, span corruption | full fine-tuning (AdamW, linear decay) |
| **LLaMA-3.2-1B-Instruct** | decoder-only | 1.24 B | ~9 T tokens, mostly English | zero-shot, kNN 4-shot prompting, LoRA r=16 |
| **Gemma-3-1B-IT** | decoder-only | 1.0 B | 2 T tokens, 140 languages, 262 k vocab | zero-shot, kNN 4-shot prompting, LoRA r=16 |

The scratch Transformer uses pre-LayerNorm blocks, fused scaled-dot-product attention, three-way tied embeddings,
label smoothing 0.1, the inverse-square-root (Noam) schedule, token-bucketed batches, an incremental KV-cache and
batched beam search with GNMT length penalty."""),
code("""from src.config import TransformerConfig
from src.transformer.model import Seq2SeqTransformer, count_parameters
m = Seq2SeqTransformer(TransformerConfig(), 0, 2, 3)
print(f"{count_parameters(m)/1e6:.2f} M parameters"); print(m.decoder[0])"""),
code("""if RUN_TRAINING:
    !{sys.executable} -m src.transformer.train
    !{sys.executable} -m src.mt5_finetune
    !{sys.executable} -m src.decoder_llm --model llama
    !{sys.executable} -m src.decoder_llm --model gemma
    !{sys.executable} -m src.evaluate
    !{sys.executable} -m src.report
display(Image("results/plots/training_curves.png"))"""),
md("""### Prompting and LoRA for the decoder-only models
Both LLMs receive the same system instruction through their own chat template. **kNN few-shot** retrieves, for each
test sentence, the 4 most similar training pairs by cosine similarity of multilingual MiniLM embeddings and inserts
them as prior chat turns. **LoRA** adds rank-16 adapters to all attention and MLP projections (≈ 0.9 % of the weights)
and trains on completion tokens only (the prompt is masked from the loss)."""),
code("""from transformers import AutoTokenizer
from src.decoder_llm import messages_for, render
tok = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it")
print(render(tok, messages_for("The train was late.", shots=[("It is raining.", "बारिश हो रही है।")]), True))"""),
md("## Part ii — Evaluation\nEvery system translates the **same 500 test sentences** with batch size 16 on the same device. "
   "BLEU/chrF/chrF++/TER come from sacreBLEU; 95 % confidence intervals and significance from 1,000 bootstrap resamples."),
code("""summary = json.loads(Path("results/summary.json").read_text())
tbl = pd.read_csv("results/results_table.csv")
tbl.sort_values("chrF", ascending=False).round(2)"""),
code("""for p in ["quality", "compute", "pareto", "fertility", "sentence_chrf", "guardrail_flags"]:
    display(Image(f"results/plots/{p}.png"))"""),
code("""sig = summary["significance"]
print("best system by chrF:", summary["systems"][sig["best"]]["label"])
pd.DataFrame(sig["vs"]).T.rename(index=lambda k: summary["systems"][k]["label"]).round(3)"""),
md("### Qualitative examples (same sentences, every system)"),
code("""rows = []
for ex in summary["examples"][:5]:
    rows.append({"system": "SOURCE", "text": ex["src"]}); rows.append({"system": "REFERENCE", "text": ex["ref"]})
    for k in summary["primary"]:
        rows.append({"system": summary["systems"][k]["label"], "text": ex["hyps"][k]["hyp"], "chrF": round(ex["hyps"][k]["chrf"], 1)})
pd.DataFrame(rows)"""),
md("## Sampling-temperature analysis"),
code("""display(Image("results/plots/temperature.png"))
pd.concat({summary["temperature"][k]["label"]: pd.DataFrame(v["sweep"]).set_index("temperature")
           for k, v in summary["temperature"].items()}).round(2)"""),
md("## Part iii — Responsible AI"),
code("""display(Image("results/plots/gender_bias.png"))
pd.DataFrame({p["label"]: p["gender"] for p in summary["probes"].values()}).T.round(1)"""),
code("""cat_rows = []
for k, p in summary["probes"].items():
    if k not in summary["primary"]: continue
    for it in p["items"]:
        if it["category"] != "gender":
            cat_rows.append({"category": it["category"], "source": it["en"], "system": p["label"], "hindi": it["hi"],
                             "flags": ", ".join(f for f, v in it["flags"].items() if v), "risk": it["risk"]})
pd.DataFrame(cat_rows).sort_values(["category", "source"])"""),
md("The discussion of these outputs, the mitigation measures and the conclusion are in **README.md** (sections 8–10)."),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "EN_HI_Machine_Translation.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
