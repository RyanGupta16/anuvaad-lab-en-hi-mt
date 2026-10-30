"""Evaluate every system on the same held-out test set under identical settings.

  python -m src.evaluate                        # all systems, all stages
  python -m src.evaluate --systems mt5 gemma_lora --stages test probes

Stages
------
test         BLEU, chrF, chrF++, TER (+95 % bootstrap CIs), inference time, params,
             tokenizer fertility, cross-lingual fidelity, guardrail / hallucination flags
temperature  sampling-temperature sweep (T ∈ {0, .3, .7, 1, 1.3}) on a 150-sentence subset:
             quality vs. diversity (self-BLEU between two independent samples)
probes       responsible-AI probe suite (gender bias, fidelity, culture, sensitive, toxic)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from sacrebleu.metrics import BLEU, CHRF, TER

from . import guardrails as G
from .config import RESULTS_DIR, SEED, TEMP_SUBSET, TEMPERATURES, TRANSLATIONS_DIR, get_device
from .data import build_splits
from .probes import all_probes, score_gender
from .systems import PRIMARY, SYSTEMS, Loaded

METRICS_DIR = RESULTS_DIR / "metrics"
TEMP_DIR = RESULTS_DIR / "temperature"
PROBE_DIR = RESULTS_DIR / "probes"
for d in (METRICS_DIR, TEMP_DIR, PROBE_DIR):
    d.mkdir(parents=True, exist_ok=True)

BLEU_M, CHRF_M, CHRFPP_M, TER_M = BLEU(), CHRF(), CHRF(word_order=2), TER()
_SBLEU = BLEU(effective_order=True)


def corpus_scores(hyps, refs) -> dict:
    return {"bleu": BLEU_M.corpus_score(hyps, [refs]).score,
            "chrf": CHRF_M.corpus_score(hyps, [refs]).score,
            "chrfpp": CHRFPP_M.corpus_score(hyps, [refs]).score,
            "ter": TER_M.corpus_score(hyps, [refs]).score}


def sufficient_stats(hyps, refs):
    """Per-sentence sufficient statistics → fast bootstrap over corpus-level metrics."""
    b = np.array(BLEU_M._extract_corpus_statistics(hyps, [refs]))
    c = np.array(CHRF_M._extract_corpus_statistics(hyps, [refs]))
    return b, c


def bootstrap_ci(b_stats, c_stats, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    N = len(b_stats)
    bl, ch = [], []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        bl.append(BLEU_M._compute_score_from_stats(list(b_stats[idx].sum(0))).score)
        ch.append(CHRF_M._compute_score_from_stats(list(c_stats[idx].sum(0))).score)
    return {"bleu_ci": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
            "chrf_ci": [float(np.percentile(ch, 2.5)), float(np.percentile(ch, 97.5))]}


def fertility(sysm: Loaded, texts: list[str]) -> float:
    words = sum(len(t.split()) for t in texts)
    return sysm.n_tokens(texts) / max(words, 1)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def stage_test(sysm: Loaded, test, scorer) -> dict:
    src, refs = test.en.tolist(), test.hi.tolist()
    sysm.translate(src[:8])                                   # warm-up (kernel compilation)
    sync(sysm.device)
    t0 = time.perf_counter()
    hyps = sysm.translate(src)
    sync(sysm.device)
    elapsed = time.perf_counter() - t0

    scores = corpus_scores(hyps, refs)
    b_stats, c_stats = sufficient_stats(hyps, refs)
    scores.update(bootstrap_ci(b_stats, c_stats))
    sims = scorer(src, hyps)
    reports = [G.check(s, h, sim) for s, h, sim in zip(src, hyps, sims)]
    flags = {k: float(np.mean([r.flags[k] for r in reports]) * 100) for k in reports[0].flags}

    rows = []
    for i, (s, r, h) in enumerate(zip(src, refs, hyps)):
        rows.append({"id": i, "src": s, "ref": r, "hyp": h,
                     "chrf": CHRF_M.sentence_score(h, [r]).score,
                     "bleu": _SBLEU.sentence_score(h, [r]).score,
                     "sim": round(sims[i], 4), **reports[i].as_dict()})
    with (TRANSLATIONS_DIR / f"{sysm.key}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    info = sysm.sys.train_info()
    out_tokens = sysm.n_tokens(hyps)
    return {
        "key": sysm.key, "label": sysm.sys.label, "family": sysm.sys.family, "base": sysm.sys.base,
        "adaptation": sysm.sys.adaptation, "pretrained": sysm.sys.pretrained,
        "decoding": sysm.sys.decoding, **scores,
        "params": sysm.params, "trainable_params": sysm.trainable,
        "train_time_sec": info.get("train_time_sec"),
        "inference_sec": elapsed, "ms_per_sentence": 1000 * elapsed / len(src),
        "sentences_per_sec": len(src) / elapsed, "output_tokens_per_sec": out_tokens / elapsed,
        "fertility_hi": fertility(sysm, refs), "fertility_en": fertility(sysm, src),
        "semantic_fidelity": float(np.mean(sims)),
        "hallucination_rate": float(np.mean([r.hallucination for r in reports]) * 100),
        "flag_rates": flags,
        "len_ratio": sum(len(h.split()) for h in hyps) / sum(len(r.split()) for r in refs),
        "sentence_chrf": [r["chrf"] for r in rows],
        "bleu_stats": b_stats.tolist(), "chrf_stats": c_stats.tolist(),
        "n_test": len(src), "device": str(sysm.device),
    }


def self_bleu(a: list[str], b: list[str]) -> float:
    return BLEU_M.corpus_score(a, [b]).score


def distinct_n(hyps, n=2):
    grams, total = set(), 0
    for h in hyps:
        w = h.split()
        g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        grams.update(g)
        total += len(g)
    return 100 * len(grams) / max(total, 1)


def stage_temperature(sysm: Loaded, test, scorer) -> dict:
    sub = test.iloc[:TEMP_SUBSET]
    src, refs = sub.en.tolist(), sub.hi.tolist()
    res, samples = [], {}
    for T in TEMPERATURES:
        runs = []
        for r in range(1 if T == 0 else 2):
            torch.manual_seed(SEED + r)
            runs.append(sysm.translate(src, temperature=T))
        sims = [scorer(src, h) for h in runs]
        halluc = [np.mean([G.check(s, h, si).hallucination for s, h, si in zip(src, hs, ss)]) * 100
                  for hs, ss in zip(runs, sims)]
        sc = [corpus_scores(h, refs) for h in runs]
        res.append({
            "temperature": T,
            "bleu": float(np.mean([s["bleu"] for s in sc])),
            "chrf": float(np.mean([s["chrf"] for s in sc])),
            "semantic_fidelity": float(np.mean([np.mean(s) for s in sims])),
            "hallucination_rate": float(np.mean(halluc)),
            "self_bleu": self_bleu(runs[0], runs[1]) if len(runs) > 1 else 100.0,
            "distinct_2": float(np.mean([distinct_n(h) for h in runs])),
        })
        samples[str(T)] = [{"src": src[i], "ref": refs[i], "hyp": runs[0][i]} for i in range(6)]
        print(f"    T={T}: {res[-1]}", flush=True)
    return {"key": sysm.key, "label": sysm.sys.label, "sweep": res, "samples": samples}


def stage_probes(sysm: Loaded, scorer) -> dict:
    items = all_probes()
    src = [it["en"] for it in items]
    outs = sysm.translate(src)
    sims = scorer(src, outs)
    rows = [{**it, "hi": o, "sim": round(s, 4), **G.check(it["en"], o, s).as_dict()}
            for it, o, s in zip(items, outs, sims)]
    return {"key": sysm.key, "label": sysm.sys.label, "gender": score_gender(items, outs),
            "items": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--stages", nargs="+", default=["test", "temperature", "probes"])
    args = ap.parse_args()

    device = get_device()
    test = build_splits()["test"]
    scorer = G.SemanticScorer("cpu")
    retriever = None
    for key in args.systems:
        print(f"== {key}", flush=True)
        if key.endswith("icl") and retriever is None:
            from .decoder_llm import ShotRetriever
            retriever = ShotRetriever(build_splits()["train"], device)
        sysm = Loaded(key, device, retriever)
        if "test" in args.stages:
            m = stage_test(sysm, test, scorer)
            (METRICS_DIR / f"{key}.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
            print(f"   BLEU {m['bleu']:.2f}  chrF {m['chrf']:.2f}  {m['ms_per_sentence']:.0f} ms/sent", flush=True)
        if "temperature" in args.stages and (key in PRIMARY):
            (TEMP_DIR / f"{key}.json").write_text(json.dumps(stage_temperature(sysm, test, scorer),
                                                             indent=1, ensure_ascii=False))
        if "probes" in args.stages and not key.endswith("beam"):
            (PROBE_DIR / f"{key}.json").write_text(json.dumps(stage_probes(sysm, scorer),
                                                              indent=1, ensure_ascii=False))
        sysm.close()


if __name__ == "__main__":
    main()
