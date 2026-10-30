"""Aggregate all metrics into tables, plots and a dashboard-ready summary.json.

  python -m src.report
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sacrebleu.metrics import BLEU, CHRF

from .config import DATASET_HOMEPAGE, DATASET_URL, PLOTS_DIR, RESULTS_DIR, SEED
from .data import load_splits
from .systems import PRIMARY, SYSTEMS

METRICS_DIR, TEMP_DIR, PROBE_DIR = (RESULTS_DIR / d for d in ("metrics", "temperature", "probes"))
BLEU_M, CHRF_M = BLEU(), CHRF()

PALETTE = {"transformer": "#7C8BA1", "transformer_beam": "#4B5A70", "mt5": "#F59E0B",
           "mt5_beam": "#B45309", "llama_zs": "#C7D2FE", "gemma_zs": "#BBF7D0", "llama_icl": "#93C5FD", "llama_lora": "#2563EB",
           "gemma_icl": "#86EFAC", "gemma_lora": "#059669"}

plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})


def load_metrics() -> dict:
    out = {}
    for key in SYSTEMS:
        p = METRICS_DIR / f"{key}.json"
        if p.exists():
            out[key] = json.loads(p.read_text())
    return out


def paired_bootstrap(a_stats, b_stats, metric, n=1000, seed=SEED):
    """p-value that system A is NOT better than B (Koehn, 2004)."""
    a, b = np.array(a_stats), np.array(b_stats)
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        sa = metric._compute_score_from_stats(list(a[idx].sum(0))).score
        sb = metric._compute_score_from_stats(list(b[idx].sum(0))).score
        wins += sa > sb
    return 1 - wins / n


def main_table(m: dict) -> pd.DataFrame:
    rows = []
    for k, r in m.items():
        rows.append({
            "key": k, "System": r["label"], "Architecture": r["family"], "Adaptation": r["adaptation"],
            "Decoding": r["decoding"], "BLEU": r["bleu"], "BLEU 95% CI": f"{r['bleu_ci'][0]:.1f}–{r['bleu_ci'][1]:.1f}",
            "chrF": r["chrf"], "chrF++": r["chrfpp"], "TER↓": r["ter"],
            "Params (M)": r["params"] / 1e6, "Trainable (M)": r["trainable_params"] / 1e6,
            "Train time (min)": (r["train_time_sec"] or 0) / 60,
            "Inference (s, 500 sent)": r["inference_sec"], "ms/sentence": r["ms_per_sentence"],
            "Fertility hi": r["fertility_hi"], "Fertility en": r["fertility_en"],
            "Fidelity (xling cos)": r["semantic_fidelity"], "Hallucination %": r["hallucination_rate"],
        })
    return pd.DataFrame(rows)


def significance(m: dict) -> dict:
    keys = [k for k in m]
    best = max(keys, key=lambda k: m[k]["chrf"])
    out = {"best": best, "vs": {}}
    for k in keys:
        if k == best:
            continue
        out["vs"][k] = {
            "p_chrf": paired_bootstrap(m[best]["chrf_stats"], m[k]["chrf_stats"], CHRF_M),
            "p_bleu": paired_bootstrap(m[best]["bleu_stats"], m[k]["bleu_stats"], BLEU_M),
        }
    return out


# ------------------------------------------------------------------ plots
def _bar(ax, df, col, title, fmt="{:.1f}"):
    colors = [PALETTE[k] for k in df.key]
    bars = ax.barh(df.System, df[col], color=colors)
    ax.set_title(title)
    ax.invert_yaxis()
    for b, v in zip(bars, df[col]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", fontsize=8)


def plot_quality(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    _bar(axes[0], df, "BLEU", "BLEU ↑")
    _bar(axes[1], df, "chrF", "chrF ↑")
    _bar(axes[2], df, "chrF++", "chrF++ ↑")
    fig.suptitle("Translation quality on the IITB test subset (500 sentences)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "quality.png")
    plt.close(fig)


def plot_compute(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    _bar(axes[0], df, "Params (M)", "Total parameters (M)", "{:.0f}")
    _bar(axes[1], df, "Train time (min)", "Training / fine-tuning time (min)", "{:.1f}")
    _bar(axes[2], df, "ms/sentence", "Inference latency (ms / sentence)", "{:.0f}")
    fig.suptitle("Computational requirements (Apple M4, MPS backend)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "compute.png")
    plt.close(fig)


def plot_pareto(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, r in df.iterrows():
        ax.scatter(r["ms/sentence"], r["chrF"], s=40 + 2.2 * np.sqrt(r["Params (M)"] * 1000) / 3,
                   color=PALETTE[r.key], alpha=.85, edgecolor="white", linewidth=1.5)
        ax.annotate(r.System, (r["ms/sentence"], r["chrF"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency, ms / sentence (log)")
    ax.set_ylabel("chrF")
    ax.set_title("Quality–latency trade-off (bubble ∝ parameters)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pareto.png")
    plt.close(fig)


def plot_sentence_chrf(m):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    data, labels, colors = [], [], []
    for k, r in m.items():
        data.append(r["sentence_chrf"]); labels.append(r["label"]); colors.append(PALETTE[k])
    parts = ax.violinplot(data, showmedians=True, widths=.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(.7)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("sentence chrF")
    ax.set_title("Distribution of sentence-level chrF")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sentence_chrf.png")
    plt.close(fig)


def plot_fertility(df):
    d = df[~df.key.str.endswith("beam") & ~df.key.str.endswith("icl") & ~df.key.str.endswith("zs")]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(d))
    ax.bar(x - .2, d["Fertility en"], .4, label="English source", color="#94A3B8")
    ax.bar(x + .2, d["Fertility hi"], .4, label="Hindi reference", color="#F97316")
    ax.set_xticks(x, d.System, rotation=10, fontsize=8)
    ax.set_ylabel("subword tokens per word")
    ax.set_title("Tokenizer fertility (lower = more efficient)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fertility.png")
    plt.close(fig)


def plot_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    p = RESULTS_DIR / "train_transformer.json"
    if p.exists():
        h = pd.DataFrame(json.loads(p.read_text())["history"])
        axes[0].plot(h.epoch, h.train_loss, "o-", label="train (label-smoothed)")
        axes[0].plot(h.epoch, h.val_loss, "o-", label="validation")
        axes[0].set_title("Transformer (scratch) – loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
        ax2 = axes[0].twinx(); ax2.plot(h.epoch, h.val_chrf_greedy, "s--", color="#059669", alpha=.6)
        ax2.set_ylabel("val chrF", color="#059669")
    p = RESULTS_DIR / "train_mt5.json"
    if p.exists():
        h = pd.DataFrame(json.loads(p.read_text())["history"])
        axes[1].plot(h.epoch, h.val_chrf_greedy, "o-", color=PALETTE["mt5"], label="val chrF")
        axes[1].plot(h.epoch, h.val_bleu_greedy, "s-", color=PALETTE["mt5_beam"], label="val BLEU")
        axes[1].set_title("mT5-small – validation"); axes[1].set_xlabel("epoch"); axes[1].legend()
    for k in ("llama", "gemma"):
        p = RESULTS_DIR / f"train_{k}_lora.json"
        if p.exists():
            h = pd.DataFrame(json.loads(p.read_text())["history"])
            axes[2].plot(h.step, h.loss_ema, label=f"{k} LoRA", color=PALETTE[f"{k}_lora"])
    axes[2].set_title("LoRA fine-tuning – training loss (EMA)"); axes[2].set_xlabel("step"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_curves.png")
    plt.close(fig)


def plot_temperature(temps):
    if not temps:
        return
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for k, t in temps.items():
        d = pd.DataFrame(t["sweep"])
        kw = dict(marker="o", color=PALETTE[k], label=t["label"])
        axes[0].plot(d.temperature, d.chrf, **kw)
        axes[1].plot(d.temperature, d.bleu, **kw)
        axes[2].plot(d.temperature, d.hallucination_rate, **kw)
        axes[3].plot(d.temperature, d.self_bleu, **kw)
    for ax, t in zip(axes, ["chrF ↑", "BLEU ↑", "Hallucination-flag rate % ↓", "Self-BLEU of 2 samples (↓ more diverse)"]):
        ax.set_title(t); ax.set_xlabel("sampling temperature")
    axes[0].legend(fontsize=8)
    fig.suptitle("Sampling-temperature analysis (150-sentence subset)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "temperature.png")
    plt.close(fig)


def plot_gender(probes):
    rows = []
    for k, p in probes.items():
        g = p["gender"]
        rows.append({"key": k, "System": p["label"], "Accuracy": g["accuracy"], "ΔG (M−F)": g["delta_G"],
                     "ΔS (pro−anti)": g["delta_S"], "Ambiguous → masc %": g["ambiguous_masc_pct"]})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, col in zip(axes, ["Accuracy", "ΔG (M−F)", "ΔS (pro−anti)", "Ambiguous → masc %"]):
        _bar(ax, d, col, col)
        ax.axvline(0, color="#999", lw=.8)
    fig.suptitle("Gender-bias probe (WinoMT-style, Hindi verb agreement)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gender_bias.png")
    plt.close(fig)
    return d


def reference_baseline() -> dict:
    """Run the same reference-free guardrails on the *human* references: this is the
    false-positive floor against which every system's flag rate should be read."""
    from . import guardrails as G
    test = load_splits()["test"]
    sims = G.SemanticScorer("cpu")(test.en.tolist(), test.hi.tolist())
    reps = [G.check(s, h, x) for s, h, x in zip(test.en, test.hi, sims)]
    return {"label": "Human reference", "semantic_fidelity": float(np.mean(sims)),
            "sim_percentiles": {p: float(np.percentile(sims, p)) for p in (5, 10, 25, 50)},
            "hallucination_rate": float(np.mean([r.hallucination for r in reps]) * 100),
            "flag_rates": {k: float(np.mean([r.flags[k] for r in reps]) * 100) for k in reps[0].flags}}


def plot_flags(m, ref):
    rows = []
    for k, r in m.items():
        for f, v in r["flag_rates"].items():
            rows.append({"System": r["label"], "flag": f, "rate": v})
    for f, v in ref["flag_rates"].items():
        rows.append({"System": ref["label"], "flag": f, "rate": v})
    d = pd.DataFrame(rows).pivot(index="System", columns="flag", values="rate")
    d = d.reindex([ref["label"]] + [m[k]["label"] for k in m])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(d, annot=True, fmt=".1f", cmap="rocket_r", cbar_kws={"label": "% of test sentences"}, ax=ax)
    ax.set_title("Guardrail flag rates on the test set (reference-free)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "guardrail_flags.png")
    plt.close(fig)


def plot_data(stats):
    s = load_splits()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, c in (("train", "#2563EB"), ("test", "#F97316")):
        axes[0].hist(s[name].en.str.split().str.len(), bins=40, alpha=.6, density=True, color=c, label=name)
        axes[1].hist(s[name].hi.str.split().str.len(), bins=40, alpha=.6, density=True, color=c, label=name)
    axes[0].set_title("English sentence length (words)"); axes[1].set_title("Hindi sentence length (words)")
    for a in axes:
        a.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "data_lengths.png")
    plt.close(fig)


def main():
    m = load_metrics()
    df = main_table(m)
    df.drop(columns=["key"]).to_csv(RESULTS_DIR / "results_table.csv", index=False)
    temps = {k: json.loads((TEMP_DIR / f"{k}.json").read_text()) for k in PRIMARY if (TEMP_DIR / f"{k}.json").exists()}
    probes = {k: json.loads((PROBE_DIR / f"{k}.json").read_text()) for k in SYSTEMS
              if (PROBE_DIR / f"{k}.json").exists()}
    stats = json.loads((RESULTS_DIR / "data_stats.json").read_text())

    plot_quality(df); plot_compute(df); plot_pareto(df); plot_sentence_chrf(m)
    ref = reference_baseline()
    plot_fertility(df); plot_training_curves(); plot_temperature(temps); plot_flags(m, ref); plot_data(stats)
    gender_df = plot_gender(probes)
    sig = significance(m)

    # examples: same test sentences across systems (for qualitative analysis)
    trans = {}
    for k in m:
        rows = [json.loads(l) for l in (RESULTS_DIR / "translations" / f"{k}.jsonl").open()]
        trans[k] = rows
    example_ids = [3, 17, 42, 88, 120, 205, 333, 410]
    examples = [{"src": trans[next(iter(trans))][i]["src"], "ref": trans[next(iter(trans))][i]["ref"],
                 "hyps": {k: {"hyp": trans[k][i]["hyp"], "chrf": trans[k][i]["chrf"]} for k in trans}}
                for i in example_ids if i < len(trans[next(iter(trans))])]

    summary = {
        "dataset": {"id": "cfilt/iitb-english-hindi", "url": DATASET_URL, "homepage": DATASET_HOMEPAGE, **stats},
        "systems": {k: {kk: vv for kk, vv in r.items() if kk not in ("bleu_stats", "chrf_stats")}
                    for k, r in m.items()},
        "primary": PRIMARY, "significance": sig, "reference_baseline": ref, "temperature": temps,
        "probes": probes, "examples": examples,
        "training": {k: json.loads(p.read_text()) for k, p in
                     ((p.stem.replace("train_", ""), p) for p in RESULTS_DIR.glob("train_*.json"))},
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    md = df.drop(columns=["key"]).round(2).to_markdown(index=False)
    (RESULTS_DIR / "results_table.md").write_text(md)
    if len(gender_df):
        (RESULTS_DIR / "gender_table.md").write_text(gender_df.drop(columns=["key"]).round(1).to_markdown(index=False))
    print(md)
    print(json.dumps(sig, indent=1))


if __name__ == "__main__":
    main()
