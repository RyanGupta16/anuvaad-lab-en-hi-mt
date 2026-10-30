"""Dataset loading, cleaning and splitting for IITB English–Hindi.

Pipeline
--------
1. Unicode NFC normalisation, whitespace collapsing, Hindi danda normalisation.
2. Noise filters tuned for IITB (which mixes news, Wikipedia, religious text,
   government documents and GNOME/KDE UI strings):
     * 3–40 words per side, word-length ratio bound,
     * ≥60 % Devanagari letters on the Hindi side, no Devanagari on the English side,
     * drop software-localisation artefacts (%s, _File, <tags>, URLs, {0} …),
     * drop untranslated copies and exact duplicates.
3. Seeded sampling of a manageable training subset; official dev set → validation;
   seeded 500-sentence sample of the official (news-domain) test set → test.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

import pandas as pd
from datasets import load_dataset

from .config import DATA_DIR, DATASET_ID, RESULTS_DIR, SEED, DataConfig

_WS = re.compile(r"\s+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_ARTEFACT = re.compile(r"(%[sd]|\{\d*\}|<[^>]+>|https?://|www\.|_[A-Za-z]|&[a-z]+;|\\n|\|)")


def normalise_en(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return _WS.sub(" ", text).strip()


def normalise_hi(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _WS.sub(" ", text).strip()
    text = text.replace(" ।", "।").replace("|", "।")
    # IITB references inconsistently end Hindi sentences with '.'; use the danda.
    if text.endswith(".") and not text.endswith(".."):
        text = text[:-1].rstrip() + "।"
    return text


def _devanagari_fraction(text: str) -> float:
    letters = [c for c in text if c.isalpha() or _DEVANAGARI.match(c)]
    if not letters:
        return 0.0
    return sum(bool(_DEVANAGARI.match(c)) for c in letters) / len(letters)


def keep_pair(en: str, hi: str, cfg: DataConfig) -> bool:
    ne, nh = len(en.split()), len(hi.split())
    if not (cfg.min_words <= ne <= cfg.max_words and cfg.min_words <= nh <= cfg.max_words + 15):
        return False
    ratio = nh / max(ne, 1)
    if ratio > cfg.max_len_ratio or ratio < 1 / cfg.max_len_ratio:
        return False
    if _ARTEFACT.search(en) or _ARTEFACT.search(hi):
        return False
    if _DEVANAGARI.search(en) or not _LATIN.search(en):
        return False
    if _devanagari_fraction(hi) < cfg.min_devanagari_frac:
        return False
    return en.lower() != hi.lower()


def _to_frame(split, cfg: DataConfig | None, filter_: bool) -> pd.DataFrame:
    rows = split["translation"]
    en = [normalise_en(r["en"]) for r in rows]
    hi = [normalise_hi(r["hi"]) for r in rows]
    df = pd.DataFrame({"en": en, "hi": hi})
    df = df[(df.en.str.len() > 0) & (df.hi.str.len() > 0)]
    if filter_:
        df = df[[keep_pair(e, h, cfg) for e, h in zip(df.en, df.hi)]]
    return df.drop_duplicates(subset=["en"]).reset_index(drop=True)


def build_splits(cfg: DataConfig = DataConfig(), force: bool = False) -> dict[str, pd.DataFrame]:
    paths = {s: DATA_DIR / f"{s}.jsonl" for s in ("train", "val", "test")}
    if not force and all(p.exists() for p in paths.values()):
        return load_splits()

    raw = load_dataset(DATASET_ID)
    raw_counts = {k: len(v) for k, v in raw.items()}

    train_full = _to_frame(raw["train"], cfg, filter_=True)
    val = _to_frame(raw["validation"], cfg, filter_=False)
    test_full = _to_frame(raw["test"], cfg, filter_=False)
    test_full = test_full[[cfg.min_words <= len(e.split()) <= cfg.max_words for e in test_full.en]]
    test = test_full.sample(n=cfg.test_size, random_state=SEED).reset_index(drop=True)

    # Leakage guard: no test/val source sentence may appear in training.
    held_out = set(val.en) | set(test.en)
    train_full = train_full[~train_full.en.isin(held_out)]
    train = train_full.sample(n=min(cfg.train_size, len(train_full)),
                              random_state=SEED).reset_index(drop=True)

    splits = {"train": train, "val": val, "test": test}
    for name, df in splits.items():
        df.to_json(paths[name], orient="records", lines=True, force_ascii=False)

    stats = describe(splits)
    stats["raw_counts"] = raw_counts
    stats["after_filtering_train_pool"] = len(train_full)
    stats["filter_retention_pct"] = round(100 * len(train_full) / raw_counts["train"], 2)
    (RESULTS_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return splits


def load_splits() -> dict[str, pd.DataFrame]:
    out = {}
    for s in ("train", "val", "test"):
        out[s] = pd.read_json(DATA_DIR / f"{s}.jsonl", lines=True, dtype=str)
    return out


def describe(splits: dict[str, pd.DataFrame]) -> dict:
    stats = {}
    for name, df in splits.items():
        en_len = df.en.str.split().str.len()
        hi_len = df.hi.str.split().str.len()
        stats[name] = {
            "pairs": int(len(df)),
            "en_words_mean": round(float(en_len.mean()), 2),
            "hi_words_mean": round(float(hi_len.mean()), 2),
            "en_words_max": int(en_len.max()),
            "hi_words_max": int(hi_len.max()),
            "en_vocab": len(Counter(w.lower() for s in df.en for w in s.split())),
            "hi_vocab": len(Counter(w for s in df.hi for w in s.split())),
        }
    return stats


if __name__ == "__main__":
    s = build_splits(force=True)
    print(json.dumps(json.loads((RESULTS_DIR / "data_stats.json").read_text()), indent=2))
    print(s["train"].sample(5, random_state=1).to_string())
