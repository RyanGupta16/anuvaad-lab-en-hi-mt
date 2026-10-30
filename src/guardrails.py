"""Responsible-MT guardrails: reference-free checks that flag risky translations.

These run on (source, hypothesis) pairs only, so they work at deployment time when no
reference translation exists. They power the hallucination statistics in the
evaluation and the warning badges in the web demo.

Checks
------
* number_mismatch – digits/dates/amounts in the source missing or altered in the output
* negation_lost   – source contains a negation but output has no Hindi negator
* repetition      – degenerate loops (a word or 3-gram repeated), a classic LLM failure
* length_anomaly  – output far shorter/longer than the source (omission / addition)
* untranslated    – output is mostly Latin script (copying instead of translating)
* wrong_script    – output drifts into another script (Bengali, Cyrillic, …) – off-target generation
* low_fidelity    – cross-lingual embedding similarity below threshold (semantic drift)
* sensitive_domain – medical / legal / religious / violence / self-harm content that
                     should always receive human review before use
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUM = re.compile(r"\d+(?:[.,:]\d+)*")
_LATIN = re.compile(r"[A-Za-z]")
_DEVA = re.compile(r"[ऀ-ॿ]")
# other Indic scripts (Bengali … Sinhala), Cyrillic, Arabic, CJK – off-target token leakage
_OTHER_INDIC = re.compile("[ঀ-෿Ѐ-ӿ؀-ۿ぀-ヿ一-鿿]")
EN_NEG = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|cannot|without)\b|n't\b", re.I)
HI_NEG_WORDS = {"नहीं", "नही", "मत", "न", "ना", "बिना", "नहीँ"}
_PUNCT = re.compile(r"[।,.!?;:\"'()\-]")

SENSITIVE = {
    "medical": r"\b(mg|tablet|dose|dosage|medicine|drug|pregnan\w*|vaccine|symptom\w*|diagnos\w*|doctor|hospital|paracetamol|insulin|surgery)\b",
    "legal": r"\b(court|bail|guilty|accused|law|lawyer|verdict|sentence[d]?|arrest\w*|police|judge|contract|FIR)\b",
    "religion_caste": r"\b(caste|dalit|brahmin|hindu|muslim|islam|christian|sikh|religion\w*|temple|mosque|church|god|allah|beef|cow)\b",
    "violence": r"\b(riot\w*|kill\w*|murder\w*|attack\w*|bomb\w*|terror\w*|weapon\w*|injur\w*|dead|death)\b",
    "self_harm": r"\b(suicid\w*|self-harm|kill (myself|himself|herself)|helpline|depress\w*)\b",
}
SENSITIVE_RE = {k: re.compile(v, re.I) for k, v in SENSITIVE.items()}


def numbers(text: str) -> Counter:
    text = text.translate(DEVANAGARI_DIGITS)
    out = Counter()
    for n in _NUM.findall(text):
        out[re.sub(r"[.,:]", "", n).lstrip("0") or "0"] += 1
    return out


def has_repetition(text: str) -> bool:
    w = _PUNCT.sub(" ", text).split()
    if any(w[i] == w[i + 1] == w[i + 2] for i in range(len(w) - 2)):
        return True
    grams = Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2))
    return any(c >= 3 for c in grams.values())


def has_hindi_negation(text: str) -> bool:
    return bool(HI_NEG_WORDS & set(_PUNCT.sub(" ", text).split()))


def latin_fraction(text: str) -> float:
    lat, dev = len(_LATIN.findall(text)), len(_DEVA.findall(text))
    return lat / max(lat + dev, 1)


def sensitive_domains(src: str) -> list[str]:
    return [k for k, r in SENSITIVE_RE.items() if r.search(src)]


@dataclass
class Report:
    flags: dict[str, bool] = field(default_factory=dict)
    domains: list[str] = field(default_factory=list)
    similarity: float | None = None

    @property
    def hallucination(self) -> bool:
        f = self.flags
        return any(f.get(k) for k in ("repetition", "length_anomaly", "low_fidelity", "empty",
                                      "number_mismatch", "wrong_script"))

    @property
    def risk(self) -> str:
        n = sum(self.flags.values())
        if n >= 2 or self.flags.get("low_fidelity") or self.flags.get("negation_lost"):
            return "high"
        if n == 1 or self.domains:
            return "medium"
        return "low"

    def as_dict(self):
        return {"flags": self.flags, "domains": self.domains, "similarity": self.similarity,
                "hallucination": self.hallucination, "risk": self.risk,
                "needs_human_review": self.risk == "high" or bool(self.domains)}


def check(src: str, hyp: str, similarity: float | None = None, sim_threshold: float = 0.55) -> Report:
    hyp = hyp or ""
    ns, nh = numbers(src), numbers(hyp)
    ratio = len(hyp.split()) / max(len(src.split()), 1)
    flags = {
        "empty": not hyp.strip(),
        "number_mismatch": bool(ns) and ns != nh,
        "negation_lost": bool(EN_NEG.search(src)) and not has_hindi_negation(hyp),
        "repetition": has_repetition(hyp),
        "length_anomaly": bool(hyp.strip()) and (ratio < 0.45 or ratio > 2.6),
        "untranslated": latin_fraction(hyp) > 0.5,
        "wrong_script": len(_OTHER_INDIC.findall(hyp)) > 2,
        "low_fidelity": similarity is not None and similarity < sim_threshold,
    }
    return Report(flags=flags, domains=sensitive_domains(src), similarity=similarity)


class SemanticScorer:
    """Cross-lingual cosine similarity (multilingual MiniLM) – reference-free fidelity."""

    def __init__(self, device="cpu",
                 model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device=str(device))

    def __call__(self, src: list[str], hyp: list[str]) -> list[float]:
        a = self.m.encode(src, convert_to_tensor=True, normalize_embeddings=True, batch_size=64,
                          show_progress_bar=False)
        b = self.m.encode([h or " " for h in hyp], convert_to_tensor=True, normalize_embeddings=True,
                          batch_size=64, show_progress_bar=False)
        return (a * b).sum(-1).cpu().tolist()
