"""Uniform wrapper around every translation system so they are evaluated identically."""
from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from typing import Callable

import torch

from .config import INFER_BATCH, RESULTS_DIR, get_device


@dataclass
class System:
    key: str
    label: str
    family: str                 # "encoder-decoder" | "decoder-only"
    base: str                   # underlying model
    adaptation: str
    pretrained: bool
    train_meta: str | None      # results/<file>.json written by the training script
    decoding: str = "greedy"

    def train_info(self) -> dict:
        if self.train_meta and (RESULTS_DIR / self.train_meta).exists():
            return json.loads((RESULTS_DIR / self.train_meta).read_text())
        return {}


SYSTEMS: dict[str, System] = {s.key: s for s in [
    System("transformer", "Transformer (scratch)", "encoder-decoder", "Custom 4+4-layer Transformer",
           "Trained from scratch", False, "train_transformer.json"),
    System("transformer_beam", "Transformer (scratch) + beam", "encoder-decoder",
           "Custom 4+4-layer Transformer", "Trained from scratch", False, "train_transformer.json", "beam=4"),
    System("mt5", "mT5-small (fine-tuned)", "encoder-decoder", "google/mt5-small",
           "Full fine-tuning", True, "train_mt5.json"),
    System("mt5_beam", "mT5-small (fine-tuned) + beam", "encoder-decoder", "google/mt5-small",
           "Full fine-tuning", True, "train_mt5.json", "beam=4"),
    System("llama_zs", "LLaMA-3.2-1B (zero-shot)", "decoder-only", "meta-llama/Llama-3.2-1B-Instruct",
           "Zero-shot prompting", True, None),
    System("llama_icl", "LLaMA-3.2-1B (kNN 4-shot)", "decoder-only", "meta-llama/Llama-3.2-1B-Instruct",
           "Retrieval few-shot prompting", True, None),
    System("llama_lora", "LLaMA-3.2-1B (LoRA)", "decoder-only", "meta-llama/Llama-3.2-1B-Instruct",
           "LoRA fine-tuning", True, "train_llama_lora.json"),
    System("gemma_zs", "Gemma-3-1B (zero-shot)", "decoder-only", "google/gemma-3-1b-it",
           "Zero-shot prompting", True, None),
    System("gemma_icl", "Gemma-3-1B (kNN 4-shot)", "decoder-only", "google/gemma-3-1b-it",
           "Retrieval few-shot prompting", True, None),
    System("gemma_lora", "Gemma-3-1B (LoRA)", "decoder-only", "google/gemma-3-1b-it",
           "LoRA fine-tuning", True, "train_gemma_lora.json"),
]}

# The four headline systems used for the main comparison / temperature sweep.
PRIMARY = ["transformer", "mt5", "llama_lora", "gemma_lora"]


class Loaded:
    """A loaded model exposing translate(sentences, temperature) and tokenizer stats."""

    def __init__(self, key: str, device=None, retriever=None):
        self.key, self.device = key, device or get_device()
        self.sys = SYSTEMS[key]
        self.retriever = retriever
        self.n_shots = 4
        if key.startswith("transformer"):
            from .transformer.train import load_trained
            self.model, self.tok = load_trained(self.device)
            self.params = sum(p.numel() for p in self.model.parameters())
            self.trainable = self.params
        elif key.startswith("mt5"):
            from .mt5_finetune import load_trained
            self.model, self.tok = load_trained(self.device)
            self.params = sum(p.numel() for p in self.model.parameters())
            self.trainable = self.params
        else:
            from . import decoder_llm as D
            short = key.split("_")[0]
            if key.endswith("lora"):
                self.model, self.tok = D.load_finetuned(short, self.device)
                info = self.sys.train_info()
                self.trainable = info.get("trainable_params", 0)
            else:
                self.model, self.tok = D.load_base(D.CONFIGS[short], self.device)
                self.trainable = 0
            self.params = sum(p.numel() for p in self.model.parameters())

    def translate(self, sentences: list[str], temperature: float = 0.0,
                  batch_size: int = INFER_BATCH) -> list[str]:
        k, dev = self.key, self.device
        if k.startswith("transformer"):
            from .transformer.train import translate
            beam = 4 if k.endswith("beam") and temperature == 0 else 1
            return translate(self.model, self.tok, sentences, dev, batch_size, beam=beam,
                             temperature=temperature)
        if k.startswith("mt5"):
            from .mt5_finetune import translate
            beam = 4 if k.endswith("beam") and temperature == 0 else 1
            return translate(self.model, self.tok, sentences, dev, batch_size, beam=beam,
                             temperature=temperature)
        from .decoder_llm import translate
        shots = self.retriever(sentences, self.n_shots) if k.endswith("icl") else None
        return translate(self.model, self.tok, sentences, dev, batch_size=batch_size, shots=shots,
                         temperature=temperature)

    def n_tokens(self, texts: list[str]) -> int:
        if self.key.startswith("transformer"):
            return sum(len(x) for x in self.tok.encode(texts))
        return sum(len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in texts)

    def close(self):
        del self.model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_translator(key: str) -> Callable[[list[str]], list[str]]:
    return Loaded(key).translate
