"""Central configuration for the English→Hindi translation benchmark.

Every experiment reads from here so that all four models are trained and
evaluated under comparable, reproducible settings.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
TRANSLATIONS_DIR = RESULTS_DIR / "translations"
PLOTS_DIR = RESULTS_DIR / "plots"
for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, TRANSLATIONS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SRC_LANG, TGT_LANG = "en", "hi"
SRC_NAME, TGT_NAME = "English", "Hindi"

DATASET_ID = "cfilt/iitb-english-hindi"
DATASET_URL = "https://huggingface.co/datasets/cfilt/iitb-english-hindi"
DATASET_HOMEPAGE = "https://www.cfilt.iitb.ac.in/iitb_parallel/"

SEED = 42


@dataclass
class DataConfig:
    # Manageable subset of the 1.66M-pair IITB corpus (permitted by the brief).
    train_size: int = 120_000
    val_size: int = 1_000
    test_size: int = 500
    min_words: int = 3
    max_words: int = 40
    max_len_ratio: float = 2.2      # |hi| / |en| word-ratio bound (both directions)
    min_devanagari_frac: float = 0.6


@dataclass
class TransformerConfig:
    vocab_size: int = 16_000         # joint SentencePiece unigram vocabulary
    d_model: int = 384
    n_heads: int = 6
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    d_ff: int = 1536
    dropout: float = 0.1
    max_len: int = 128
    label_smoothing: float = 0.1
    lr: float = 7e-4
    warmup_steps: int = 2_500
    batch_tokens: int = 8_000        # token-bucketed batching (padded tokens per batch)
    epochs: int = 14
    beam_size: int = 4
    length_penalty: float = 0.6


@dataclass
class MT5Config:
    model_id: str = "google/mt5-small"
    train_size: int = 32_000
    epochs: int = 1
    batch_size: int = 8
    grad_accum: int = 2              # effective batch 16
    lr: float = 1e-3                 # Adafactor, constant-scale (standard T5 fine-tuning LR)
    max_len: int = 96
    beam_size: int = 4
    prefix: str = "translate English to Hindi: "


@dataclass
class DecoderLLMConfig:
    model_id: str
    short: str
    train_size: int = 6_000
    epochs: int = 1
    batch_size: int = 8
    grad_accum: int = 2
    lr: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_len: int = 160
    max_new_tokens: int = 120
    n_shots: int = 4
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])


LLAMA = DecoderLLMConfig(model_id="unsloth/Llama-3.2-1B-Instruct", short="llama")
# Gemma's 262k-entry vocabulary doubles the logit memory → half the micro-batch, same effective batch.
GEMMA = DecoderLLMConfig(model_id="unsloth/gemma-3-1b-it", short="gemma", batch_size=4, grad_accum=4)

TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.3]
TEMP_SUBSET = 150          # sentences used in the sampling-temperature sweep
INFER_BATCH = 16           # identical inference batch size for every model


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if os.environ.get("FORCE_CPU"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
