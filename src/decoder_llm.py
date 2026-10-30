"""Decoder-only LLMs (LLaMA-3.2-1B-Instruct, Gemma-3-1B-IT) for English→Hindi MT.

Two adaptation methods are applied to each model:

1. **Retrieval-augmented few-shot prompting (kNN-ICL)** – no weight updates. For every
   test sentence, the k most similar training pairs (cosine similarity of multilingual
   MiniLM sentence embeddings) are inserted as chat demonstrations.
2. **LoRA parameter-efficient fine-tuning** – low-rank adapters (r=16) on every attention
   and MLP projection, trained on completion tokens only (prompt tokens masked).

  python -m src.decoder_llm --model llama --stage all
  python -m src.decoder_llm --model gemma --stage all
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .config import (CKPT_DIR, GEMMA, LLAMA, RESULTS_DIR, SEED, DecoderLLMConfig, get_device,
                     seed_everything)
from .data import build_splits

SYSTEM = ("You are a professional English-to-Hindi translator. Translate the user's English "
          "sentence into fluent, faithful Hindi written in Devanagari. Preserve names, numbers "
          "and meaning exactly; do not add or omit information. Output only the Hindi translation.")
CONFIGS = {"llama": LLAMA, "gemma": GEMMA}
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVAL_POOL = 30_000


def user_turn(en: str) -> str:
    return f"English: {en}\nHindi:"


def messages_for(en: str, shots: list[tuple[str, str]] | None = None, hi: str | None = None):
    msgs = [{"role": "system", "content": SYSTEM}]
    for s_en, s_hi in shots or []:
        msgs += [{"role": "user", "content": user_turn(s_en)}, {"role": "assistant", "content": s_hi}]
    msgs.append({"role": "user", "content": user_turn(en)})
    if hi is not None:
        msgs.append({"role": "assistant", "content": hi})
    return msgs


def render(tok, msgs, add_generation_prompt: bool) -> str:
    # fixed date keeps the LLaMA system header deterministic across runs
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   date_string="01 Jan 2025")


def clean_output(text: str) -> str:
    text = text.strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    text = re.sub(r"^(Hindi|हिंदी|हिन्दी|Translation|अनुवाद)\s*[:：]\s*", "", text, flags=re.I)
    return text.strip().strip('"“”\'').strip()


def stop_ids(tok) -> list[int]:
    ids = {tok.eos_token_id}
    for t in ("<end_of_turn>", "<eos>", "<|eot_id|>", "<|end_of_text|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid != tok.unk_token_id and tid is not None:
            ids.add(tid)
    return sorted(i for i in ids if i is not None)


# ---------------------------------------------------------------- model loading
def load_base(cfg: DecoderLLMConfig, device):
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16, attn_implementation="eager" if "gemma" in cfg.model_id else "sdpa"
    ).to(device)
    model.eval()
    return model, tok


def load_finetuned(short: str, device=None, merge: bool = True):
    from peft import PeftModel
    device = device or get_device()
    cfg = CONFIGS[short]
    model, tok = load_base(cfg, device)
    model = PeftModel.from_pretrained(model, CKPT_DIR / f"{short}_lora")
    if merge:
        model = model.merge_and_unload()
    model.eval()
    return model, tok


# ---------------------------------------------------------------- kNN demonstrations
class ShotRetriever:
    """Retrieves the k most similar training pairs to use as in-context demonstrations."""

    def __init__(self, train_df, device, pool: int = RETRIEVAL_POOL):
        from sentence_transformers import SentenceTransformer
        self.df = train_df.iloc[:pool].reset_index(drop=True)
        # Prefer short demonstrations: they carry the pattern at a fraction of prompt cost.
        self.df = self.df[self.df.en.str.split().str.len() <= 25].reset_index(drop=True)
        self.enc = SentenceTransformer(EMBED_MODEL, device=str(device))
        cache = CKPT_DIR / f"retrieval_emb_{len(self.df)}.pt"
        if cache.exists():
            self.emb = torch.load(cache).to(device)
        else:
            self.emb = self.enc.encode(self.df.en.tolist(), batch_size=256, convert_to_tensor=True,
                                       normalize_embeddings=True, show_progress_bar=False)
            torch.save(self.emb.cpu(), cache)
            self.emb = self.emb.to(device)

    def __call__(self, sentences: list[str], k: int) -> list[list[tuple[str, str]]]:
        q = self.enc.encode(sentences, convert_to_tensor=True, normalize_embeddings=True,
                            show_progress_bar=False).to(self.emb.device)
        top = (q @ self.emb.T).topk(k, dim=-1).indices.cpu().tolist()
        # most-similar demonstration placed last (closest to the query) – recency helps ICL
        return [[(self.df.en[j], self.df.hi[j]) for j in reversed(row)] for row in top]


# ---------------------------------------------------------------- generation
@torch.no_grad()
def translate(model, tok, sentences, device, batch_size=16, shots=None, temperature=0.0,
              max_new_tokens=120):
    """Batched left-padded generation. `shots` is a per-sentence list of demonstrations."""
    tok.padding_side = "left"
    prompts = [render(tok, messages_for(s, shots[i] if shots else None), True)
               for i, s in enumerate(sentences)]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out, eos = [None] * len(prompts), stop_ids(tok)
    for b in range(0, len(order), batch_size):
        idx = order[b: b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        kw = dict(max_new_tokens=max_new_tokens, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
        else:
            kw.update(do_sample=False, num_beams=1)
            kw.update(temperature=None, top_p=None, top_k=None)
        gen = model.generate(**enc, **kw)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            out[i] = clean_output(t)
    return out


# ---------------------------------------------------------------- LoRA fine-tuning
def build_example(tok, en, hi, max_len):
    prompt = render(tok, messages_for(en), True)
    full = render(tok, messages_for(en, hi=hi), False)
    assert full.startswith(prompt), "chat template prefix mismatch"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = (p_ids + c_ids)[:max_len]
    labels = ([-100] * len(p_ids) + c_ids)[:max_len]
    return ids, labels


def finetune(short: str):
    from peft import LoraConfig, get_peft_model
    seed_everything(SEED)
    cfg, device = CONFIGS[short], get_device()
    splits = build_splits()
    # use pairs disjoint from the retrieval pool's head so LoRA and ICL see different data
    train = splits["train"].iloc[RETRIEVAL_POOL: RETRIEVAL_POOL + cfg.train_size]
    val = splits["val"].iloc[:100]

    model, tok = load_base(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{short}: total {total_params/1e6:.1f}M, LoRA trainable {trainable/1e6:.2f}M "
          f"({100*trainable/total_params:.2f}%)", flush=True)

    data = [build_example(tok, e, h, cfg.max_len) for e, h in zip(train.en, train.hi)]
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    batches = [order[i: i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
    random.shuffle(batches)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    updates = math.ceil(len(batches) * cfg.epochs / cfg.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * updates), updates)

    model.train()
    history, t0, ema = [], time.time(), None
    for epoch in range(cfg.epochs):
        for bi, idx in enumerate(batches):
            L = max(len(data[i][0]) for i in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id)
            lab = torch.full((len(idx), L), -100)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                x, y = data[i]
                ids[r, : len(x)], lab[r, : len(y)], att[r, : len(x)] = torch.tensor(x), torch.tensor(y), 1
            loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
            (loss / cfg.grad_accum).backward()
            if (bi + 1) % cfg.grad_accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if device.type == "mps":
                torch.mps.empty_cache()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if bi % 50 == 0:
                print(f"  {short} ep{epoch} {bi}/{len(batches)} loss={ema:.3f} "
                      f"{time.time()-t0:.0f}s", flush=True)
                history.append({"step": bi + epoch * len(batches), "loss_ema": ema})
    train_time = time.time() - t0

    out_dir = CKPT_DIR / f"{short}_lora"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    import sacrebleu
    model.eval()
    model.config.use_cache = True
    hyps = translate(model, tok, val.en.tolist(), device, batch_size=16)
    meta = {"model": short, "base_model": cfg.model_id, "params": total_params,
            "trainable_params": trainable, "train_time_sec": train_time, "train_pairs": len(data),
            "device": str(device), "history": history,
            "val_bleu": sacrebleu.corpus_bleu(hyps, [val.hi.tolist()]).score,
            "val_chrf": sacrebleu.corpus_chrf(hyps, [val.hi.tolist()]).score,
            "config": {k: v for k, v in cfg.__dict__.items()}}
    (RESULTS_DIR / f"train_{short}_lora.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("val_bleu", "val_chrf", "train_time_sec")}), "| sample:", hyps[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CONFIGS), required=True)
    args = ap.parse_args()
    finetune(args.model)
