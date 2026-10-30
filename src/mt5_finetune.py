"""Fine-tune mT5-small (encoder–decoder, multilingual span-corruption pre-training).

  python -m src.mt5_finetune
"""
from __future__ import annotations

import json
import math
import random
import time

import sacrebleu
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

from .config import CKPT_DIR, RESULTS_DIR, SEED, MT5Config, get_device, seed_everything
from .data import build_splits

OUT = CKPT_DIR / "mt5"


def load_trained(device=None):
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(OUT)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUT, dtype=torch.float32).to(device).eval()
    return model, tok


@torch.no_grad()
def translate(model, tok, sentences, device, batch_size=16, beam=4, temperature=0.0,
              max_len=MT5Config.max_len, prefix=MT5Config.prefix):
    model.eval()
    out = [None] * len(sentences)
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    for b in range(0, len(order), batch_size):
        idx = order[b: b + batch_size]
        enc = tok([prefix + sentences[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        if temperature > 0:
            gen = model.generate(**enc, do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                                 max_new_tokens=max_len)
        else:
            gen = model.generate(**enc, num_beams=beam, max_new_tokens=max_len,
                                 length_penalty=1.0, early_stopping=True)
        for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = text.strip()
    return out


def token_loss(model, enc, pad_id):
    """Cross-entropy computed only on real target tokens: the 250k-way vocabulary projection
    is applied to non-pad decoder states instead of the full padded (B, T, V) tensor."""
    labels = enc["labels"]
    hidden = model.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    dec = model.decoder(input_ids=model._shift_right(labels), encoder_hidden_states=hidden,
                        encoder_attention_mask=enc["attention_mask"]).last_hidden_state
    keep = labels != pad_id
    return F.cross_entropy(model.lm_head(dec[keep]).float(), labels[keep])


def main():
    seed_everything(SEED)
    c = MT5Config()
    device = get_device()
    splits = build_splits()
    train = splits["train"].iloc[: c.train_size]
    val = splits["val"].iloc[:200]

    tok = AutoTokenizer.from_pretrained(c.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(c.model_id, dtype=torch.float32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"mT5 params {n_params/1e6:.1f}M on {device}")

    src = [c.prefix + s for s in train.en]
    tgt = train.hi.tolist()
    # length-sorted mega-batches (shuffled) keep padding low on the 250k-vocab model
    order = sorted(range(len(src)), key=lambda i: len(src[i]) + len(tgt[i]))
    batches = [order[i: i + c.batch_size] for i in range(0, len(order), c.batch_size)]

    # Adafactor (factored second moments, no momentum) – the optimiser T5/mT5 were pre-trained
    # with; it needs a fraction of AdamW's state memory, which matters for the 250k-row embeddings.
    opt = Adafactor(model.parameters(), lr=c.lr, scale_parameter=False, relative_step=False,
                    warmup_init=False, weight_decay=0.0)
    total = len(batches) * c.epochs // c.grad_accum
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)

    history, best, t0, step = [], -1.0, time.time(), 0
    for epoch in range(1, c.epochs + 1):
        random.shuffle(batches)
        model.train()
        run, e0 = 0.0, time.time()
        for bi, idx in enumerate(batches):
            enc = tok([src[i] for i in idx], text_target=[tgt[i] for i in idx], return_tensors="pt",
                      padding=True, truncation=True, max_length=c.max_len).to(device)
            loss = token_loss(model, enc, tok.pad_token_id)
            if not math.isfinite(loss.item()):
                print("non-finite loss, skipping batch")
                opt.zero_grad(set_to_none=True)
                continue
            (loss / c.grad_accum).backward()
            if (bi + 1) % c.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
            if device.type == "mps":   # 250k-vocab logits fragment the MPS cache quickly
                torch.mps.empty_cache()
            run = 0.98 * run + 0.02 * loss.item() if bi else loss.item()
            if bi % 200 == 0:
                print(f"  ep{epoch} {bi}/{len(batches)} loss={run:.3f}", flush=True)
        hyps = translate(model, tok, val.en.tolist(), device, beam=1)
        chrf = sacrebleu.corpus_chrf(hyps, [val.hi.tolist()]).score
        bleu = sacrebleu.corpus_bleu(hyps, [val.hi.tolist()]).score
        rec = {"epoch": epoch, "train_loss_ema": run, "val_bleu_greedy": bleu, "val_chrf_greedy": chrf,
               "epoch_sec": time.time() - e0}
        history.append(rec)
        print(json.dumps(rec), "| sample:", hyps[0], flush=True)
        if chrf > best:
            best = chrf
            model.save_pretrained(OUT)
            tok.save_pretrained(OUT)

    meta = {"model": "mt5", "params": n_params, "trainable_params": n_params,
            "train_time_sec": time.time() - t0, "train_pairs": len(src), "steps": step,
            "device": str(device), "history": history, "best_val_chrf": best, "config": c.__dict__}
    (RESULTS_DIR / "train_mt5.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
