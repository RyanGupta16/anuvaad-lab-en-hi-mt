"""Train the from-scratch Transformer on the IITB subset.

  python -m src.transformer.train

Steps: joint SentencePiece unigram tokenizer → token-bucketed batches →
AdamW + inverse-sqrt (Noam) schedule with warm-up → label-smoothed CE →
per-epoch validation loss and chrF → best checkpoint kept (early-stopping style).
"""
from __future__ import annotations

import json
import math
import random
import time

import sacrebleu
import sentencepiece as spm
import torch
import torch.nn.functional as F

from ..config import CKPT_DIR, DATA_DIR, RESULTS_DIR, SEED, TransformerConfig, get_device, seed_everything
from ..data import build_splits
from .model import Seq2SeqTransformer, count_parameters

SPM_PREFIX = CKPT_DIR / "transformer" / "spm_enhi"
CKPT = CKPT_DIR / "transformer" / "best.pt"


def train_tokenizer(train_df, vocab_size: int) -> spm.SentencePieceProcessor:
    SPM_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    if not SPM_PREFIX.with_suffix(".model").exists():
        corpus = DATA_DIR / "spm_corpus.txt"
        with corpus.open("w") as f:
            for e, h in zip(train_df.en, train_df.hi):
                f.write(e + "\n" + h + "\n")
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(SPM_PREFIX), vocab_size=vocab_size,
            model_type="unigram", character_coverage=0.9995, pad_id=0, unk_id=1,
            bos_id=2, eos_id=3, input_sentence_size=240_000, shuffle_input_sentence=True,
            normalization_rule_name="nmt_nfkc", num_threads=8, minloglevel=2)
    return load_tokenizer()


def load_tokenizer() -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(SPM_PREFIX.with_suffix(".model")))


def encode_pairs(sp, df, max_len):
    src = sp.encode(df.en.tolist())
    tgt = sp.encode(df.hi.tolist())
    pairs = [(s[: max_len - 1] + [sp.eos_id()], [sp.bos_id()] + t[: max_len - 2] + [sp.eos_id()])
             for s, t in zip(src, tgt)]
    return pairs


def token_batches(pairs, batch_tokens, shuffle=True):
    """Bucket by length so each batch holds ~batch_tokens (padded) tokens."""
    order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][0]), len(pairs[i][1])))
    batches, cur, longest = [], [], 0
    for i in order:
        longest_new = max(longest, _round8(len(pairs[i][0])), _round8(len(pairs[i][1])))
        if cur and longest_new * (len(cur) + 1) > batch_tokens:
            batches.append(cur)
            cur, longest_new = [], max(_round8(len(pairs[i][0])), _round8(len(pairs[i][1])))
        cur.append(i)
        longest = longest_new
    if cur:
        batches.append(cur)
    if shuffle:
        random.shuffle(batches)
    return batches


def _round8(n):
    return (n + 7) // 8 * 8


def collate(pairs, idx, pad, device):
    # lengths rounded up to multiples of 8 → few distinct shapes → allocator re-uses blocks
    S = _round8(max(len(pairs[i][0]) for i in idx))
    T = _round8(max(len(pairs[i][1]) for i in idx) - 1) + 1
    src = torch.full((len(idx), S), pad, dtype=torch.long)
    tgt = torch.full((len(idx), T), pad, dtype=torch.long)
    for r, i in enumerate(idx):
        s, t = pairs[i]
        src[r, : len(s)] = torch.tensor(s)
        tgt[r, : len(t)] = torch.tensor(t)
    return src.to(device), tgt.to(device)


def noam_lambda(warmup: int):
    def f(step):
        step = max(step, 1)
        return min(step / warmup, math.sqrt(warmup / step))
    return f


def build_model(c: TransformerConfig, sp, device):
    return Seq2SeqTransformer(c, sp.pad_id(), sp.bos_id(), sp.eos_id()).to(device)


@torch.no_grad()
def translate(model, sp, sentences, device, batch_size=32, beam=4, alpha=0.6,
              temperature=0.0):
    model.eval()
    out = [None] * len(sentences)
    enc = [sp.encode(s)[: model.c.max_len - 1] + [sp.eos_id()] for s in sentences]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    for b in range(0, len(order), batch_size):
        idx = order[b: b + batch_size]
        S = max(len(enc[i]) for i in idx)
        src = torch.full((len(idx), S), sp.pad_id(), dtype=torch.long)
        for r, i in enumerate(idx):
            src[r, : len(enc[i])] = torch.tensor(enc[i])
        src = src.to(device)
        if temperature > 0 or beam <= 1:
            hyp = model.sample(src, temperature=temperature,
                               max_len=min(model.c.max_len, int(S * 1.6) + 10))
        else:
            hyp = model.beam_search(src, beam=beam, alpha=alpha)
        for r, i in enumerate(idx):
            ids = [t for t in hyp[r].tolist() if t not in (sp.pad_id(), sp.bos_id())]
            if sp.eos_id() in ids:
                ids = ids[: ids.index(sp.eos_id())]
            out[i] = sp.decode(ids)
    return out


def load_trained(device=None):
    device = device or get_device()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = TransformerConfig(**ck["config"])
    sp = load_tokenizer()
    model = build_model(c, sp, device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, sp


def main():
    seed_everything(SEED)
    c = TransformerConfig()
    device = get_device()
    splits = build_splits()
    sp = train_tokenizer(splits["train"], c.vocab_size)
    train_pairs = encode_pairs(sp, splits["train"], c.max_len)
    val_pairs = encode_pairs(sp, splits["val"], c.max_len)

    model = build_model(c, sp, device)
    n_params = count_parameters(model)
    print(f"Transformer parameters: {n_params/1e6:.2f}M on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, noam_lambda(c.warmup_steps))

    history, best_chrf, step = [], -1.0, 0
    val_subset = splits["val"].iloc[:300]
    t0 = time.time()
    for epoch in range(1, c.epochs + 1):
        model.train()
        tot_loss, tot_tok, e0 = 0.0, 0, time.time()
        for bi, idx in enumerate(token_batches(train_pairs, c.batch_tokens)):
            src, tgt = collate(train_pairs, idx, sp.pad_id(), device)
            gold = tgt[:, 1:]
            keep = gold != sp.pad_id()
            # project only real (non-pad) positions onto the vocabulary: ~40 % less memory
            hidden = model(src, tgt[:, :-1], project=False)[keep]
            loss = F.cross_entropy(model.project(hidden), gold[keep],
                                   label_smoothing=c.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            ntok = int(keep.sum().item())
            tot_loss += loss.item() * ntok
            tot_tok += ntok
            if device.type == "mps":   # free fragmented cache blocks (variable shapes)
                torch.mps.empty_cache()
            if bi % 25 == 0:
                print(f"  ep{epoch} step{step} loss={loss.item():.3f} lr={sched.get_last_lr()[0]:.2e} t={time.time()-t0:.0f}s mem={torch.mps.driver_allocated_memory()/1e9 if device.type=="mps" else 0:.1f}G shape={tuple(src.shape)}{tuple(tgt.shape)}", flush=True)

        # ---- validation
        model.eval()
        vloss, vtok = 0.0, 0
        with torch.no_grad():
            for idx in token_batches(val_pairs, c.batch_tokens, shuffle=False):
                src, tgt = collate(val_pairs, idx, sp.pad_id(), device)
                logits = model(src, tgt[:, :-1])
                gold = tgt[:, 1:]
                vloss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1),
                                         ignore_index=sp.pad_id(), reduction="sum").item()
                vtok += (gold != sp.pad_id()).sum().item()
        hyps = translate(model, sp, val_subset.en.tolist(), device, beam=1)
        chrf = sacrebleu.corpus_chrf(hyps, [val_subset.hi.tolist()]).score
        bleu = sacrebleu.corpus_bleu(hyps, [val_subset.hi.tolist()]).score
        rec = {"epoch": epoch, "train_loss": tot_loss / tot_tok, "val_loss": vloss / vtok,
               "val_ppl": math.exp(vloss / vtok), "val_bleu_greedy": bleu, "val_chrf_greedy": chrf,
               "epoch_sec": time.time() - e0, "lr": sched.get_last_lr()[0]}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        print("   sample:", val_subset.en.iloc[0], "→", hyps[0], flush=True)
        if chrf > best_chrf:
            best_chrf = chrf
            torch.save({"model": model.state_dict(), "config": c.__dict__, "epoch": epoch}, CKPT)

    train_time = time.time() - t0
    meta = {"model": "transformer", "params": n_params, "train_time_sec": train_time,
            "train_pairs": len(train_pairs), "device": str(device), "history": history,
            "best_val_chrf": best_chrf, "config": c.__dict__}
    (RESULTS_DIR / "train_transformer.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {train_time/60:.1f} min, best val chrF {best_chrf:.2f}")


if __name__ == "__main__":
    main()
