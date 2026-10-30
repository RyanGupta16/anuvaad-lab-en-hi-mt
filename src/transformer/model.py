"""A from-scratch encoder–decoder Transformer (Vaswani et al., 2017) in PyTorch.

Modernised details (all standard in current NMT systems):
  * Pre-LayerNorm residual blocks  → stable training without long warm-up tricks.
  * Fused scaled-dot-product attention (FlashAttention / memory-efficient kernels
    where the backend supports them).
  * Three-way weight tying: encoder embedding = decoder embedding = output projection
    (possible because the SentencePiece vocabulary is shared across en+hi).
  * Incremental key/value cache in the decoder → O(n) work per generated token.
  * Batched beam search with GNMT length penalty, plus temperature sampling.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TransformerConfig


class SinusoidalPositions(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return x + self.pe[offset: offset + x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, self_attn: bool):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(d_model, 3 * d_model)
        else:
            self.q = nn.Linear(d_model, d_model)
            self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split(self, x):  # (B, T, D) -> (B, H, T, dk)
        return x.view(x.size(0), x.size(1), self.h, self.dk).transpose(1, 2)

    def forward(self, x, memory=None, mask=None, is_causal=False, cache=None):
        if self.self_attn:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q, k, v = self._split(q), self._split(k), self._split(v)
            if cache is not None:                       # incremental decoding
                if "k" in cache:
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
                is_causal = False                       # single new query sees the whole prefix
        else:
            q = self._split(self.q(x))
            if cache is not None and "k" in cache:      # encoder memory is static
                k, v = cache["k"], cache["v"]
            else:
                k, v = self.kv(memory).chunk(2, dim=-1)
                k, v = self._split(k), self._split(v)
                if cache is not None:
                    cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        return self.out(y)


class FeedForward(nn.Sequential):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(d_ff, d_model))


class EncoderLayer(nn.Module):
    def __init__(self, c: TransformerConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.attn = MultiHeadAttention(c.d_model, c.n_heads, c.dropout, self_attn=True)
        self.ff = FeedForward(c.d_model, c.d_ff, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, src_mask):
        x = x + self.drop(self.attn(self.ln1(x), mask=src_mask))
        return x + self.drop(self.ff(self.ln2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, c: TransformerConfig):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(c.d_model) for _ in range(3))
        self.self_attn = MultiHeadAttention(c.d_model, c.n_heads, c.dropout, self_attn=True)
        self.cross_attn = MultiHeadAttention(c.d_model, c.n_heads, c.dropout, self_attn=False)
        self.ff = FeedForward(c.d_model, c.d_ff, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, y, memory, mem_mask, cache=None):
        sc = cc = None
        if cache is not None:
            sc, cc = cache.setdefault("self", {}), cache.setdefault("cross", {})
        y = y + self.drop(self.self_attn(self.ln1(y), is_causal=cache is None, cache=sc))
        y = y + self.drop(self.cross_attn(self.ln2(y), memory=memory, mask=mem_mask, cache=cc))
        return y + self.drop(self.ff(self.ln3(y)))


class Seq2SeqTransformer(nn.Module):
    def __init__(self, c: TransformerConfig, pad_id: int, bos_id: int, eos_id: int):
        super().__init__()
        self.c, self.pad_id, self.bos_id, self.eos_id = c, pad_id, bos_id, eos_id
        self.embed = nn.Embedding(c.vocab_size, c.d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositions(c.d_model, max_len=c.max_len + 8)
        self.emb_drop = nn.Dropout(c.dropout)
        self.encoder = nn.ModuleList(EncoderLayer(c) for _ in range(c.n_enc_layers))
        self.decoder = nn.ModuleList(DecoderLayer(c) for _ in range(c.n_dec_layers))
        self.enc_ln, self.dec_ln = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.scale = math.sqrt(c.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=m.embedding_dim ** -0.5)

    def _embed(self, ids, offset=0):
        return self.emb_drop(self.pos(self.embed(ids) * self.scale, offset))

    def _pad_mask(self, src):  # (B,1,1,S) boolean: True = attend
        return (src != self.pad_id)[:, None, None, :]

    def encode(self, src):
        mask = self._pad_mask(src)
        x = self._embed(src)
        for layer in self.encoder:
            x = layer(x, mask)
        return self.enc_ln(x), mask

    def decode(self, tgt_in, memory, mem_mask, caches=None, offset=0, project=True):
        y = self._embed(tgt_in, offset)
        for i, layer in enumerate(self.decoder):
            y = layer(y, memory, mem_mask, None if caches is None else caches[i])
        y = self.dec_ln(y)
        return self.project(y) if project else y

    def project(self, h):
        return F.linear(h, self.embed.weight)                  # tied output projection

    def forward(self, src, tgt_in, project=True):
        memory, mask = self.encode(src)
        return self.decode(tgt_in, memory, mask, project=project)

    # ------------------------------------------------------------------ decoding
    @staticmethod
    def _reorder(caches, idx):
        for layer in caches:
            for kind in ("self", "cross"):
                for key in ("k", "v"):
                    if key in layer.get(kind, {}):
                        layer[kind][key] = layer[kind][key].index_select(0, idx)

    @torch.no_grad()
    def sample(self, src, max_len=None, temperature: float = 0.0, top_p: float = 1.0):
        """Greedy decoding (temperature=0) or temperature / nucleus sampling."""
        max_len = max_len or self.c.max_len
        B = src.size(0)
        memory, mask = self.encode(src)
        caches = [{} for _ in self.decoder]
        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for t in range(max_len):
            logits = self.decode(ys[:, -1:], memory, mask, caches, offset=t)[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sp, si = probs.sort(descending=True)
                    keep = sp.cumsum(-1) - sp < top_p
                    sp = sp * keep
                    nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)).squeeze(-1)
                else:
                    nxt = torch.multinomial(probs, 1).squeeze(-1)
            nxt = nxt.masked_fill(done, self.pad_id)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= nxt == self.eos_id
            if done.all():
                break
        return ys[:, 1:]

    @torch.no_grad()
    def beam_search(self, src, beam: int = 4, alpha: float = 0.6, max_len=None):
        """Batched beam search with KV cache and GNMT length penalty ((5+|y|)/6)^α."""
        max_len = max_len or min(self.c.max_len, int(src.size(1) * 1.6) + 10)
        B, dev = src.size(0), src.device
        memory, mask = self.encode(src)
        memory = memory.repeat_interleave(beam, 0)
        mask = mask.repeat_interleave(beam, 0)
        caches = [{} for _ in self.decoder]

        seqs = torch.full((B * beam, 1), self.bos_id, dtype=torch.long, device=dev)
        scores = torch.zeros(B, beam, device=dev)
        scores[:, 1:] = -1e9                       # only first beam alive at t=0
        scores = scores.view(-1)
        finished = torch.zeros(B * beam, dtype=torch.bool, device=dev)
        V = self.c.vocab_size

        for t in range(max_len):
            logp = F.log_softmax(self.decode(seqs[:, -1:], memory, mask, caches, offset=t)[:, -1], -1)
            # finished hypotheses can only emit PAD at zero cost
            logp[finished] = -1e9
            logp[finished, self.pad_id] = 0.0
            cand = (scores[:, None] + logp).view(B, beam * V)
            top_s, top_i = cand.topk(beam, dim=-1)
            src_beam = top_i // V
            tok = top_i % V
            base = (torch.arange(B, device=dev) * beam)[:, None]
            sel = (base + src_beam).view(-1)
            self._reorder(caches, sel)
            seqs = torch.cat([seqs.index_select(0, sel), tok.view(-1, 1)], dim=1)
            finished = finished.index_select(0, sel) | (tok.view(-1) == self.eos_id)
            scores = top_s.view(-1)
            if finished.all():
                break

        lengths = ((seqs[:, 1:] != self.pad_id) & (seqs[:, 1:] != self.eos_id)).sum(1).float() + 1
        lp = ((5 + lengths) / 6) ** alpha
        best = (scores / lp).view(B, beam).argmax(-1)
        idx = torch.arange(B, device=dev) * beam + best
        return seqs.index_select(0, idx)[:, 1:]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
