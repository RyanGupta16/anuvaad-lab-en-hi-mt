"""Anuvaad Lab – FastAPI backend for the live translation arena and results dashboard.

  .venv/bin/uvicorn app.server:app --port 8000
  open http://localhost:8000

Models are loaded lazily on first use and kept in an LRU cache (at most two 1B-parameter
LLMs resident at a time) so the server fits in 16 GB of unified memory.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import guardrails as G
from src.config import CKPT_DIR, RESULTS_DIR
from src.systems import SYSTEMS, Loaded

STATIC = Path(__file__).parent / "static"
LIVE_SYSTEMS = ["transformer", "mt5", "llama_lora", "gemma_lora", "llama_icl", "gemma_icl",
                "llama_zs", "gemma_zs"]
CKPT_FOR = {"transformer": CKPT_DIR / "transformer" / "best.pt", "mt5": CKPT_DIR / "mt5",
            "llama_lora": CKPT_DIR / "llama_lora", "gemma_lora": CKPT_DIR / "gemma_lora",
            "llama_icl": None, "gemma_icl": None, "llama_zs": None, "gemma_zs": None}
MAX_LLMS = 2

app = FastAPI(title="Anuvaad Lab – EN→HI MT benchmark")
_cache: OrderedDict[str, Loaded] = OrderedDict()
_lock = threading.Lock()
_scorer = None
_retriever = None


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=600)
    systems: list[str] = Field(default_factory=lambda: ["transformer", "mt5", "llama_lora", "gemma_lora"])
    temperature: float = Field(0.0, ge=0.0, le=1.5)


def available(key: str) -> bool:
    p = CKPT_FOR.get(key)
    return p is None or p.exists()


def get_system(key: str) -> Loaded:
    global _retriever
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        llms = [k for k in _cache if "llama" in k or "gemma" in k]
        if ("llama" in key or "gemma" in key) and len(llms) >= MAX_LLMS:
            _cache.pop(llms[0]).close()
        if key.endswith("icl") and _retriever is None:
            from src.data import load_splits
            from src.decoder_llm import ShotRetriever
            from src.config import get_device
            _retriever = ShotRetriever(load_splits()["train"], get_device())
        _cache[key] = Loaded(key, retriever=_retriever)
        return _cache[key]


def scorer():
    global _scorer
    if _scorer is None:
        _scorer = G.SemanticScorer("cpu")
    return _scorer


@app.get("/api/status")
def status():
    return {"systems": [{"key": k, "label": SYSTEMS[k].label, "family": SYSTEMS[k].family,
                         "available": available(k), "loaded": k in _cache} for k in LIVE_SYSTEMS]}


@app.get("/api/summary")
def summary():
    p = RESULTS_DIR / "summary.json"
    if not p.exists():
        raise HTTPException(404, "Run `python -m src.report` to build results/summary.json first.")
    return FileResponse(p, media_type="application/json")


@app.get("/api/translations/{key}")
def translations(key: str):
    p = RESULTS_DIR / "translations" / f"{key}.jsonl"
    if key not in SYSTEMS or not p.exists():
        raise HTTPException(404, f"No test-set translations for '{key}'.")
    return JSONResponse([json.loads(l) for l in p.open()])


@app.post("/api/translate")
def translate(req: TranslateRequest):
    out = []
    for key in req.systems:
        if key not in LIVE_SYSTEMS or not available(key):
            out.append({"key": key, "error": "Model checkpoint not found. Train it first."})
            continue
        sysm = get_system(key)
        t0 = time.perf_counter()
        hyp = sysm.translate([req.text.strip()], temperature=req.temperature, batch_size=1)[0]
        ms = 1000 * (time.perf_counter() - t0)
        sim = scorer()([req.text], [hyp])[0]
        out.append({"key": key, "label": sysm.sys.label, "hyp": hyp, "ms": ms,
                    **G.check(req.text, hyp, sim).as_dict()})
    return {"results": out}


@app.get("/summary.json")
def summary_static():
    return summary()


app.mount("/plots", StaticFiles(directory=RESULTS_DIR / "plots"), name="plots")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
