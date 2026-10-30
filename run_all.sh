#!/usr/bin/env bash
# End-to-end reproduction: data → 4 models → evaluation → report → static dashboard.
# Usage:  ./run_all.sh            (≈3–4 h on an Apple M4 / ~1.5 h on a single T4 GPU)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
mkdir -p logs

step() { echo -e "\n=== $1  ($(date +%H:%M:%S))"; }

step "1/7 preprocess IITB English–Hindi";      $PY -m src.data                       2>&1 | tee logs/data.log
[ -f checkpoints/transformer/best.pt ] && [ -f results/train_transformer.json ] || {
step "2/7 train Transformer from scratch";     $PY -m src.transformer.train          2>&1 | tee logs/train_transformer.log; }
[ -f results/train_mt5.json ] || {
step "3/7 fine-tune mT5-small";                $PY -m src.mt5_finetune               2>&1 | tee logs/train_mt5.log; }
[ -f results/train_llama_lora.json ] || {
step "4/7 LoRA fine-tune LLaMA-3.2-1B";        $PY -m src.decoder_llm --model llama  2>&1 | tee logs/train_llama.log; }
[ -f results/train_gemma_lora.json ] || {
step "5/7 LoRA fine-tune Gemma-3-1B";          $PY -m src.decoder_llm --model gemma  2>&1 | tee logs/train_gemma.log; }
step "6/7 evaluate all systems";               $PY -m src.evaluate                   2>&1 | tee logs/evaluate.log
step "7/7 report, plots, static dashboard";    $PY -m src.report                     2>&1 | tee logs/report.log
$PY scripts/export_static.py
echo "done – open docs/index.html via a static server, or run: .venv/bin/uvicorn app.server:app"
