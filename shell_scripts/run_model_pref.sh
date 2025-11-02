#!/usr/bin/env bash

set -euo pipefail

# Configuration
PYTHON_BIN="python"
INPUTS="Data/ag_news/valid.jsonl"
LABELS=("world" "sports" "business" "sci/tech")
MODEL="google/gemma-3-1b-it"
OUT_JSON="results/pref/ag_news_gemma_pref.json"
OUT_MD="results/pref/ag_news_gemma_pref.md"
MAX_TOKENS=64
TEMPERATURE=0.7
TOP_P=0.9
PRIOR_SAMPLES=50
PLOTS_DIR="results/pref/gemma_plots"

# Run preference estimation with MLX backend
"${PYTHON_BIN}" src/evals/infer_preference.py \
	--inputs "${INPUTS}" \
	--labels "${LABELS[@]}" \
	--model "${MODEL}" \
	--out "${OUT_JSON}" \
	--report_md "${OUT_MD}" \
	--max_tokens "${MAX_TOKENS}" \
	--temperature "${TEMPERATURE}" \
	--top_p "${TOP_P}" \
	--prior_samples "${PRIOR_SAMPLES}" \
	--plots_dir "${PLOTS_DIR}"