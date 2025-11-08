#!/bin/bash

set -euo pipefail

PYTHON="/projects/beqt/jesuszhou/envs/bllm/bin/python"

MODELS=(
	"meta-llama/Llama-3.1-8B-Instruct"
	"Qwen/Qwen3-8B"
	"google/gemma-3-1b-it"
	"google/gemma-7b"
	"unsloth/Apriel-1.5-15b-Thinker-GGUF"
)

DEVICE="cuda"
DATASETS="all"
BATCH_SIZE=64
SHOT_MIN=8
SHOT_MAJ=8
MAJORITY_LABEL="none"
MAX_TOKENS=3
SEED=42

USE_SC=false
SC_SAMPLES=5
SC_TEMP=0.7

MINORITY_FIRST=true
MF_SAMPLES=25
MF_THRESHOLD=10
MF_TOP_P=0.9

if [ "${USE_SC}" = true ] && [ "${MINORITY_FIRST}" = true ]; then
	echo "Error: USE_SC and MINORITY_FIRST cannot both be true. Please choose one mode." >&2
	exit 1
fi

for MODEL in "${MODELS[@]}"; do
	echo "--------------------------------------"
	echo "Running model: ${MODEL}"
	echo "--------------------------------------"

	CMD=("${PYTHON}" src/evals/eval_llm.py
		--model "${MODEL}"
		--device "${DEVICE}"
		--datasets "${DATASETS}"
		--different-shots
		--batch-size "${BATCH_SIZE}"
		--max-tokens "${MAX_TOKENS}"
		--seed "${SEED}"
	)

	if [ "${MAJORITY_LABEL}" != "none" ]; then
		CMD+=(--majority-label "${MAJORITY_LABEL}")
	fi

	if [ "${USE_SC}" = true ]; then
		echo "Running with SELF-CONSISTENCY (samples=${SC_SAMPLES}, temp=${SC_TEMP})"
		CMD+=(
			--use-self-consistency
			--sc-samples "${SC_SAMPLES}"
			--sc-temperature "${SC_TEMP}"
			--shots-minority "${SHOT_MIN}"
			--shots-majority "${SHOT_MAJ}"
		)
	elif [ "${MINORITY_FIRST}" = true ]; then
		echo "Running with MINORITY-FIRST voting (samples=${MF_SAMPLES}, threshold=${MF_THRESHOLD})"
		CMD+=(
			--shots-minority "${SHOT_MIN}"
			--shots-majority "${SHOT_MAJ}"
			--minority-first
			--mf-samples "${MF_SAMPLES}"
			--mf-threshold "${MF_THRESHOLD}"
			--mf-top-p "${MF_TOP_P}"
		)
	else
		echo "Running with GREEDY decoding"
		CMD+=(
			--shots-minority "${SHOT_MIN}"
			--shots-majority "${SHOT_MAJ}"
		)
	fi

	"${CMD[@]}"
	echo "Finished model: ${MODEL}"; echo
done

echo "All models finished!"

