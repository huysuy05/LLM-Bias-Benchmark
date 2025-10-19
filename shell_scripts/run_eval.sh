MODELS=(
"meta-llama/Llama-3.1-8B-Instruct"
"Qwen/Qwen3-8B"
"google/gemma-3-1b-it"
"google/gemma-7b"
"unsloth/Apriel-1.5-15b-Thinker-GGUF"
)
DEVICE="cuda"
DATASETS="all"
BS=64
SHOT_MIN=8
SHOT_MAJ=8
MAJORITY_LABEL=none
MAX_TOKENS=3
USE_SC=true
SC_SAMPLES=5
SC_TEMP=0.7

====== Step 2: Loop over models ======
for MODEL in "${MODELS[@]}"; do
echo "--------------------------------------"
echo "Running model: $MODEL"
echo "--------------------------------------"
if [ "$USE_SC" = true ]; then
echo "Running with SELF-CONSISTENCY (samples=$SC_SAMPLES, temp=$SC_TEMP)"
/projects/beqt/jesuszhou/envs/bllm/bin/python eval_llm.py --model "$MODEL"
--device "$DEVICE"
--datasets "$DATASETS"
--different-shots
--batch-size "$BS"
--majority-label "$MAJORITY_LABEL"
--max-tokens "$MAX_TOKENS"
--use-self-consistency
--sc-samples "$SC_SAMPLES"
--sc-temperature "$SC_TEMP"
else
echo "Running with GREEDY DECODING"
/projects/beqt/jesuszhou/envs/bllm/bin/python eval_llm.py --model "$MODEL"
--device "$DEVICE"
--datasets "$DATASETS"
--different-shots
--batch-size "$BS"
--shots-minority "$SHOT_MIN"
--shots-majority "$SHOT_MAJ"
--majority-label "$MAJORITY_LABEL"
--max-tokens "$MAX_TOKENS"
fi
echo "Finished model: $MODEL"
echo ""
done
echo "All models finished!"

