MODEL="google/gemma-3-1b-it"
# More models to test: 
# deepseek-ai/DeepSeek-OCR
# ibm-granite/granite-4.0-h-1b
# meta-llama/Llama-3.2-1B
# HuggingFaceTB/SmolLM3-3B
FINETUNED_MODEL="finetuned_models/ag_news/Qwen/Qwen2.5-0.5B-Instruct_finetuned"
DEVICE="mps"
DATASETS="ag_news" 
SHOT_MIN=2
SHOT_MAJ=2
TOP_P=0
TEMPERATURE=0
MAX_TOKENS=16
MAJORITY_LABEL="sports" #This label is in agnews

# Label count parameters
LABEL_COUNT_SAMPLES=20
LABEL_COUNT_TEMP=0.7

# === ICL (ZERO SHOT AND FEW-SHOT) ===
# python3 src/evals/eval_mlx_models.py --model $MODEL \
#                             --datasets $DATASETS \
#                             --shot-minority $SHOT_MIN --shot-majority $SHOT_MAJ \
#                             --temperature $TEMPERATURE \
#                             --top-p $TOP_P \
#                             --max-tokens $MAX_TOKENS \
#                             --majority-label $MAJORITY_LABEL

# === SELF-CONSISTENCY === 
# python3 src/evals/eval_mlx_models.py \
#     --model $MODEL \
#     --datasets $DATASETS \
#     --use-self-consistency \
#     --sc-samples 3 \
#     --sc-temperature 0.6 \
#     --rows-per-class 10


# === LABEL COUNT ===
# python3 src/evals/eval_mlx_models.py \
#     --model $MODEL \
#     --datasets $DATASETS \
#     --shot-minority $SHOT_MIN \
#     --shot-majority $SHOT_MAJ \
#     --max-tokens $MAX_TOKENS \
#     --majority-label $MAJORITY_LABEL \
#     --label-counts-only \
#     --label-count-samples $LABEL_COUNT_SAMPLES \
#     --label-count-temperature $LABEL_COUNT_TEMP \
#     --rows-per-class 100


# === TBVM (THRESHOLD-BASED VOTING MITITGATION) ===
python3 src/evals/eval_mlx_models.py \
  --datasets ag_news \
  --minority-first \
  --mf-threshold 5 \
  --mf-samples 8 \
  --model google/gemma-3-1b-it \
  --shot-minority 3 \
  --shot-majority 3 \
  --temperature 1.0