MODEL="deepseek-ai/DeepSeek-OCR"
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

# Run Python script - Standard evaluation
# python3 src/evals/eval_mlx_models.py --model $MODEL \
#                             --datasets $DATASETS \
#                             --shot-minority $SHOT_MIN --shot-majority $SHOT_MAJ \
#                             --temperature $TEMPERATURE \
#                             --top-p $TOP_P \
#                             --max-tokens $MAX_TOKENS \
#                             --majority-label $MAJORITY_LABEL

# (Optional) Script for running with self_consistency
# python3 src/evals/eval_mlx_models.py \
#     --model $MODEL \
#     --datasets $DATASETS \
#     --use-self-consistency \
#     --sc-samples 3 \
#     --sc-temperature 0.6

# Script for collecting label counts only (no metrics)
python3 src/evals/eval_mlx_models.py \
    --model $MODEL \
    --datasets $DATASETS \
    --shot-minority $SHOT_MIN \
    --shot-majority $SHOT_MAJ \
    --max-tokens $MAX_TOKENS \
    --majority-label $MAJORITY_LABEL \
    --label-counts-only \
    --label-count-samples $LABEL_COUNT_SAMPLES \
    --label-count-temperature $LABEL_COUNT_TEMP \
    --rows-per-class 100