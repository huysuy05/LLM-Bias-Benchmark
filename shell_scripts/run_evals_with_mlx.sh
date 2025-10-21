MODEL="google/gemma-3-1b-it"
# More models to test: 
# Qwen/Qwen2.5-0.5B-Instruct
# google/gemma-3-270m
# Qwen/Qwen3-0.6B
# distilbert/distilgpt2
FINETUNED_MODEL="finetuned_models/ag_news/Qwen/Qwen2.5-0.5B-Instruct_finetuned"
DEVICE="mps"
DATASETS="ag_news" 
SHOT_MIN=4
SHOT_MAJ=4
TOP_P=0
TEMPERATURE=0
MAX_TOKENS=256
MAJORITY_LABEL="sports" #This label is in agnews

# Run Python script
# python3 src/eval_mlx_models.py --model $MODEL \
#                             --datasets $DATASETS \
#                             --shot-minority $SHOT_MIN --shot-majority $SHOT_MAJ \
#                             --temperature $TEMPERATURE \
#                             --top-p $TOP_P \
#                             --max-tokens $MAX_TOKENS \
#                             --majority-label $MAJORITY_LABEL

# (Optional) Script for running with self_consistency
python3 src/eval_mlx_models.py \
    --model $MODEL \
    --datasets $DATASETS \
    --use-self-consistency \
    --sc-samples 3 \
    --sc-temperature 0.6