MODEL="Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_MODEL="finetuned_models/ag_news/Qwen/Qwen2.5-0.5B-Instruct_finetuned"
DEVICE="mps"
DATASETS="toxic_text" 
SHOT_MIN=8
SHOT_MAJ=8
TOP_P=0
TEMPERATURE=0
MAX_TOKENS=50
MAJORITY_LABEL="toxic" #This label is in agnews

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
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --datasets ag_news \
    --use-self-consistency \
    --sc-samples 5 \
    --sc-temperature 0.7