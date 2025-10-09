MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DEVICE="mps"
DATASETS="ag_news" #Could be changed to 'toxic_text', 'twitter_emotion'
SHOT_MIN=8
SHOT_MAJ=8
TOP_P=0
TEMPERATURE=0
MAX_TOKENS=10
MAJORITY_LABEL="sports" #This label is in agnews

# Run Python script
python3 src/eval_mlx_models.py --model $MODEL \
                            --datasets $DATASETS \
                            --shot-minority $SHOT_MIN --shot-majority $SHOT_MAJ \
                            --temperature $TEMPERATURE \
                            --top-p $TOP_P \
                            --max-tokens $MAX_TOKENS \
                            --majority-label $MAJORITY_LABEL