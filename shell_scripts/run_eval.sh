# Variables
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DEVICE="mps"
DATASETS="ag_news" #Could be changed to 'toxic_text', 'twitter_emotion'
BS=16
SHOT_MIN=8
SHOT_MAJ=8

# Run Python script
python3 src/eval_llm.py --model $MODEL \
                            --device $DEVICE \
                            --datasets $DATASETS \
                            --different-shots \
                            --batch-size $BS --shots-minority $SHOT_MIN --shots-majority $SHOT_MAJ