MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASETS="ag_news" #Could be changed to 'toxic_text', 'twitter_emotion'



python3 src/fine_tune.py \
    --model $MODEL \
    --do-train \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --output-dir fine_tuned_models/$DATASETS \
    --dataset ag_news