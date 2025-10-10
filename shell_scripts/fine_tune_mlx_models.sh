#!/bin/bash

# --- 1. Define Variables ---
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH="Data/ag_news"
# The base directory where all adapters will be saved
ADAPTERS_BASE="fine_tuned_models/ag_news" 

# --- 2. Sanitize and Construct the Adapter Path ---

# Replace all '/' characters in MODEL_NAME with '-'
SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's/\//-/g')

# Combine the base path with the sanitized model name and a suffix
ADAPTER_PATH="$ADAPTERS_BASE/${SANITIZED_MODEL_NAME}_lora_adapters"
ITERS=1000
LEARNING_RATE=2e-4

echo "Fine-tuning model: $MODEL_NAME"
echo "Saving adapters to: $ADAPTER_PATH"

# --- 3. Run the MLX-LM LoRA Fine-Tuning Command ---
python -m mlx_lm.lora \
    --model "$MODEL_NAME" \
    --train \
    --data "$DATA_PATH" \
    --iters $ITERS \
    --batch-size 16 \
    --num-layers 16 \
    --adapter-path "$ADAPTER_PATH" \
    --save-every $ITERS \
    --learning-rate $LEARNING_RATE

# Expected adapter path: Data/ag_news/Qwen-Qwen2.5-0.5B-Instruct_lora_adapters