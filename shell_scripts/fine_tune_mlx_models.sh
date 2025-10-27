#!/bin/bash

# --- 1. Define Variables ---
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME="ag_news"
DATA_PATH="Data/$DATASET_NAME"
# The base directory where all adapters will be saved
ADAPTERS_BASE="adapters/$DATASET_NAME/" 

# --- 2. Sanitize and Construct the Adapter Path ---

# Replace all '/' characters in MODEL_NAME with '-'
SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's/\//-/g')

# Combine the base path with the sanitized model name and a suffix
ADAPTER_PATH="$ADAPTERS_BASE/${SANITIZED_MODEL_NAME}_lora_adapters"
ITERS=500
LEARNING_RATE=2e-4
BATCH_SZ=16
N_LAYERS=4
OPTIMIZER="adamw"


echo "--------------------------------------------------------"
echo "Model to Fuse: ${MODEL_NAME}"
echo "Dataset Name: ${DATASET_NAME}"
echo "Adapter Path: ${ADAPTER_PATH}"
echo "--------------------------------------------------------"

# Ask for confirmation
read -r -p "Is this the correct dataset and path for fusion? (y/n): " confirmation_input

# Check the user's input
if [[ "$confirmation_input" == "y" || "$confirmation_input" == "Y" ]]; then
    echo "Confirmation received. Proceeding with model fusion..."
else
    echo "Fusion cancelled by user. Exiting script."
    exit 1 # Exit with a non-zero status to indicate an abnormal exit
fi


# --- 3. Run the MLX-LM LoRA Fine-Tuning Command ---
mlx_lm.lora \
    --model "$MODEL_NAME" \
    --train \
    --data "$DATA_PATH" \
    --iters $ITERS \
    --batch-size $BATCH_SZ \
    --num-layers $N_LAYERS \
    --adapter-path "$ADAPTER_PATH" \
    --save-every $ITERS \
    --learning-rate $LEARNING_RATE \
    --report-to wandb \
    --project-name finetuning-mlx-models \
    --grad-checkpoint \
    --optimizer $OPTIMIZER \

# Expected adapter path: Data/ag_news/Qwen-Qwen2.5-0.5B-Instruct_lora_adapters

# Fuse model after fine tuning
#!/bin/bash

FINETUNED_MODEL_DIR="finetuned_models/${DATASET_NAME}/${MODEL_NAME}_finetuned"


echo "--------------------------------------------------------"
echo "Model to Fuse: ${MODEL_NAME}"
echo "Dataset Name: ${DATASET_NAME}"
echo "Adapter Path: ${ADAPTER_PATH}"
echo "Fine-tuned Model Directory: ${FINETUNED_MODEL_DIR}"
echo "--------------------------------------------------------"

# Ask for confirmation
read -r -p "Is this the correct dataset and path for fusion? (y/n): " confirmation_input

# Check the user's input
if [[ "$confirmation_input" == "y" || "$confirmation_input" == "Y" ]]; then
    echo "Confirmation received. Proceeding with model fusion..."
else
    echo "Fusion cancelled by user. Exiting script."
    exit 1 # Exit with a non-zero status to indicate an abnormal exit
fi

# =======================================================
# --- FUSION COMMAND (Only runs if user confirms 'y') ---
# =======================================================

# FUSE MODEL

mlx_lm.fuse \
    --model $MODEL_NAME \
    --adapter-path $ADAPTER_PATH \
    --save-path $FINETUNED_MODEL_DIR
