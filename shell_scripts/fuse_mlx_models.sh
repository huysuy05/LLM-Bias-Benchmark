#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME="ag_news"
FINETUNED_MODEL_DIR="finetuned_models/${DATASET_NAME}/${MODEL_NAME}_finetuned"

# 1. Sanitize the Model Name for the directory
# Result: Qwen-Qwen2.5-0.5B-Instruct
SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's/\//-/g')

# 2. Define the Base Directory using the DATASET_NAME variable
ADAPTERS_BASE="adapters/$DATASET_NAME/" 

# 3. Construct the FINAL path for the adapter weights
# This is the path where your LORA adapters are located
# This should match your example path: /.../fine_tuned_models/ag_news/Qwen-Qwen2.5-0.5B-Instruct_lora_adapters
ADAPTER_PATH="${ADAPTERS_BASE}${SANITIZED_MODEL_NAME}_lora_adapters"

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

# PUSH TO HUGGING FACE
python3 src/push_models.py