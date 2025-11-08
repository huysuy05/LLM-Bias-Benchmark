#!/bin/bash

# Script to create fixed test sets for consistent evaluation

set -euo pipefail

DATASETS=(ag_news toxic_text twitter_emotion)
ROWS_PER_CLASS=100
SEED=42

for dataset in "${DATASETS[@]}"; do
    echo "Creating fixed ${ROWS_PER_CLASS}-per-class test set for ${dataset}..."
    python3 src/create_fixed_test_set.py \
        --dataset "${dataset}" \
        --rows-per-class "${ROWS_PER_CLASS}" \
        --seed "${SEED}"
    echo ""
done

echo "✅ Fixed test sets created for: ${DATASETS[*]}"
echo "  - Rows per class: ${ROWS_PER_CLASS}"
echo "  - Seed: ${SEED}"
echo "  - Files stored under Data/<dataset>/test_fixed_${ROWS_PER_CLASS}per_class.{json,jsonl}"

echo "Run this script again if you need to refresh the fixed splits." 
