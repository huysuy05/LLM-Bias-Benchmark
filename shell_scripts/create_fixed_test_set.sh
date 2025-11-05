#!/bin/bash

# Script to create fixed test sets for consistent evaluation

echo "Creating fixed test set for AG News..."
python3 src/create_fixed_test_set.py \
    --dataset ag_news \
    --rows-per-class 200 \
    --seed 42

echo ""
echo "✅ Done! Fixed test set created."
echo ""
echo "To use the fixed test set in your evaluations:"
echo "  - The dataset loader will automatically use it by default"
echo "  - It contains 200 rows per class (800 total)"
echo "  - Consistent across all runs (seed=42)"
echo ""
echo "To create different sizes, run:"
echo "  python3 src/create_fixed_test_set.py --rows-per-class 100"
