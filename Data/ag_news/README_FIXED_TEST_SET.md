# Fixed Test Set for Consistent Evaluation

## Overview

This directory contains fixed test sets for consistent evaluation across runs. Using fixed test sets ensures:

- **Reproducibility**: Same examples every time
- **Speed**: Smaller, manageable dataset size
- **Consistency**: No randomness across evaluations
- **Fairness**: No lucky/unlucky samples affecting results

## Current Fixed Test Sets

### AG News
- **File**: `test_fixed_200per_class.jsonl`
- **Size**: 800 examples (200 per class)
- **Classes**: world, sports, business, sci/tech
- **Seed**: 42
- **Created**: 2025-11-04

## Usage

### Automatic (Recommended)

The DatasetLoader will automatically use the fixed test set if it exists:

```python
from packages.dataset_loader import DatasetLoader

label_maps = {'ag_news': {0: 'world', 1: 'sports', 2: 'business', 3: 'sci/tech'}}
dl = DatasetLoader(label_maps)
variants = dl.load_ag_news_data()  # Automatically uses fixed test set
df = variants['ag_news_balanced']
```

### Manual Control

To disable fixed test set and use the full dataset:

```python
variants = dl.load_ag_news_data(use_fixed_test=False)
```

To use a different size:

```python
variants = dl.load_ag_news_data(use_fixed_test=True, fixed_test_rows=100)
```

## Creating New Fixed Test Sets

### For AG News with 200 rows per class:

```bash
bash shell_scripts/create_fixed_test_set.sh
```

### For custom sizes:

```bash
python3 src/create_fixed_test_set.py --dataset ag_news --rows-per-class 100 --seed 42
```

### For other datasets:

```bash
python3 src/create_fixed_test_set.py --dataset toxic_text --rows-per-class 200
python3 src/create_fixed_test_set.py --dataset twitter_emotion --rows-per-class 150
```

## Files

- `test_fixed_200per_class.jsonl` - The actual test data
- `test_fixed_200per_class.json` - Metadata (creation date, seed, label counts)

## Benefits

### Before (Full Dataset):
```
4000 examples × 25 samples = 100,000 inferences
≈ 14-15 hours runtime 😱
Random shuffling each time
```

### After (Fixed Test Set):
```
800 examples × 25 samples = 20,000 inferences
≈ 3-4 hours runtime ✅
Consistent results every time
```

## Time Estimates

| Rows/Class | Total Examples | Time (25 samples) | Use Case |
|------------|----------------|-------------------|----------|
| 10 | 40 | ~10 min | Quick testing |
| 50 | 200 | ~40-50 min | Development |
| 100 | 400 | ~1.5-2 hours | Validation |
| 200 | 800 | ~3-4 hours | Production |
| Full (1000) | 4000 | ~14-15 hours | Final evaluation |

## Verification

To verify your fixed test set is loaded:

```python
import sys
sys.path.insert(0, 'src')
from packages.dataset_loader import DatasetLoader

label_maps = {'ag_news': {0: 'world', 1: 'sports', 2: 'business', 3: 'sci/tech'}}
dl = DatasetLoader(label_maps)
variants = dl.load_ag_news_data()
df = variants['ag_news_balanced']

print(f'Total rows: {len(df)}')
print(f'Label distribution:\n{df["label"].value_counts().sort_index()}')
```

You should see:
```
[INFO] Loading fixed test set: Data/ag_news/test_fixed_200per_class.jsonl
[INFO] Fixed test set created: 2025-11-04T...
[INFO] Total examples: 800, Rows per class: 200
```
