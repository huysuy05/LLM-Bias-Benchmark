# Python Script Usage Guide

This guide explains how to use the `eval_with_hgf_few_shots.py` script, which is a Python implementation of the Jupyter notebook functionality.

## Overview

The Python script provides the same functionality as the notebook but in a more production-ready format with:

- Command-line interface
- Programmatic API
- Better error handling
- Incremental result saving
- Memory management

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your HuggingFace token (optional but recommended):

```bash
export HF_TOKEN="your_huggingface_token_here"
```

## Command Line Usage

### Basic Usage

Run all datasets with default settings:

```bash
python src/eval_with_hgf_few_shots.py
```

### Advanced Usage

```bash
python src/eval_with_hgf_few_shots.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --device cuda \
    --datasets ag_news toxic_text \
    --shots 0 2 4 8 \
    --batch-size 16 \
    --hf-token "your_token_here"
```

### Command Line Arguments

- `--model`: HuggingFace model name (default: "Qwen/Qwen2.5-1.5B-Instruct")
- `--device`: Device to use - cuda, mps, or cpu (auto-detect if not specified)
- `--datasets`: Which datasets to evaluate (choices: ag_news, toxic_text, twitter_emotion)
- `--shots`: Number of few-shot examples to use (default: [0, 2, 4, 8])
- `--batch-size`: Batch size for inference (default: 16)
- `--hf-token`: HuggingFace token (or set HF_TOKEN environment variable)
- `--data-dir`: Directory containing the datasets (default: "Data")

## Programmatic Usage

### Basic Example

```python
from src.eval_with_hgf_few_shots import LLMEvaluator

# Initialize evaluator
evaluator = LLMEvaluator("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")

# Authenticate with HuggingFace
evaluator.authenticate_hf()

# Load datasets
ag_news_datasets = evaluator.load_ag_news_data("Data/ag_news")

# Run experiments
results = evaluator.run_experiments(
    ag_news_datasets,
    'ag_news',
    evaluator.label_maps['ag_news'],
    shots_list=[0, 2, 4, 8],
    batch_size=16
)
```

### Advanced Example

```python
import torch
from src.eval_with_hgf_few_shots import LLMEvaluator

# Initialize with specific device
device = "cuda" if torch.cuda.is_available() else "cpu"
evaluator = LLMEvaluator("Qwen/Qwen2.5-1.5B-Instruct", device=device)

# Set HuggingFace token
evaluator.authenticate_hf("your_token_here")

# Load all datasets
ag_news_datasets = evaluator.load_ag_news_data("Data/ag_news")
toxic_datasets = evaluator.load_toxic_text_data("Data/toxic_text")
emotion_datasets = evaluator.load_twitter_emotion_data("Data/twit")

# Run experiments on specific datasets
for dataset_name, datasets, label_map in [
    ("ag_news", ag_news_datasets, evaluator.label_maps['ag_news']),
    ("toxic_text", toxic_datasets, evaluator.label_maps['toxic_text']),
    ("twitter_emotion", emotion_datasets, evaluator.label_maps['twitter_emotion'])
]:
    print(f"Evaluating {dataset_name}...")
    results = evaluator.run_experiments(
        datasets, dataset_name, label_map,
        shots_list=[0, 2, 4], batch_size=8
    )
    print(f"Completed {dataset_name}: {results.shape[0]} experiments")
```

## Class Methods

### LLMEvaluator Class

#### Initialization

```python
evaluator = LLMEvaluator(model_name, device=None)
```

#### Data Loading Methods

- `load_ag_news_data(data_dir)`: Load AG News datasets
- `load_toxic_text_data(data_dir)`: Load toxic text datasets
- `load_twitter_emotion_data(data_dir)`: Load Twitter emotion datasets

#### Evaluation Methods

- `run_experiments(datasets_dict, dataset_name, label_map, shots_list, batch_size)`: Run full evaluation
- `classify(df, label_map, shots, batch_size, max_new_tokens, dataset_name)`: Run classification
- `eval_llm(y_true, y_pred, label_map)`: Calculate evaluation metrics

#### Utility Methods

- `authenticate_hf(token)`: Authenticate with HuggingFace Hub
- `build_prompt(df, text, label_map, shots_per_class)`: Build few-shot prompts
- `normalize_label(label, dataset_name)`: Normalize predictions using semantic similarity

## Output Files

The script generates several output files in the `results/` directory:

### Per-Dataset Results

- `results/{dataset_name}/few_shot_results_{model_name}.csv`: Aggregated results for all experiments
- `results/{dataset_name}/results__{model}__{dataset}__ratio-{ratio}__majority-{majority}__shots-{shots}.csv`: Individual experiment results

### File Structure

```
results/
├── ag_news/
│   ├── few_shot_results_Qwen_Qwen2.5-1.5B-Instruct.csv
│   └── results__Qwen_Qwen2.5-1.5B-Instruct__ag_news_balanced__ratio-balanced__majority-unknown__shots-0.csv
├── toxic_text/
│   └── ...
└── twitter_emotion/
    └── ...
```

## Performance Tips

1. **Memory Management**: Use smaller batch sizes (8-16) for large models
2. **Device Selection**: Use CUDA for NVIDIA GPUs, MPS for Apple Silicon
3. **Incremental Saving**: Results are saved after each experiment to prevent data loss
4. **Shot Counts**: Start with fewer shots (0, 2, 4) for faster initial testing

## Error Handling

The script includes comprehensive error handling:

- Automatic device detection and fallback
- Graceful handling of missing datasets
- Incremental result saving to prevent data loss
- Clear error messages and progress indicators

## Differences from Notebook

1. **No Interactive Output**: Results are saved to files instead of displayed
2. **Command Line Interface**: Can be run from terminal with arguments
3. **Better Memory Management**: More efficient memory usage
4. **Incremental Saving**: Results saved after each experiment
5. **Error Recovery**: Better error handling and recovery

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Missing Datasets**: Check data directory path and file existence
3. **HuggingFace Authentication**: Set HF_TOKEN environment variable
4. **Model Loading Issues**: Check model name and internet connection

### Debug Mode

For debugging, you can modify the script to add more verbose output:

```python
# Add this to see detailed progress
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Example Run

```bash
# Quick test with small batch
python src/eval_with_hgf_few_shots.py \
    --datasets ag_news \
    --shots 0 2 \
    --batch-size 4

# Full evaluation
python src/eval_with_hgf_few_shots.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --device cuda \
    --shots 0 2 4 8 \
    --batch-size 16
```
