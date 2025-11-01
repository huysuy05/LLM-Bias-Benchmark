# LLM Evaluation Scripts Usage Guide

This guide explains how to use the evaluation scripts for testing LLMs on imbalanced datasets.

## Overview

This project provides evaluation scripts for testing language models on imbalanced classification datasets using:

- Few-shot in-context learning (ICL)
- Self-consistency prompting
- Multiple model backends (HuggingFace, MLX, OpenAI)
- Comprehensive metrics (F1, MCC, balanced accuracy, per-class AUPRC)
- Support for AG News, toxic text, and Twitter emotion datasets
- MLX-based preference estimation for label skew analysis

## Project Structure

```
src/
├── evals/
│   ├── eval_llm.py          # HuggingFace models evaluation
│   ├── eval_mlx_models.py   # MLX models (Apple Silicon optimized)
│   ├── eval_openai.py       # OpenAI API evaluation
│   └── infer_preference.py  # MLX-based label preference estimation
├── packages/
│   ├── dataset_loader.py    # Dataset loading utilities
│   └── self_consistency.py  # Self-consistency implementation
├── analysis/
│   └── analyze_results.py   # Results analysis tools
├── fine_tune.py             # Model fine-tuning script
└── pref_inf.py              # (legacy) preference inference helpers
```

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your HuggingFace token (optional but recommended):

```bash
export HF_TOKEN="your_huggingface_token_here"
```

---

## Evaluation Scripts

### 1. MLX Models (Apple Silicon - `eval_mlx_models.py`)

Optimized for Apple Silicon (M1/M2/M3) with 4-bit quantization support.

#### Basic Usage

```bash
python src/evals/eval_mlx_models.py \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --datasets ag_news \
    --shot-minority 4 \
    --shot-majority 4 \
    --temperature 0 \
    --max-tokens 16
```

#### With Self-Consistency

```bash
python src/evals/eval_mlx_models.py \
    --model "google/gemma-3-1b-it" \
    --datasets ag_news \
    --use-self-consistency \
    --sc-samples 5 \
    --sc-temperature 0.7 \
    --max-tokens 16
```

#### Arguments

- `--model`: HuggingFace model name
- `--datasets`: Dataset to evaluate (ag_news, toxic_text, twitter_emotion)
- `--shot-minority`: Number of minority class examples
- `--shot-majority`: Number of majority class examples
- `--temperature`: Sampling temperature (0 for greedy)
- `--top-p`: Nucleus sampling parameter
- `--max-tokens`: Maximum tokens to generate
- `--majority-label`: Force a specific label as majority (e.g., "sports")
- `--use-self-consistency`: Enable self-consistency prompting
- `--sc-samples`: Number of samples for self-consistency
- `--sc-temperature`: Temperature for self-consistency sampling
- `--quantize`: Use 4-bit quantization

#### Shell Script

```bash
bash shell_scripts/run_evals_with_mlx.sh
```

---

### 2. HuggingFace Models (`eval_llm.py`)

For CUDA-enabled GPUs with larger models.

#### Basic Usage

```bash
python src/evals/eval_llm.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --device cuda \
    --datasets ag_news \
    --batch-size 64 \
    --shots-minority 8 \
    --shots-majority 8 \
    --max-tokens 3
```

#### With Self-Consistency

```bash
python src/evals/eval_llm.py \
    --model "Qwen/Qwen3-8B" \
    --device cuda \
    --datasets all \
    --use-self-consistency \
    --sc-samples 5 \
    --sc-temperature 0.7
```

#### Arguments

- `--model`: HuggingFace model name
- `--device`: Device (cuda/cpu)
- `--datasets`: Datasets to evaluate (ag_news, toxic_text, twitter_emotion, or all)
- `--batch-size`: Batch size for inference
- `--different-shots`: Use different shot counts for minority/majority
- `--shots-minority`: Minority class shot count
- `--shots-majority`: Majority class shot count
- `--majority-label`: Force specific majority label
- `--max-tokens`: Maximum tokens to generate
- `--use-self-consistency`: Enable self-consistency
- `--sc-samples`: Number of self-consistency samples
- `--sc-temperature`: Self-consistency temperature

#### Shell Script

```bash
bash shell_scripts/run_eval.sh
```

---

### 3. OpenAI Models (`eval_openai.py`)

For evaluating OpenAI models (GPT-3.5, GPT-4, etc.) with API caching.

#### Basic Usage

```bash
python src/evals/eval_openai.py \
    --model "gpt-4o-mini" \
    --datasets ag_news \
    --shots-minority 4 \
    --shots-majority 4
```

#### With Self-Consistency

```bash
python src/evals/eval_openai.py \
    --model "gpt-4o-mini" \
    --datasets ag_news \
    --use-self-consistency \
    --sc-samples 5 \
    --sc-temperature 0.7
```

#### Features

- Automatic caching to reduce API costs
- Hash-based cache management
- Support for all OpenAI chat models

---

### 4. Preference Estimation (`infer_preference.py`)

Estimate label preference distributions using local MLX models. This pipeline requires [`mlx`](https://github.com/ml-explore/mlx) and [`mlx-lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms).

#### Basic Usage

```bash
python src/evals/infer_preference.py \
    --inputs Data/ag_news/valid.jsonl \
    --labels world sports business sci/tech \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --out results/pref/ag_news_pref.json \
    --report_md results/pref/ag_news_pref.md \
    --max_tokens 64 \
    --temperature 0.7 \
    --top_p 0.9 \
    --prior_samples 50
```

#### Key Arguments

- `--inputs`: Path to JSON, JSONL, or CSV with `id`, `text`, `label` fields (Parquet not supported).
- `--labels`: Explicit label list. If omitted, labels are inferred from the data file.
- `--model`: MLX model name or local path (defaults to `mlx-community/Qwen2.5-0.5B-Instruct-4bit`).
- `--max_tokens`, `--temperature`, `--top_p`: Generation controls passed to `mlx_lm.generate`.
- `--vote_samples`: Number of samples per dataset prompt (default 5).
- `--prior_samples`: Samples per content-free prior prompt (default 20).
- `--prior_prompts`: Custom content-free prompts; falls back to `DEFAULT_PRIOR_PROMPTS` in code.
- `--mlx_prompt_template`: Optional prompt template (use `{text}`/`{label_list}` placeholders).
- `--seed`: Seeds Python, NumPy, and MLX RNGs for reproducibility.

#### Outputs

- JSON results with prior (`P0`), conditional (`Vbar`), over-prediction, and skew metrics.
- Optional Markdown report and PNG plots (when `--report_md` and `--plots_dir` are passed).

> **Tip:** Ensure CLI options use ASCII hyphen-minus (`--`) rather than typographic dashes (`–`) so `argparse` recognizes the flags (e.g., `--prior_samples`, not `–-prior_samples`).

---

## Fine-Tuning

### Fine-Tune with MLX (Apple Silicon)

```bash
bash shell_scripts/fine_tune_mlx_models.sh
```

This script:
1. Fine-tunes a model with LoRA adapters
2. Automatically fuses the adapters after training
3. Saves the fine-tuned model for evaluation

#### Manual Fine-Tuning

```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --train \
    --data "Data/ag_news" \
    --iters 500 \
    --batch-size 16 \
    --adapter-path "adapters/ag_news/Qwen-Qwen2.5-0.5B-Instruct_lora_adapters"
```

#### Fuse Adapters

```bash
bash shell_scripts/fuse_mlx_models.sh
```

### Fine-Tune with HuggingFace

```bash
python src/fine_tune.py \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --do-train \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --output-dir fine_tuned_models/ag_news \
    --dataset ag_news
```

Or use the shell script:

```bash
bash shell_scripts/fine_tune.sh
```

---

## Dataset Structure

The scripts expect datasets in the following structure:

```
Data/
├── ag_news/
│   ├── train.jsonl
│   └── valid.jsonl
├── toxic_text/
│   ├── train.csv
│   └── test.csv
└── twitter_emotion/
    └── ... (dataset files)
```

### Supported Datasets

1. **AG News**: 4-class news classification (world, sports, business, sci/tech)
2. **Toxic Text**: Binary toxicity detection
3. **Twitter Emotion**: 6-class emotion classification

---

## Output Files

### MLX Models

Results saved to `mlx_models_results/{dataset_name}/`:

- `SC_results_{model}_{date}.csv` - Self-consistency results
- `ICL_results_{model}_{date}.csv` - In-context learning results

### HuggingFace Models

Results saved to `results/{dataset_name}/`:

- `SC_results_{model}_{date}.csv` - Self-consistency results
- `few_shot_results_{model}_{date}.csv` - Few-shot ICL results

### OpenAI Models

Results saved to `results/openai/`:

- Cached API responses in `results/openai_cache/`

### Preference Estimation

Results saved to `results/pref/` (JSON and optional Markdown reports) when running `infer_preference.py`.

---

## Programmatic Usage

### Example: MLX Evaluation

```python
import sys
sys.path.insert(0, 'src')

from packages.dataset_loader import DatasetLoader
from packages.self_consistency import SelfConsistency

# Load datasets
label_map = {0: "world", 1: "sports", 2: "business", 3: "sci/tech"}
dl = DatasetLoader({'ag_news': label_map})
datasets = dl.load_ag_news_data("Data/ag_news")

# Run evaluation
from evals.eval_mlx_models import classify

predictions = classify(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    df=datasets['ag_news_balanced'],
    label_map=label_map,
    shots_minority=4,
    shots_majority=4,
    max_new_tokens=16,
    temp=0,
    use_self_consistency=True,
    sc_num_samples=5
)
```

---

## Metrics

All evaluation scripts compute:

- **Macro F1**: F1 score averaged across classes
- **Macro Recall**: Recall averaged across classes  
- **Balanced Accuracy**: Accuracy adjusted for class imbalance
- **MCC**: Matthews Correlation Coefficient
- **Per-class Metrics**: Precision, recall, F1, AUPRC for each class

---

---

## Performance Tips

1. **Memory Management**: 
   - MLX: Use 4-bit quantization with `--quantize` flag
   - HuggingFace: Use smaller batch sizes for large models
   
2. **Device Selection**: 
   - Use MLX for Apple Silicon (M1/M2/M3)
   - Use CUDA for NVIDIA GPUs
   - CPU fallback available for all scripts

3. **Self-Consistency**: 
   - Improves accuracy but increases compute time
   - Start with 3-5 samples for testing
   - Use temperature 0.6-0.8 for diversity

4. **Shot Counts**: 
   - Start with 4-8 shots for initial testing
   - More shots = better performance but slower inference

---

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **CUDA Out of Memory**: Reduce batch size or use quantization
3. **Missing Datasets**: Check `Data/` directory structure and file paths
4. **HuggingFace Authentication**: Set `HF_TOKEN` environment variable
5. **MLX Installation**: Only works on Apple Silicon (macOS with M1/M2/M3)

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Key Features

✅ **Multiple Model Backends**: HuggingFace, MLX (Apple Silicon), OpenAI  
✅ **Self-Consistency**: Improves reliability through sampling & voting  
✅ **Quantization Support**: 4-bit quantization for MLX models  
✅ **Comprehensive Metrics**: F1, MCC, balanced accuracy, per-class AUPRC  
✅ **Imbalanced Datasets**: Specialized support for class imbalance  
✅ **Fine-Tuning**: LoRA-based fine-tuning with MLX and HuggingFace  
✅ **Caching**: OpenAI API response caching to reduce costs  
✅ **Shell Scripts**: Ready-to-use scripts for common workflows  

---

