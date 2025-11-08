# LLM Evaluation on Imbalanced Datasets

Evaluate language models on imbalanced text classification tasks using few-shot prompting, self-consistency, and minority-first voting strategies.

## Features

- **Multiple backends**: HuggingFace (CUDA/CPU), MLX (Apple Silicon), OpenAI API
- **Evaluation modes**: Greedy ICL, self-consistency prompting, threshold-based minority-first voting
- **Datasets**: AG News (4-class), Toxic Text (binary), Twitter Emotion (6-class)
- **Metrics**: Macro F1, balanced accuracy, MCC, per-class precision/recall/F1/AUPRC
- **Preference estimation**: Analyze label skew and model bias with MLX-based sampling

## Quick Start

```bash
pip install -r requirements.txt
export HF_TOKEN="your_token"  # optional

# Evaluate with minority-first voting (mitigates label bias)
python src/evals/eval_llm.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --datasets all \
    --minority-first \
    --mf-samples 5 \
    --mf-threshold 3

# Or use shell scripts
bash shell_scripts/run_eval.sh              # HuggingFace models
bash shell_scripts/run_evals_with_mlx.sh    # MLX (Apple Silicon)
```

## Evaluation Modes

### Greedy In-Context Learning
Standard few-shot classification with deterministic decoding.

### Self-Consistency
Sample multiple outputs per example, aggregate via majority vote.

```bash
--use-self-consistency --sc-samples 5 --sc-temperature 0.7
```

### Minority-First Voting (New)
Threshold-based mitigation: when a preferred label dominates vote counts, defer to the next-most-common label.

```bash
--minority-first --mf-samples 5 --mf-threshold 3 --seed 42
```

## Key Scripts

- `src/evals/eval_llm.py` – HuggingFace model evaluation (CUDA/CPU)
- `src/evals/eval_mlx_models.py` – MLX model evaluation (Apple Silicon, quantized)
- `src/evals/eval_openai.py` – OpenAI API evaluation (cached)
- `src/evals/infer_preference.py` – Label preference/skew estimation (MLX)
- `shell_scripts/run_eval.sh` – Batch evaluator for HuggingFace models
- `shell_scripts/create_fixed_test_set.sh` – Generate reproducible test splits


## Common Arguments

- `--model` – Model identifier (HuggingFace path or MLX community model)
- `--datasets` – One or more of: `ag_news`, `toxic_text`, `twitter_emotion`, `all`
- `--shots-minority` / `--shots-majority` – Few-shot example counts per class
- `--temperature`, `--top-p`, `--max-tokens` – Generation hyperparameters
- `--seed` – Random seed for reproducibility

## Outputs

Results are saved to:
- `results/{dataset}/` – HuggingFace/OpenAI evaluations
- `mlx_models_results/{dataset}/` – MLX evaluations  
- File prefixes: `ICL_`, `SC_`, `MF_` (in-context learning, self-consistency, minority-first)

Each CSV includes per-dataset metrics, shot configurations, and timestamp.

## Fine-Tuning (Optional)

Train LoRA adapters on imbalanced splits:

```bash
# MLX (Apple Silicon)
bash shell_scripts/fine_tune_mlx_models.sh

# HuggingFace
bash shell_scripts/fine_tune.sh
```

Fuse adapters after training:

```bash
bash shell_scripts/fuse_mlx_models.sh
```

## Datasets

Fixed test splits (100 examples/class, seed=42) are stored in `Data/<dataset>/test_fixed_100per_class.jsonl`. Regenerate with:

```bash
bash shell_scripts/create_fixed_test_set.sh
```

---

**Questions?** Check `--help` on any script for full argument lists.



