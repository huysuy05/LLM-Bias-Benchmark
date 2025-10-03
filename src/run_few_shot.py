
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score, matthews_corrcoef, precision_score, average_precision_score
from sklearn.preprocessing import label_binarize

try:
    from transformers import pipeline, logging
except Exception:

    pipeline = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None


def load_ag_news(data_dir: Path, minority_count: int = 20):
    base = data_dir
    p_bal = base / 'ag_news_train_balanced.parquet'
    p_99 = base / 'ag_news_train_imbalanced_99_to_1.parquet'
    p_49 = base / 'ag_news_train_imbalanced_49_to_1_ratio.parquet'
    if not p_bal.exists():
        raise FileNotFoundError(f"Missing {p_bal}")
    bal = pd.read_parquet(p_bal)
    im99 = pd.read_parquet(p_99) if p_99.exists() else None
    im49 = pd.read_parquet(p_49) if p_49.exists() else None
    # also provide dynamic resampled variants using minority_count for parity with other loaders
    variants = {
        'ag_news_balanced': bal,
        'ag_news_imbalanced_99_to_1': im99 if im99 is not None else bal,
        'ag_news_imbalanced_49_to_1': im49 if im49 is not None else bal,
    }
    return variants


def load_toxic_text(data_dir: Path, minority_count: int = 20, label_col: str = 'label'):
    base = data_dir / 'toxic_text'
    p_train = base / 'train.csv'
    if not p_train.exists():
        raise FileNotFoundError(p_train)
    df = pd.read_csv(p_train)
    df = df[["comment_text", "toxic"]]
    df = df.rename(columns={"comment_text": "text", "toxic": "label"})
    variants = {'toxic_text_all': df}
    labels = df[label_col].unique().tolist()
    # balanced: sample min class size across labels
    min_count = min(df[df[label_col] == lab].shape[0] for lab in labels)
    bal_parts = [df[df[label_col] == lab].sample(min_count, random_state=42) for lab in labels]
    variants['toxic_text_balanced'] = pd.concat(bal_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    for lab in labels:
        # create two imbalanced variants per label
        variants[f'toxic_majority_{lab}_99_to_1'] = split_with_majority(df, lab, majority_count=minority_count * 99, minority_count=minority_count, label_col=label_col)
        variants[f'toxic_majority_{lab}_49_to_1'] = split_with_majority(df, lab, majority_count=minority_count * 49, minority_count=minority_count, label_col=label_col)
    return variants


def load_twitter_emotion(data_dir: Path, minority_count: int = 20, label_col: str = 'label'):
    base = data_dir / 'twit'
    p = base / 'twitter_emotion.parquet'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    variants = {'twitter_emotion_all': df}
    labels = df[label_col].unique().tolist()
    # balanced
    min_count = min(df[df[label_col] == lab].shape[0] for lab in labels)
    bal_parts = [df[df[label_col] == lab].sample(min_count, random_state=42) for lab in labels]
    variants['twitter_emotion_balanced'] = pd.concat(bal_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    for lab in labels:
        variants[f'emotion_majority_{lab}_99_to_1'] = split_with_majority(df, lab, majority_count=minority_count * 99, minority_count=minority_count, label_col=label_col)
        variants[f'emotion_majority_{lab}_49_to_1'] = split_with_majority(df, lab, majority_count=minority_count * 49, minority_count=minority_count, label_col=label_col)
    return variants


def split_with_majority(df: pd.DataFrame, majority_label: str, majority_count: int, minority_count: int, label_col: str = 'label') -> pd.DataFrame:
    # If majority_count is None, keep the existing counts for majority and just cap minorities
    uniq = df[label_col].unique().tolist()
    parts = []
    rng = random.Random(42)
    for lab in uniq:
        sub = df[df[label_col] == lab]
        if lab == majority_label and majority_count is not None:
            cnt = min(majority_count, len(sub))
            parts.append(sub.sample(cnt, random_state=42))
        else:
            cnt = min(minority_count, len(sub))
            parts.append(sub.sample(cnt, random_state=42))
    out = pd.concat(parts, ignore_index=True, sort=False)
    return out.sample(frac=1, random_state=42).reset_index(drop=True)


def build_prompt(df: pd.DataFrame, text: str, label_map: Dict[int, str], shots_per_class: int) -> str:
    prompt_header = (
        "You are a helpful assistant that classifies text into one of the given categories.\n"
        f"Choose exactly one label from: {', '.join(list(label_map.values()))}.\n"
        "Respond with exactly one word which is the category name; do not add any other text.\n\n"
    )
    examples = []
    for lab in list(label_map.values()):
        pool = df[df['label'] == lab]
        if len(pool) == 0:
            continue
        n = min(shots_per_class, len(pool))
        exs = pool.sample(n, random_state=42)
        for _, r in exs.iterrows():
            examples.append((r['text'], r['label']))

    for t, l in examples:
        prompt_header += f"Review: \"{t}\"\nCategory: {l}\n\n"
    prompt_header += f"Review: \"{text}\"\nCategory:"
    return prompt_header


def normalize_label(pred: str, embedding_model, valid_embeddings, valid_labs):
    if embedding_model is None or valid_embeddings is None:
        return pred
    emb = embedding_model.encode(pred, convert_to_tensor=True)
    cos = util.cos_sim(emb, valid_embeddings)[0]
    best = cos.argmax().item()
    return valid_labs[best]



def eval_llm(y_true: List[str], y_pred: List[str], label_map: Dict[int, str]):
    labels = [lab.lower() for lab in list(label_map.values())]
    y_true_arr = np.array([str(x).lower().strip() for x in y_true])
    y_pred_arr = np.array([str(x).lower().strip() for x in y_pred])

    macro_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels, average='macro', zero_division=0)
    macro_recall = recall_score(y_true_arr, y_pred_arr, labels=labels, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
    mcc = matthews_corrcoef(y_true_arr, y_pred_arr)


    p_per = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    r_per = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    f_per = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)

    precision_per_class = {labels[i]: float(p_per[i]) for i in range(len(labels))}
    recall_per_class = {labels[i]: float(r_per[i]) for i in range(len(labels))}
    f1_per_class = {labels[i]: float(f_per[i]) for i in range(len(labels))}

    # AUPRC per class
    y_true_bin = label_binarize(y_true_arr, classes=labels)
    y_pred_bin = label_binarize(y_pred_arr, classes=labels)
    if y_true_bin.shape[1] == 1:  # binary -> make 2 columns
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        y_pred_bin = np.hstack([1 - y_pred_bin, y_pred_bin])

    auprc_per_class = {}
    for i, lab in enumerate(labels):
        auprc_per_class[lab] = float(average_precision_score(y_true_bin[:, i], y_pred_bin[:, i]))

    return {
        'macro_f1': float(macro_f1),
        'macro_recall': float(macro_recall),
        'balanced_accuracy': float(bal_acc),
        'mcc': float(mcc),
        'auprc_per_class': auprc_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
    }


def run_experiments(
    model_name: str,
    dataset: str,
    label_map: Dict[int, str],
    data_dir: Path,
    out_dir: Path,
    shots_list: List[int],
    batch_size: int = 16,
    majority: Optional[str] = None,
    majority_count: Optional[int] = None,
    minority_count: int = 20,
    dry_run: bool = False,
):
    # Load dataset variants
    if dataset == 'ag_news':
        variants = load_ag_news(data_dir, minority_count=minority_count)
        label_col = 'label'
        out_dir = out_dir / dataset
    elif dataset == 'toxic_text':
        variants = load_toxic_text(data_dir, minority_count=minority_count)
        label_col = 'label'
        out_dir = out_dir / dataset
    elif dataset == 'twit':
        variants = load_twitter_emotion(data_dir, minority_count=minority_count)
        label_col = 'label'
        out_dir = out_dir / "twitter_emotion"
    else:
        raise NotImplementedError(dataset)

    # Setup pipeline if not dry-run
    pipe = None
    if not dry_run:
        if pipeline is None:
            raise RuntimeError('transformers not available in this Python environment')
        pipe = pipeline('text-generation', model=model_name, dtype='float16')
        logging.set_verbosity_error()

    # embedding model for normalization
    embedding_model = None
    valid_embeddings = None
    valid_labs = None
    if SentenceTransformer is not None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        valid_labs = list(label_map.values())
        valid_embeddings = embedding_model.encode(valid_labs, convert_to_tensor=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for var_name, df in variants.items():
        # optionally create a variant that resamples according to majority/minority
        df_variant = df
        if majority is not None:
            if majority not in df[label_col].unique():
                raise ValueError(f"Majority label {majority} not present in dataset")
            df_variant = split_with_majority(df, majority, majority_count, minority_count, label_col=label_col)

        for shots in shots_list:
            texts = df_variant['text'].tolist()
            true_labels = df_variant[label_col].tolist()

            prompts = [build_prompt(df_variant, t, label_map, shots_per_class=shots) for t in texts]

            preds = []
            if dry_run:
                choices = list(label_map.values())
                for _ in prompts:
                    preds.append(random.choice(choices))
            else:
                for i in range(0, len(prompts), batch_size):
                    batch = prompts[i:i+batch_size]
                    out = pipe(batch, max_new_tokens=3, do_sample=False)
                    for prompt, r in zip(batch, out):
                        generated = r[0]['generated_text'][len(prompt):].strip().split()
                        if generated not in list(label_map.values()):
                            generated = normalize_label(generated, embedding_model, valid_embeddings, valid_labs)
                        preds.append(generated)

            metrics = eval_llm(true_labels, preds, label_map)
            row = {
                'model': model_name,
                'dataset': dataset,
                'variant': var_name,
                'majority': majority,
                'majority_count': majority_count,
                'minority_count': minority_count,
                'shots': shots,
                **metrics,
            }
            all_results.append(row)
            # flatten nested metric dicts for CSV friendliness
            def flatten_row(r: dict) -> dict:
                flat = {}
                for k, v in r.items():
                    if isinstance(v, dict):
                        for subk, subv in v.items():
                            flat_key = f"{k}__{subk}"
                            flat[flat_key] = subv
                    else:
                        flat[k] = v
                return flat

            # save per-model aggregated and per-params CSVs using flattened rows
            agg_path = out_dir / f'few_shot_results_{model_name.replace("/","_")}.csv'
            # convert all_results to flattened rows for consistent columns
            flat_rows = [flatten_row(r) for r in all_results]
            pd.DataFrame(flat_rows).to_csv(agg_path, index=False)

            params_fname = f"results__{model_name.replace('/','_')}__{dataset}__{var_name}__ratio-majority-{majority}__shots-{shots}.csv"
            params_path = out_dir / params_fname
            pd.DataFrame([flatten_row(row)]).to_csv(params_path, index=False, mode='a', header=not params_path.exists())

    return pd.DataFrame(all_results)


def parse_label_map(arg: str):
    builtins = {
        'ag_news': {0: 'world', 1: 'sports', 2: 'business', 3: 'sci/tech'},
        'toxic_text': {0: 'toxic', 1: 'nontoxic'},
        'twitter_emotion': {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    }
    if arg in builtins:
        return builtins[arg]
    p = Path(arg)
    if p.exists():
        return json.loads(p.read_text())
    try:
        return json.loads(arg)
    except Exception:
        raise ValueError('label_map must be builtin name, JSON string, or path to JSON file')


def infer_label_map_from_data(dataset: str, data_dir: Path, label_col: str = 'label') -> Dict[int, str]:
    """Load a dataset sample and deterministically build a label_map mapping integers -> label names.

    The mapping uses sorted unique labels to ensure reproducibility.
    """
    if dataset == 'ag_news':
        variants = load_ag_news(data_dir)
        df = variants['ag_news_balanced']
    elif dataset == 'toxic_text':
        variants = load_toxic_text(data_dir)
        df = variants.get('toxic_text_all')
    elif dataset == 'twitter_emotion':
        variants = load_twitter_emotion(data_dir)
        df = variants.get('twitter_emotion_all')
    else:
        raise NotImplementedError(dataset)

    if df is None or label_col not in df.columns:
        raise ValueError(f"Could not find label column '{label_col}' for dataset {dataset}")
    uniq = sorted(df[label_col].unique().tolist(), key=lambda x: str(x))
    label_map = {i: str(lbl) for i, lbl in enumerate(uniq)}
    return label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['ag_news', 'toxic_text', 'twit'])
    parser.add_argument('--label-map', type=str, default=None, help="Builtin name (ag_news) or JSON string or path to JSON file")
    parser.add_argument('--data-dir', type=str, default='../Data/{dataset}')
    parser.add_argument('--out-dir', type=str, default='../results')
    parser.add_argument('--shots', type=int, nargs='+', default=[0,2,4,8])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--majority', type=str, default=None)
    parser.add_argument('--majority-count', type=int, default=980)
    parser.add_argument('--minority-count', type=int, default=20)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    # After parsing, we can now use args.dataset to set the default for data-dir if it wasn't provided
    if args.data_dir == '../Data/{dataset}':
        args.data_dir = f'../Data/{args.dataset}'

    if args.label_map is None:
        # infer label map automatically from the dataset files
        label_map = infer_label_map_from_data(args.dataset, Path(args.data_dir))
    else:
        label_map = parse_label_map(args.label_map)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    df = run_experiments(
        model_name=args.model,
        dataset=args.dataset,
        label_map=label_map,
        data_dir=data_dir,
        out_dir=out_dir,
        shots_list=args.shots,
        batch_size=args.batch_size,
        majority=args.majority,
        majority_count=args.majority_count,
        minority_count=args.minority_count,
        dry_run=args.dry_run,
    )
    print(df)


if __name__ == '__main__':
    main()
