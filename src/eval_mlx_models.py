"""
Python script to run MLX Models, which are designed to be running on MacOS machines
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import os
import numpy as np
import argparse
import re
import torch
from dataset_loader import DatasetLoader
from self_consistency import SelfConsistency
import random
import subprocess
from sklearn.metrics import (
    f1_score, recall_score, balanced_accuracy_score,
    matthews_corrcoef, precision_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from time import time
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from tqdm.auto import tqdm


print(f"MPS AVAILABLE: {torch.mps.is_available()}")

# Thought process:
# Build the prompt --> Load dataset (Through Python package already written) --> Load Model --> Quantize the model into 4-bit --> Run evals on the models

def build_prompt(df, text, label_map, shots_minority=0, shots_majority=0, forced_maj_label=None):
    """
    Build prompt using `shots_majority` for the inferred majority label (from df)
    and `shots_minority` for the other labels.
    """
    assert shots_minority is not None and shots_majority is not None, (
        "Please provide 'shots_minority' and 'shots_majority' parameters"
    )

    labels = list(label_map.values())
    prompt = (
        f"You are a powerful, precise, and helpful assistant that classifies text into well-defined categories, NO MATTER THE CONTEXT."
        f" IMPORTANT: CHOOSE ONE WORD FROM THESE CATEGORIES: {', '.join(labels)}."
        f" Respond with exactly one word: the single best category inside the given categories, DO NOT ANSWER ANY OTHER CATEGORIES BESIDES THE GIVEN ONE."
        f" Do not explain your choice, provide reasoning, or output anything else."
    )

    # Use forced majority label if provided; otherwise infer from df (if possible)
    maj_label = None
    if forced_maj_label is not None:
        maj_label = forced_maj_label
    else:
        try:
            counts = df['label'].value_counts()
            if len(counts) > 0:
                maj_label = counts.idxmax()
        except Exception:
            maj_label = None

    # Collect few-shot examples per label according to inferred majority/minority shots
    few_shots_example = []
    for lab in labels:
        # If we can't infer majority, treat all labels equally (use shots_minority for all)
        if maj_label is not None and lab == maj_label:
            n = int(shots_majority)
        else:
            n = int(shots_minority)

        if n <= 0:
            continue

        avail = df[df['label'] == lab]
        k = min(n, len(avail))
        if k <= 0:
            continue

        samples = avail.sample(k, random_state=42)
        for _, r in samples.iterrows():
            few_shots_example.append({'text': r['text'], 'label': r['label']})

    if few_shots_example:
        random.shuffle(few_shots_example)
        prompt += "\n\nLearn from these examples to understand context and edge cases:\n\n"
        for ex in few_shots_example:
            prompt += f"Review: \"{ex['text']}\"\nCategory: {ex['label']}\n\n"

    prompt += f"Review: \"{text}\"\So what is this the label for this text? Answer here: "
    return prompt

emb_model = SentenceTransformer("all-MiniLM-L6-v2")
def normalize_label(label, label_map):
    """Normalize a predicted label to the closest valid label using semantic similarity."""
    if not label or label.strip() == "":
        return 'unknown'
    
    
    valid_labels = emb_model.encode(list(label_map.values()), convert_to_tensor=True)
    pred_emb = emb_model.encode(label[0], convert_to_tensor=True)
    cos_scores = util.cos_sim(pred_emb, valid_labels)[0]
    closest_idx = cos_scores.argmax().item()
    return list(label_map.values())[closest_idx]


def load_model_tokenizer(model_name):
    safe_model_name = model_name.replace("/", "__")
    quantized_model_folder = "4bit-Models/"
    quantized_path = os.path.join(quantized_model_folder, safe_model_name)
    os.makedirs(os.path.dirname(quantized_path), exist_ok=True)


    # Quantize the model using CL (Optional)
    def quantize():
        import shlex
        command = f"mlx_lm.convert --hf-path {model_name} --quantize --mlx-path {quantized_path}"
        cl = shlex.split(command)
        try:
            result = subprocess.run(
                cl,
                check=True,
                capture_output=True,
                text=True
            )
            print("STDOUT:\n", result.stdout)
            if result.stderr:
                print("STDERR:\n", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error code {e.returncode}")
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)

    # ONLY quantize for the first time, else load the model directly
    if not os.path.exists(quantized_path):
        quantize()
    
    model, tokenizer = load(quantized_path) #Could change back into model_name if you dont want to use the quantized version
    return model, tokenizer

def run_evaluation(y_true, y_pred, label_map):
    y_true_arr = np.array([x.lower().strip() for x in y_true])
    y_pred_arr = np.array([x.lower().strip() for x in y_pred])
    
    labels = [lab.lower() for lab in list(label_map.values())]
    
    # Calculate macro scores
    macro_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels, zero_division=0, average='macro')
    macro_recall = recall_score(y_true_arr, y_pred_arr, labels=labels, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
    mcc = matthews_corrcoef(y_true_arr, y_pred_arr)
    
    # Calculate per-class metrics
    precision_per_class_vals = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    recall_per_class_vals = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    f1_per_class_vals = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    for idx, cls in enumerate(labels):
        precision_per_class[cls] = float(precision_per_class_vals[idx])
        recall_per_class[cls] = float(recall_per_class_vals[idx])
        f1_per_class[cls] = float(f1_per_class_vals[idx])
    
    # Calculate AUPRC per class
    y_true_bin = label_binarize(y_true_arr, classes=labels)
    y_pred_bin = label_binarize(y_pred_arr, classes=labels)
    if len(labels) == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        y_pred_bin = np.hstack([1 - y_pred_bin, y_pred_bin])
    
    auprc_per_class = {}
    for idx, cls in enumerate(labels):
        ap = average_precision_score(y_true_bin[:, idx], y_pred_bin[:, idx])
        auprc_per_class[cls] = float(ap)
    
    return {
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_recall),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "auprc_per_class": auprc_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class
    }


def classify(model_name, df, label_map, shots_minority=0, shots_majority=0, max_new_tokens=3, temp=0, top_p=0, forced_maj_label=None, use_self_consistency=False, sc_num_samples=5):
    
    model, tokenizer = load_model_tokenizer(model_name)
    
    # Build prompted_text for each example
    df["prompted_text"] = df.apply(
        lambda row: build_prompt(
            df,
            row["text"],
            label_map,
            shots_minority,
            shots_majority,
            forced_maj_label=forced_maj_label
        ) + f"\nText: {row['text']}\nCategory:",
        axis=1
    )

    pred_arr = []
    start_time = time()

    if use_self_consistency:
        # Self-consistency mode: sample multiple times and aggregate
        print(f"Using self-consistency with {sc_num_samples} samples, temp={temp}")
        sc = SelfConsistency(num_samples=sc_num_samples, temperature=temp)
        
        # Use temperature > 0 for diversity
        sc_temp = 0.7 if temp <= 0 else temp  # Ensure non-zero temperature
        sampler = make_sampler(temp=sc_temp, top_p=top_p)
        
        for row in tqdm(df.itertuples(), desc="Running with Self-Consistency"):
            curr_row_prompt = row.prompted_text
            
            # Define generate function for self-consistency
            def generate_fn(prompt, **kwargs):
                result = generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max_new_tokens,
                    sampler=sampler,
                    verbose=False  # Disable verbose for cleaner output
                )
                return result if result else ""
            
            # Use self-consistency sampling and aggregation
            pred = sc.sample_and_aggregate(
                generate_fn=generate_fn,
                prompt=curr_row_prompt,
                valid_labels=list(label_map.values()),
                normalize_fn=lambda x: normalize_label(x, label_map) if x else "unknown"
            )
            pred_arr.append(pred)
    else:
        # Greedy mode: single deterministic generation
        print(f"Using greedy decoding with temp={temp}")
        sampler = make_sampler(temp=temp, top_p=top_p)
        
        for row in tqdm(df.itertuples(), desc="Running with Greedy Decoding"):
            curr_row_prompt = row.prompted_text
            res = generate(
                model,
                tokenizer,
                curr_row_prompt,
                max_tokens=max_new_tokens,
                sampler=sampler,
                verbose=True
            )
            
            # Handle empty generation
            if not res or res.strip() == "":
                print(f"Warning: Empty generation, using 'unknown'")
                pred_arr.append("unknown")
                continue
            
            # Normalize if not in valid labels
            if res not in list(label_map.values()):
                res = normalize_label(res, label_map)
            
            pred_arr.append(res)
    
    end_time = time()
    print(f"Total running time: {end_time - start_time:.2f} seconds")

    return pred_arr

def infer_metadata(ds_name, df):
    """Infer dataset metadata from name and content.

    If a forced majority label is encoded in the dataset name (or passed separately),
    that will be used. This function can be extended to accept an explicit forced
    majority label parameter if needed.
    """
    ratio = 'unknown'
    maj_label = 'unknown'

    # Check common name patterns
    if 'balanced' in ds_name:
        ratio = 'balanced'
    m = re.search(r"(\d+)_to_(\d+)", ds_name)
    if m:
        ratio = f"{m.group(1)}:{m.group(2)}"

    # Look for "_majority_" pattern
    m2 = re.search(r"([A-Za-z0-9]+)_majority", ds_name)
    if m2:
        maj_label = m2.group(1)

    # Fallback: infer from df counts
    try:
        counts = df['label'].value_counts()
        if len(counts) > 0:
            maj_label_from_df = counts.idxmax()
            if maj_label == 'unknown':
                maj_label = str(maj_label_from_df)
            # Compute numeric ratio
            maj_count = int(counts.max())
            others = counts.drop(maj_label_from_df)
            if len(others) > 0:
                min_count = int(others.min())
            else:
                min_count = 0
            ratio = f"{maj_count}:{min_count}"
    except Exception:
        pass

    return ratio, maj_label

def _save_results(results, out_dir, model_name, use_self_consistency=False):
    """Save results to CSV files with timestamp and shot metadata."""
    print(results)
    os.makedirs(out_dir, exist_ok=True)

    # Create aggregated DataFrame
    df_agg = pd.json_normalize(results)
    df_agg.columns = [c.replace('.', '_') for c in df_agg.columns]

    # Add timestamp
    timestamp = datetime.now().strftime("%m_%d_%Y")
    df_agg["saved_timestamp"] = timestamp

    # Save aggregated results (timestamped so old runs are kept)
    if use_self_consistency:
        agg_name = f"SC_results_{model_name.replace('/', '_')}_{timestamp}.csv"
    else:
        agg_name = f"ICL_results_{model_name.replace('/', '_')}_{timestamp}.csv"
    agg_path = os.path.join(out_dir, agg_name)
    
    # If there are 2 runs on the same day, we create a different version for this
    if os.path.exists(agg_path):
        create_new = input("Do you want to create another version for this, since this file name already exists? (y/n)")
        if create_new.lower() == 'y':
            agg_name = "ver2_" + agg_name
            agg_path = os.path.join(out_dir, agg_name)

    df_agg.to_csv(agg_path, index=False)



def run(model_name, datasets_dict, dataset_name, label_map, shots_minority=0, shots_majority=8, top_p=0, temp=0, max_new_tokens=0, forced_maj_label=None, use_self_consistency=False, sc_num_samples=5):
    results = []

    output_dir = os.path.join("mlx_models_results", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    min_range = [0] if shots_minority == 0 or use_self_consistency else list(range(0, shots_minority + 1, 4))
    maj_range = [0] if shots_majority == 0 or use_self_consistency else list(range(0, shots_majority + 1, 4))

    for ds_name, df in datasets_dict.items():
            if ds_name != "ag_news_imbalanced_data_99_to_1":
                print(f"=== RUNNING DATASET {ds_name} ===")
                test_df = df.sample(frac=1).reset_index(drop=True)

                if use_self_consistency:
                    preds = classify(
                                model_name,
                                test_df,
                                label_map,
                                max_new_tokens=max_new_tokens,
                                top_p=top_p,
                                temp=temp,
                                forced_maj_label=forced_maj_label,
                                use_self_consistency=use_self_consistency,
                                sc_num_samples=sc_num_samples
                            )
                    metrics = run_evaluation(test_df['label'].tolist(), preds, label_map=label_map)

                    # Infer dataset metadata
                    ratio, maj_label = infer_metadata(ds_name, df)

                    row = {
                        "model": model_name,
                        "dataset": ds_name,
                        "dataset_ratio": ratio,
                        "majority_label": maj_label,
                        **metrics
                    }
                    results.append(row)

                    # Save results incrementally
                    _save_results(results, output_dir, model_name, use_self_consistency=use_self_consistency)
                else:
                    for shot_min in min_range:
                        for shot_maj in maj_range:
                            print(f"    === SHOTS (majority={shot_maj}, minority={shot_min}) ===")
                            preds = classify(
                                model_name,
                                test_df,
                                label_map,
                                shots_minority=shot_min,
                                shots_majority=shot_maj,
                                max_new_tokens=max_new_tokens,
                                top_p=top_p,
                                temp=temp,
                                forced_maj_label=forced_maj_label,
                                use_self_consistency=use_self_consistency,
                                sc_num_samples=sc_num_samples
                            )

                            metrics = run_evaluation(test_df['label'].tolist(), preds, label_map=label_map)

                            # Infer dataset metadata
                            ratio, maj_label = infer_metadata(ds_name, df)

                            row = {
                                "model": model_name,
                                "dataset": ds_name,
                                # shot_min refers to shots_minority and shot_maj to shots_majority
                                "shots_majority": int(shot_maj),
                                "shots_minority": int(shot_min),
                                "dataset_ratio": ratio,
                                "majority_label": maj_label,
                                **metrics
                            }
                            results.append(row)

                            # Save results incrementally
                            _save_results(results, output_dir, model_name, use_self_consistency=use_self_consistency)

    return pd.DataFrame(results)


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate LLMs on imbalanced datasets")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                       help="HuggingFace model name")
    parser.add_argument("--datasets", type=str, choices=['ag_news', 'toxic_text', 'twitter_emotion'], 
                       default='ag_news',
                       help="Datasets to evaluate on")
    parser.add_argument(
        "--shot-minority",
        type=int,
        default=4,
        help="Number of shots for minority class (used if --different-shots is set)."
    )
    parser.add_argument(
        "--shot-majority",
        type=int,
        default=4,
        help="Number of shots for majority class (used if --different-shots is set)."
    )
    parser.add_argument("--data-dir", type=str, default="Data",
                       help="Directory containing the datasets")
    parser.add_argument("--top-p", type=float, default=0, help="Set Top P")
    parser.add_argument("--temperature", type=float, default=0.4, help="Set temperature for the model")
    parser.add_argument("--max-tokens", type=int, default=3)
    parser.add_argument("--majority-label", type=str, default=None,
                       help="Force a particular label to be treated as majority (value should match label text, e.g. 'sports')")
    parser.add_argument("--use-self-consistency", action='store_true', 
                       help="Use self-consistency prompting (samples multiple diverse outputs and aggregates via majority vote)")
    parser.add_argument("--sc-samples", type=int, default=1,
                       help="Number of samples for self-consistency (default: 1)")
    parser.add_argument("--sc-temperature", type=float, default=0.7,
                       help="Temperature for self-consistency sampling (default: 0.7, ignored if not using self-consistency)")
    
    args = parser.parse_args()
    
    # Label maps for each dataset
    label_maps = {
            'ag_news': {
                0: "world",
                1: "sports", 
                2: "business",
                3: "sci/tech"
            },
            'toxic_text': {
                0: "nontoxic",
                1: "toxic"
            },
            'twitter_emotion': {
                0: "sadness",
                1: "joy",
                2: "love", 
                3: "anger",
                4: "fear",
                5: "surprise"
            }
        }
    
    dl = DatasetLoader(label_maps)

    chosen = None
    variants = None
    if args.datasets == 'ag_news':
        variants = dl.load_ag_news_data(os.path.join(args.data_dir, 'ag_news'))
        chosen = variants.get('ag_news_balanced')
    elif args.datasets == 'toxic_text':
        variants = dl.load_toxic_text_data(os.path.join(args.data_dir, 'toxic_text'))
        chosen = variants.get('toxic_text')
    else:
        variants = dl.load_twitter_emotion_data(os.path.join(args.data_dir, 'twitter_emotion'))
        chosen = variants.get('emotion_df')

    print(variants)

    # dl.reduce_size(variants, 10) #Set to 100 rows per class for now for testing
    # print("Reducing dataset to 100 rows per class")

    # Get label map
    curr_label_map = label_maps[args.datasets]

    # RUN EVALS
    print()
    print(f"Original model has been quantized to 4-bit")
    
    # Use self-consistency temperature if enabled, otherwise use provided temperature
    effective_temp = args.sc_temperature if args.use_self_consistency else args.temperature
    
    if args.use_self_consistency:
        print(f"\n{'='*60}")
        print(f"SELF-CONSISTENCY MODE ENABLED")
        print(f"  - Samples per prediction: {args.sc_samples}")
        print(f"  - Sampling temperature: {effective_temp}")
        print(f"  - Aggregation: majority vote")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"GREEDY DECODING MODE")
        print(f"  - Temperature: {effective_temp}")
        print(f"{'='*60}\n")
    
    run(
        model_name=args.model, 
        datasets_dict=variants, 
        dataset_name=args.datasets, 
        label_map=curr_label_map, 
        shots_minority=args.shot_minority, 
        shots_majority=args.shot_majority,
        top_p=args.top_p,
        temp=effective_temp,
        max_new_tokens=args.max_tokens,
        forced_maj_label=args.majority_label,
        use_self_consistency=args.use_self_consistency,
        sc_num_samples=args.sc_samples
    )

    


if __name__ == "__main__":
    main()




