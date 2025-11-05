"""
Python script to run MLX Models, which are designed to be running on MacOS machines
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import os
import sys
from pathlib import Path
import numpy as np
import argparse
import re
import torch
import random
import subprocess
from collections import Counter
from typing import List, Optional, Mapping, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from packages.dataset_loader import DatasetLoader
from packages.self_consistency import SelfConsistency
from packages.min_first_voting import ThresholdBasedVoting
from evals import infer_preference as preference
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

    prompt += f"Review: \"{text}\"\nSo what is the label for this text? Answer here: "
    return prompt

def _extract_label(raw_text, label_map):
    if not raw_text:
        return None

    canon_map = {v.lower(): v for v in label_map.values()}
    lowered = raw_text.strip().lower()

    explicit_pattern = re.compile(r"\b(?:answer|category|label)\s*[:\-]\s*([a-z/]+)")
    for match in explicit_pattern.finditer(lowered):
        candidate = match.group(1)
        if candidate in canon_map:
            return canon_map[candidate]

    for line in lowered.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        for key, original in canon_map.items():
            if re.fullmatch(rf"(?:the\s+)?{re.escape(key)}", line_clean):
                return original

    tokens = re.findall(r"[a-z/]+", lowered)
    for token in tokens:
        if token in canon_map:
            return canon_map[token]

    return None

emb_model = SentenceTransformer("all-MiniLM-L6-v2")
def normalize_label(label, label_map):
    """Normalize a predicted label to the closest valid label using semantic similarity."""
    if not label or label.strip() == "":
        return 'unknown'
    
    direct = _extract_label(label, label_map)
    if direct:
        return direct

    valid_labels = emb_model.encode(list(label_map.values()), convert_to_tensor=True)
    pred_emb = emb_model.encode(label, convert_to_tensor=True)
    cos_scores = util.cos_sim(pred_emb, valid_labels)[0]
    closest_idx = cos_scores.argmax().item()
    return list(label_map.values())[closest_idx]


def load_model_tokenizer(model_name, use_quantize=False):
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
    needs_quant = not os.path.isdir(quantized_path) or len(os.listdir(quantized_path)) == 0
    if use_quantize:
        if needs_quant:
            print(f"Quantized weights not found for {model_name}. Running mlx_lm.convert...")
            quantize()

            try:
                model, tokenizer = load(quantized_path)
            except Exception as err:
                print(f"Failed to load quantized weights from {quantized_path}: {err}")
                print("Falling back to loading original HF model (may be slower and require bias-compatible architecture)...")
                model, tokenizer = load(model_name)
    else:
        try:
            print("Using the original 16-bit model...")
            model, tokenizer = load(model_name)
        except Exception as e:
            print(f"Failed to load {model_name}")
    return model, tokenizer


def _prepare_pref_dataset(df: pd.DataFrame, label_map: Mapping[int, str]) -> preference.Dataset:
    labels = list(dict.fromkeys(label_map.values()))
    records = []
    for idx, row in enumerate(df.itertuples()):
        raw_label = getattr(row, "label")
        if isinstance(raw_label, (int, np.integer)):
            label = label_map.get(int(raw_label), str(raw_label))
        else:
            label = str(raw_label)
        records.append({
            "id": getattr(row, "id", idx),
            "text": str(row.text),
            "label": label,
        })

    counts = Counter(record["label"] for record in records if record.get("label") in labels)
    total = sum(counts.values())
    if total == 0:
        prior = {label: 1.0 / max(len(labels), 1) for label in labels}
    else:
        prior = {label: counts.get(label, 0) / total for label in labels}

    return preference.Dataset(records=records, labels=labels, dataset_prior=prior)


def _infer_weak_labels_from_samples(
    all_samples: List[List[str]],
    true_labels: List[str],
    valid_labels: List[str],
    weak_percent: int,
) -> List[str]:
    """Infer weak labels (under-represented classes) from sampling results using recall."""
    # Use majority vote from samples as predictions
    predictions = []
    for samples in all_samples:
        if samples:
            vote_counts = Counter(samples)
            predictions.append(vote_counts.most_common(1)[0][0])
        else:
            predictions.append("unknown")
    
    # Calculate per-class recall
    y_true = np.array([str(x).lower().strip() for x in true_labels])
    y_pred = np.array([str(x).lower().strip() for x in predictions])
    labels_lower = [lab.lower() for lab in valid_labels]
    
    recall_vals = recall_score(y_true, y_pred, labels=labels_lower, average=None, zero_division=0)
    
    recall_scores = {}
    for idx, label in enumerate(valid_labels):
        recall_scores[label] = float(recall_vals[idx])
    
    # Identify weak labels: either bottom percent OR all labels with recall below mean
    labels_sorted = sorted(recall_scores.items(), key=lambda x: x[1])
    
    # Use percentage-based selection
    count = max(1, int(len(labels_sorted) * weak_percent / 100))
    weak_labels = [label for label, score in labels_sorted[:count]]
    
    # Alternative: Also include all labels with recall significantly below average
    mean_recall = np.mean(list(recall_scores.values()))
    low_recall_labels = [label for label, score in recall_scores.items() if score < mean_recall * 0.5]
    
    # Combine both approaches
    weak_labels = list(set(weak_labels) | set(low_recall_labels))
    
    print(f"[INFO] Calculated recall scores from samples: {recall_scores}")
    print(f"[INFO] Mean recall: {mean_recall:.3f}")
    print(f"[INFO] Identified weak labels (low recall): {weak_labels}")
    
    return weak_labels


def _detect_preferred_label(
    df: pd.DataFrame,
    label_map: Mapping,
    model_name: str,
    shots_minority: int,
    shots_majority: int,
    temp: float,
    top_p: float,
    max_new_tokens: int,
    use_quantize: bool,
    min_difference: float = 0.5,
) -> List[str]:
    """
    Detect the preferred (biased) labels by running ICL on a small balanced sample.
    
    Returns all labels where recall/precision > 1 (model is biased toward them).
    """
    print("\n[INFO] Detecting preferred labels from balanced sample...")
    
    # Create balanced sample (equal rows per class)
    balanced_dfs = []
    min_class_size = df['label'].value_counts().min()
    sample_size = min(10, min_class_size)  # Use up to 10 samples per class
    
    for label in df['label'].unique():
        label_df = df[df['label'] == label].sample(n=sample_size, random_state=42)
        balanced_dfs.append(label_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"[INFO] Using balanced sample: {len(balanced_df)} rows ({sample_size} per class)")
    
    # Run ICL inference with greedy decoding
    preds, _, _ = classify(
        model_name,
        balanced_df,
        label_map,
        shots_minority=shots_minority,
        shots_majority=shots_majority,
        max_new_tokens=max_new_tokens,
        temp=temp,
        top_p=top_p,
        use_self_consistency=False,
        use_quantize=use_quantize,
        collect_label_counts=False,
    )
    
    # Calculate metrics
    metrics = run_evaluation(balanced_df['label'].tolist(), preds, label_map=label_map)
    
    # Calculate preference scores (recall / precision)
    preference_scores = {}
    precision_per_class = metrics.get("precision_per_class", {})
    recall_per_class = metrics.get("recall_per_class", {})
    
    for label in label_map.values():
        label_lower = label.lower()
        prec = precision_per_class.get(label_lower, 0.0)
        rec = recall_per_class.get(label_lower, 0.0)
        
        if prec > 0:
            preference_scores[label] = rec / prec
        else:
            preference_scores[label] = 0.0
    
    print(f"[INFO] Preference scores (recall/precision): {preference_scores}")
    
    # Find labels where recall/precision >= 1 OR significantly different from others
    preferred_labels = []
    sorted_prefs = sorted(preference_scores.items(), key=lambda x: x[1], reverse=True)
    
    for label, score in sorted_prefs:
        # Check if this label has significantly higher score than all others
        is_significantly_higher = all(
            label == other_label or score - other_score >= min_difference
            for other_label, other_score in sorted_prefs
        )
        
        # Add to preferred if score >= 1.0 OR significantly higher than others
        if score >= 1.0 or is_significantly_higher:
            preferred_labels.append(label)
            reason = []
            if score >= 1.0:
                reason.append(f"score >= 1.0")
            if is_significantly_higher:
                reason.append(f"significantly higher")
            print(f"[INFO] Detected preferred label: '{label}' (score={score:.3f}, reason: {' and '.join(reason)})")
    
    if not preferred_labels:
        print(f"[INFO] No preferred labels detected")
    else:
        print(f"[INFO] Total preferred labels: {preferred_labels}")
    
    return preferred_labels


def _classify_minority_first(
    model_name: str,
    df: pd.DataFrame,
    label_map,
    *,
    temp: float,
    top_p: float,
    max_new_tokens: int,
    samples_per_example: int,
    weak_labels,
    weak_from_metrics,
    weak_percent: int,
    prompt_template: Optional[str],
    seed: int,
    collect_label_counts: bool,
    auto_infer_weak: bool = False,
    use_threshold: bool = True,
    threshold: int = 20,
    shots_minority: int = 0,
    shots_majority: int = 0,
    use_quantize: bool = False,
    model=None,
    tokenizer=None,
) -> Tuple[List[str], List[dict], List[tuple]]:
    """Apply threshold-based preference mitigation (replaces minority-first voting)."""
    preference.set_seed(seed)
    dataset = _prepare_pref_dataset(df, label_map)
    
    print(f"\n[INFO] Using threshold-based preference mitigation (threshold={threshold})")
    
    # Detect preferred labels from balanced sample
    preferred_labels = _detect_preferred_label(
        df=df,
        label_map=label_map,
        model_name=model_name,
        shots_minority=shots_minority,
        shots_majority=shots_majority,
        temp=temp if temp and temp > 0 else 0.0,
        top_p=top_p if top_p and top_p > 0 else 0.0,
        max_new_tokens=max_new_tokens if max_new_tokens and max_new_tokens > 0 else 32,
        use_quantize=use_quantize,
        min_difference=0.5,
    )
    
    if not preferred_labels:
        print("[WARN] Could not detect any preferred labels, falling back to majority voting")
    else:
        print(f"[INFO] Detected preferred labels: {preferred_labels}")
        print(f"[INFO] Will use shots_majority={shots_majority} for preferred labels, shots_minority={shots_minority} for others")
    
    # Determine forced_maj_label: use first preferred label if detected
    forced_maj_label = preferred_labels[0] if preferred_labels else None
    
    # Run sampling with threshold-based aggregation
    from packages.min_first_voting import choose_with_threshold_override
    from mlx_lm import load as mlx_load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    
    # Only load model if not provided (reuse from caller to avoid reloading)
    if model is None or tokenizer is None:
        print("[INFO] Loading model (this is slow - consider passing model/tokenizer to avoid reloading)")
        model, tokenizer = load_model_tokenizer(model_name, use_quantize)
    else:
        print("[INFO] Reusing provided model/tokenizer (faster!)")
    
    predictions: List[str] = []
    label_count_records: List[dict] = []
    label_pairs: List[tuple] = []
    
    # Create sampler once, reuse for all samples (performance optimization)
    effective_temp = temp if temp and temp > 0 else 0.7
    effective_top_p = top_p if top_p and top_p > 0 else 0.9
    effective_max_tokens = max_new_tokens if max_new_tokens and max_new_tokens > 0 else 32
    sampler = make_sampler(temp=effective_temp, top_p=effective_top_p)
    
    print(f"[INFO] Processing {len(dataset.records)} examples with {samples_per_example} samples each...")
    
    for idx, record in enumerate(tqdm(dataset.records, desc="Classifying")):
        text = record.get("text", "")
        true_label = record.get("label")
        
        # Generate multiple samples using build_prompt with preferred label as majority
        samples = []
        # Use build_prompt to create few-shot prompt with preferred label getting shots_majority
        prompt = build_prompt(
            df=df,
            text=text,
            label_map=label_map,
            shots_minority=shots_minority,
            shots_majority=shots_majority,
            forced_maj_label=forced_maj_label
        )
        
        for _ in range(max(1, samples_per_example)):
            try:
                response = mlx_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=effective_max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                if response:
                    normalized = normalize_label(response, label_map)
                    samples.append(normalized)
            except Exception as e:
                samples.append("unknown")
        
        # Apply threshold-based decision
        final_label, mode = choose_with_threshold_override(samples, preferred_labels, threshold)
        predictions.append(final_label or "unknown")
        
        if collect_label_counts:
            counts = Counter(samples)
            record_data = {
                "prompt_index": idx,
                "text": text,
                "true_label": true_label,
                "predicted_label": final_label,
                "total_samples": len(samples),
                "sampling_temperature": temp if temp and temp > 0 else 0.7,
                "decision_mode": mode,
                "preferred_labels": preferred_labels,
                "threshold": threshold,
            }
            for label in dataset.labels:
                record_data[f"count_{label}"] = counts.get(label, 0)
            record_data["count_unknown"] = counts.get("unknown", 0)
            label_count_records.append(record_data)
        
        for sample_pred in samples:
            label_pairs.append((sample_pred, true_label))
    
    return predictions, label_count_records, label_pairs


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
    preference_per_class = {}
    for idx, cls in enumerate(labels):
        prec = float(precision_per_class_vals[idx])
        rec = float(recall_per_class_vals[idx])
        precision_per_class[cls] = prec
        recall_per_class[cls] = rec
        f1_per_class[cls] = float(f1_per_class_vals[idx])
        # Calculate preference as recall/precision (higher = more false positives relative to true positives)
        if prec > 0:
            preference_per_class[cls] = rec / prec
        else:
            preference_per_class[cls] = 0.0
    
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
        "f1_per_class": f1_per_class,
        "preference_per_class": preference_per_class
    }


def classify(
    model_name,
    df,
    label_map,
    shots_minority=0,
    shots_majority=0,
    max_new_tokens=3,
    temp=0,
    top_p=0,
    forced_maj_label=None,
    use_self_consistency=False,
    sc_num_samples=5,
    use_quantize=False,
    collect_label_counts=False,
    label_count_samples=None,
    label_count_temperature=None,
    minority_first=False,
    mf_samples=None,
    mf_prompt_template=None,
    mf_threshold=20,
    seed=0,
):

    # Early exit when minority-first voting pipeline is requested
    if minority_first:
        effective_samples = mf_samples
        if not effective_samples or effective_samples <= 0:
            effective_samples = label_count_samples or (sc_num_samples if use_self_consistency else 5)
        
        # Load model ONCE before calling threshold-based function
        model, tokenizer = load_model_tokenizer(model_name, use_quantize)
        
        preds_mf, label_counts_mf, label_pairs_mf = _classify_minority_first(
            model_name,
            df,
            label_map,
            temp=temp,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            samples_per_example=max(1, effective_samples),
            weak_labels=None,
            weak_from_metrics=None,
            weak_percent=25,
            prompt_template=mf_prompt_template,
            seed=seed,
            collect_label_counts=collect_label_counts,
            auto_infer_weak=False,
            use_threshold=True,
            threshold=mf_threshold,
            shots_minority=shots_minority,
            shots_majority=shots_majority,
            use_quantize=use_quantize,
            model=model,
            tokenizer=tokenizer,
        )
        return preds_mf, label_counts_mf, label_pairs_mf

    model, tokenizer = load_model_tokenizer(model_name, use_quantize)

    df = df.copy()
    df["prompted_text"] = df.apply(
        lambda row: build_prompt(
            df,
            row["text"],
            label_map,
            shots_minority,
            shots_majority,

            forced_maj_label=forced_maj_label,
        )
        + f"\nText: {row['text']}\nCategory:",
        axis=1,
    )

    pred_arr: List[str] = []
    label_count_records: List[dict] = []
    label_pairs: List[tuple] = []  # Store (predicted, ground_truth) pairs
    collect_counts = bool(collect_label_counts)
    effective_label_count_samples = (
        label_count_samples if label_count_samples is not None else (sc_num_samples if use_self_consistency else 1)
    )
    if effective_label_count_samples is None or effective_label_count_samples <= 0:
        effective_label_count_samples = 1
    effective_label_count_temperature = (
        label_count_temperature if label_count_temperature is not None else temp
    )

    start_time = time()

    if use_self_consistency:
        sc_temp = effective_label_count_temperature if (collect_counts and label_count_temperature is not None) else temp
        if sc_temp <= 0:
            sc_temp = 0.7

        print(f"Using self-consistency with {sc_num_samples} samples, temp={sc_temp}")
        sc = SelfConsistency(num_samples=sc_num_samples, temperature=sc_temp)
        valid_labels = list(label_map.values())

        for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Self-Consistency")):
            sampler = make_sampler(temp=sc_temp, top_p=top_p)
            samples_collected: List[str] = []

            def generate_fn(prompt, **kwargs):
                result = generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max_new_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                result = result or ""
                samples_collected.append(result)
                return result

            pred = sc.sample_and_aggregate(
                generate_fn=generate_fn,
                prompt=row.prompted_text,
                valid_labels=valid_labels,
                normalize_fn=lambda x: normalize_label(x, label_map),
            )
            pred_arr.append(pred)
            
            if collect_counts:
                normalized_samples = [
                    normalize_label(sample, label_map) if sample else "unknown" for sample in samples_collected
                ]
                
                # Collect multiple label pairs - one for each sample
                ground_truth = label_map.get(row.label, str(row.label))
                for sample_pred in normalized_samples:
                    label_pairs.append((sample_pred, ground_truth))
                
                counts = Counter(normalized_samples)
                record = {
                    "prompt_index": idx,
                    "text": row.text,
                    "true_label": row.label,
                    "predicted_label": pred,
                    "total_samples": len(normalized_samples),
                    "sampling_temperature": sc_temp,
                }
                for label in valid_labels:
                    record[f"count_{label}"] = counts.get(label, 0)
                record["count_unknown"] = counts.get("unknown", 0)
                label_count_records.append(record)
    else:
        print(f"Using greedy decoding with temp={temp}")
        base_sampler = make_sampler(temp=temp, top_p=top_p)

        for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Greedy Decoding")):
            prompt = row.prompted_text
            if collect_counts:
                samples_needed = max(1, effective_label_count_samples)
                temperature_to_use = effective_label_count_temperature
                normalized_samples: List[str] = []

                for _ in range(samples_needed):
                    sampler = make_sampler(temp=temperature_to_use, top_p=top_p)
                    res = generate(
                        model,
                        tokenizer,
                        prompt,
                        max_tokens=max_new_tokens,
                        sampler=sampler,
                        verbose=False,
                    )
                    res = res or ""
                    normalized_samples.append(normalize_label(res, label_map))

                counts = Counter(normalized_samples)
                if normalized_samples:
                    top_label = counts.most_common(1)[0][0]
                else:
                    top_label = "unknown"
                pred_arr.append(top_label)
                
                # Collect multiple label pairs - one for each sample
                ground_truth = label_map.get(row.label, str(row.label))
                for sample_pred in normalized_samples:
                    label_pairs.append((sample_pred, ground_truth))

                record = {
                    "prompt_index": idx,
                    "text": row.text,
                    "true_label": row.label,
                    "predicted_label": top_label,
                    "total_samples": len(normalized_samples),
                    "sampling_temperature": temperature_to_use,
                }
                for label in label_map.values():
                    record[f"count_{label}"] = counts.get(label, 0)
                record["count_unknown"] = counts.get("unknown", 0)
                label_count_records.append(record)
            else:
                res = generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max_new_tokens,
                    sampler=base_sampler,
                    verbose=True,
                )

                if not res or res.strip() == "":
                    print("Warning: Empty generation, using 'unknown'")
                    pred_arr.append("unknown")
                    continue

                if res not in list(label_map.values()):
                    res = normalize_label(res, label_map)

                pred_arr.append(res)

    end_time = time()
    print(f"Total running time: {end_time - start_time:.2f} seconds")

    return pred_arr, label_count_records, label_pairs

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

def _save_results(results, out_dir, model_name, use_self_consistency=False, minority_first=False):
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
    if minority_first:
        agg_name = f"TB_results_{model_name.replace('/', '_')}_{timestamp}.csv"
    elif use_self_consistency:
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


def _save_label_counts(
    records,
    dataset_name,
    variant_name,
    model_name,
    shots_minority,
    shots_majority,
    sc_samples,
    out_dir,
    label_count_output_dir=None,
):
    if not records:
        return

    df = pd.DataFrame(records)
    timestamp = datetime.now().strftime("%m_%d_%Y_%H%M%S")
    
    # Add timestamp column
    df["timestamp"] = timestamp
    
    # Add shots as a single integer (assuming minority and majority are the same)
    df["shots"] = int(shots_minority)
    
    # Select and reorder columns to keep only what's needed
    # Keep: text (input), total_samples (n_samples), shots, count_*, timestamp
    cols_to_keep = ["text", "total_samples", "shots", "sampling_temperature"]
    
    # Add all count_* columns
    count_cols = [col for col in df.columns if col.startswith("count_")]
    cols_to_keep.extend(count_cols)
    cols_to_keep.append("timestamp")
    
    # Filter to only the columns we want
    df = df[cols_to_keep]

    target_dir = label_count_output_dir or os.path.join(out_dir, "label_counts")
    os.makedirs(target_dir, exist_ok=True)

    prefix = "SC" if sc_samples else "ICL"
    sc_suffix = f"_sc{int(sc_samples)}" if sc_samples else ""
    # Use model name in filename instead of shot counts
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    filename = (
        f"{prefix}_label_counts_{variant_name}_{safe_model_name}{sc_suffix}_{timestamp}.csv"
    )
    path = os.path.join(target_dir, filename)
    df.to_csv(path, index=False)
    print(f"Label counts saved to {path}")


def _save_label_pairs(
    label_pairs,
    dataset_name,
    variant_name,
    model_name,
    sc_samples,
    out_dir,
    label_count_output_dir=None,
):
    """Save (predicted, ground_truth) label pairs as a PyTorch .pt file."""
    if not label_pairs:
        return

    timestamp = datetime.now().strftime("%m_%d_%Y_%H%M%S")
    target_dir = label_count_output_dir or os.path.join(out_dir, "label_counts")
    os.makedirs(target_dir, exist_ok=True)

    prefix = "SC" if sc_samples else "ICL"
    sc_suffix = f"_sc{int(sc_samples)}" if sc_samples else ""
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    
    filename = (
        f"{prefix}_label_pairs_{variant_name}_{safe_model_name}{sc_suffix}_{timestamp}.pt"
    )
    path = os.path.join(target_dir, filename)
    
    # Save as PyTorch tensor file
    torch.save(label_pairs, path)
    print(f"Label pairs saved to {path}")


def run(
    model_name,
    datasets_dict,
    dataset_name,
    label_map,
    shots_minority=0,
    shots_majority=8,
    top_p=0,
    temp=0,
    max_new_tokens=0,
    forced_maj_label=None,
    use_self_consistency=False,
    sc_num_samples=5,
    use_quantize=False,
    collect_label_counts=False,
    label_count_samples=None,
    label_count_temperature=None,
    label_count_output_dir=None,
    label_counts_only=False,
    minority_first=False,
    mf_samples=None,
    mf_prompt_template=None,
    mf_threshold=20,
    seed=0,
):
    results = []

    output_dir = os.path.join("mlx_models_results", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # If label_counts_only is True, automatically enable collect_label_counts
    if label_counts_only:
        collect_label_counts = True

    # When label_counts_only is True, use fixed shot counts (no loop)
    if label_counts_only:
        min_range = [shots_minority]
        maj_range = [shots_majority]
    else:
        min_range = [0] if shots_minority == 0 or use_self_consistency else list(range(0, shots_minority + 1, 4))
        maj_range = [0] if shots_majority == 0 or use_self_consistency else list(range(0, shots_majority + 1, 4))

    # START TIME
    start_time = datetime.now()

    for ds_name, df in datasets_dict.items():
            if ds_name != "ag_news_imbalanced_data_99_to_1":
                print(f"=== RUNNING DATASET {ds_name} ===")
                
                # Use fixed seed for reproducible sampling when collecting label counts
                if label_counts_only:
                    test_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                else:
                    test_df = df.sample(frac=1).reset_index(drop=True)

                if minority_first:
                    mf_sample_count = mf_samples if mf_samples and mf_samples > 0 else None
                    if not mf_sample_count or mf_sample_count <= 0:
                        mf_sample_count = label_count_samples or (sc_num_samples if use_self_consistency else 5)
                    mf_sample_count = max(1, int(mf_sample_count))

                    preds, label_counts, label_pairs = classify(
                        model_name,
                        test_df,
                        label_map,
                        max_new_tokens=max_new_tokens,
                        top_p=top_p,
                        temp=temp,
                        forced_maj_label=forced_maj_label,
                        use_self_consistency=use_self_consistency,
                        sc_num_samples=sc_num_samples,
                        use_quantize=use_quantize,
                        collect_label_counts=collect_label_counts,
                        label_count_samples=label_count_samples,
                        label_count_temperature=label_count_temperature,
                        minority_first=True,
                        mf_samples=mf_sample_count,
                        mf_prompt_template=mf_prompt_template,
                        mf_threshold=mf_threshold,
                        seed=seed,
                    )

                    if not label_counts_only:
                        metrics = run_evaluation(test_df['label'].tolist(), preds, label_map=label_map)
                        ratio, maj_label = infer_metadata(ds_name, df)
                        row = {
                            "model": model_name,
                            "dataset": ds_name,
                            "dataset_ratio": ratio,
                            "majority_label": maj_label,
                            "minority_first": True,
                            **metrics,
                        }
                        results.append(row)

                    if collect_label_counts:
                        _save_label_counts(
                            label_counts,
                            dataset_name,
                            ds_name,
                            model_name,
                            mf_sample_count,
                            mf_sample_count,
                            mf_sample_count,
                            output_dir,
                            label_count_output_dir,
                        )

                        if "balanced" in ds_name.lower():
                            _save_label_pairs(
                                label_pairs,
                                dataset_name,
                                ds_name,
                                model_name,
                                mf_sample_count,
                                output_dir,
                                label_count_output_dir,
                            )

                    if not label_counts_only:
                        _save_results(results, output_dir, model_name, use_self_consistency=use_self_consistency, minority_first=minority_first)
                    continue

                if use_self_consistency:
                    shot_min = min_range[0] if min_range else 0
                    shot_maj = maj_range[0] if maj_range else 0
                    preds, label_counts, label_pairs = classify(
                        model_name,
                        test_df,
                        label_map,
                        max_new_tokens=max_new_tokens,
                        top_p=top_p,
                        temp=temp,
                        forced_maj_label=forced_maj_label,
                        use_self_consistency=use_self_consistency,
                        sc_num_samples=sc_num_samples,
                        use_quantize=use_quantize,
                        collect_label_counts=collect_label_counts,
                        label_count_samples=label_count_samples,
                        label_count_temperature=label_count_temperature,
                        seed=seed,
                    )
                    
                    # Only compute metrics if not in label_counts_only mode
                    if not label_counts_only:
                        metrics = run_evaluation(test_df['label'].tolist(), preds, label_map=label_map)

                        ratio, maj_label = infer_metadata(ds_name, df)

                        row = {
                            "model": model_name,
                            "dataset": ds_name,
                            "dataset_ratio": ratio,
                            "majority_label": maj_label,
                            **metrics,
                        }
                        results.append(row)

                    if collect_label_counts:
                        _save_label_counts(
                            label_counts,
                            dataset_name,
                            ds_name,
                            model_name,
                            shot_min,
                            shot_maj,
                            sc_num_samples,
                            output_dir,
                            label_count_output_dir,
                        )
                        
                        # Save label pairs only for balanced dataset
                        if "balanced" in ds_name.lower():
                            _save_label_pairs(
                                label_pairs,
                                dataset_name,
                                ds_name,
                                model_name,
                                sc_num_samples,
                                output_dir,
                                label_count_output_dir,
                            )

                    # Only save results if we computed metrics
                    if not label_counts_only:
                        _save_results(results, output_dir, model_name, use_self_consistency=use_self_consistency, minority_first=False)
                else:
                    for shot_min in min_range:
                        for shot_maj in maj_range:
                            print(f"    === SHOTS (majority={shot_maj}, minority={shot_min}) ===")
                            preds, label_counts, label_pairs = classify(
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
                                sc_num_samples=sc_num_samples,
                                use_quantize=use_quantize,
                                collect_label_counts=collect_label_counts,
                                label_count_samples=label_count_samples,
                                label_count_temperature=label_count_temperature,
                                seed=seed,
                            )

                            # Only compute metrics if not in label_counts_only mode
                            if not label_counts_only:
                                metrics = run_evaluation(test_df['label'].tolist(), preds, label_map=label_map)

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

                            if collect_label_counts:
                                _save_label_counts(
                                    label_counts,
                                    dataset_name,
                                    ds_name,
                                    model_name,
                                    shot_min,
                                    shot_maj,
                                    None,
                                    output_dir,
                                    label_count_output_dir,
                                )
                                
                                # Save label pairs only for balanced dataset
                                if "balanced" in ds_name.lower():
                                    _save_label_pairs(
                                        label_pairs,
                                        dataset_name,
                                        ds_name,
                                        model_name,
                                        None,
                                        output_dir,
                                        label_count_output_dir,
                                    )

                            # Only save results if we computed metrics
                            if not label_counts_only:
                                # Save results incrementally
                                _save_results(results, output_dir, model_name, use_self_consistency=use_self_consistency, minority_first=False)

    # END TIME
    end_time = datetime.now()
    
    # Calculate elapsed time
    elapsed = end_time - start_time
    total_seconds = int(elapsed.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    print(f"Total run time for the run: {hours}h {minutes}m {seconds}s")

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
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--majority-label", type=str, default=None,
                       help="Force a particular label to be treated as majority (value should match label text, e.g. 'sports')")
    parser.add_argument("--use-self-consistency", action='store_true', 
                       help="Use self-consistency prompting (samples multiple diverse outputs and aggregates via majority vote)")
    parser.add_argument("--sc-samples", type=int, default=1,
                       help="Number of samples for self-consistency (default: 1)")
    parser.add_argument("--sc-temperature", type=float, default=0.7,
                       help="Temperature for self-consistency sampling (default: 0.7, ignored if not using self-consistency)")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument(
        "--rows-per-class",
        type=int,
        default=0,
        help="Subsample each dataset variant to N rows per class (default: 0 keeps all rows)",
    )
    parser.add_argument(
        "--collect-label-counts",
        action="store_true",
        help="Collect per-prompt label frequency statistics",
    )
    parser.add_argument(
        "--label-count-samples",
        type=int,
        default=0,
        help="Number of samples to draw per prompt when collecting label counts (default: 0 uses 1 or sc-samples)",
    )
    parser.add_argument(
        "--label-count-temperature",
        type=float,
        default=None,
        help="Temperature to use for label-count sampling (defaults to evaluation temperature)",
    )
    parser.add_argument(
        "--label-count-output-dir",
        type=str,
        default="mlx_models_results/label_counts",
        help="Directory where label-count summaries are saved",
    )
    parser.add_argument(
        "--label-counts-only",
        action="store_true",
        help="Only collect label counts without running full evaluation metrics (automatically enables --collect-label-counts)",
    )
    parser.add_argument(
        "--minority-first",
        action="store_true",
        help="Apply threshold-based preference mitigation using multiple samples per prompt.",
    )
    parser.add_argument(
        "--mf-samples",
        type=int,
        default=0,
        help="Samples per example for threshold-based voting (defaults to label-count or SC samples).",
    )
    parser.add_argument(
        "--mf-prompt-template",
        type=str,
        help="Custom prompt template for threshold-based voting (uses {text} and {label_list}).",
    )
    parser.add_argument(
        "--mf-threshold",
        type=int,
        default=20,
        help="Threshold count for accepting preferred label (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling operations.",
    )
    
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

    if args.rows_per_class > 0:
        variants = dl.reduce_size(variants, args.rows_per_class)

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
    
    if args.label_counts_only:
        print(f"\n{'='*60}")
        print(f"LABEL COUNTS ONLY MODE")
        print(f"  - Will collect label frequency statistics only")
        print(f"  - Metrics evaluation will be skipped")
        print(f"{'='*60}\n")
    elif args.use_self_consistency:
        print(f"\n{'='*60}")
        print(f"SELF-CONSISTENCY MODE ENABLED")
        print(f"  - Samples per prediction: {args.sc_samples}")
        print(f"  - Sampling temperature: {effective_temp}")
        print(f"  - Aggregation: majority vote")
        print(f"{'='*60}\n")
    elif args.minority_first:
        print(f"\n{'='*60}")
        print(f"THRESHOLD-BASED PREFERENCE MITIGATION MODE")
        print(f"  - Threshold: {args.mf_threshold}")
        print(f"  - Samples per example: {args.mf_samples if args.mf_samples > 0 else 'auto'}")
        print(f"  - Temperature: {effective_temp}")
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
        sc_num_samples=args.sc_samples,
        use_quantize=args.quantize,
        collect_label_counts=args.collect_label_counts,
        label_count_samples=args.label_count_samples if args.label_count_samples > 0 else None,
        label_count_temperature=args.label_count_temperature,
        label_count_output_dir=args.label_count_output_dir,
        label_counts_only=args.label_counts_only,
        minority_first=args.minority_first,
        mf_samples=args.mf_samples,
        mf_prompt_template=args.mf_prompt_template,
        mf_threshold=args.mf_threshold,
        seed=args.seed,
    )

    


if __name__ == "__main__":
    main()




