
import os
import argparse
import random
import re
from datetime import datetime
from time import time

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import pipeline, logging
from sklearn.metrics import (
    f1_score, recall_score, balanced_accuracy_score,
    matthews_corrcoef, precision_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from huggingface_hub import login
from datetime import datetime
from self_consistency import SelfConsistency

# Suppress warnings
logging.set_verbosity_info()

class LLMEvaluator:
    """Main class for evaluating LLMs on imbalanced datasets."""
    
    def __init__(self, model_name, device=None):
        """
        Initialize the evaluator.
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to use ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Label mappings for different datasets
        self.label_maps = {
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
        
        # Initialize valid embeddings for normalization
        self.valid_embeddings = {}
        self.valid_labels = {}
        
    def _setup_device(self, device):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using MPS device")
            elif torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA device")
            else:
                device = "cpu"
                print("Using CPU device")
        else:
            print(f"Using specified device: {device}")
        return device
    
    def authenticate_hf(self, token=None):
        if token is None:
            # Try to get token from environment
            load_dotenv()
            token = os.getenv("hf_token")
            
        if token:
            login(token=token)
            print("Successfully authenticated with HuggingFace Hub")
        else:
            print("Warning: No HuggingFace token provided. Some models may not be accessible.")
    
    def clean_time(self, time_seconds):
        """Format time in a readable format."""
        if time_seconds <= 60:
            return f"{time_seconds:.2f} seconds"
        
        minutes = int(time_seconds // 60)
        remaining_seconds = time_seconds - minutes * 60
        return f"{minutes} minutes, {remaining_seconds:.2f} seconds"
    
    def load_ag_news_data(self, data_dir="Data/ag_news"):
        label_map = self.label_maps['ag_news']
        
        # Load existing prepared files
        ag_news_imbalanced_data_99_to_1 = pd.read_parquet(f"{data_dir}/ag_news_train_imbalanced_99_to_1.parquet")
        balanced_data = pd.read_parquet(f"{data_dir}/ag_news_train_balanced.parquet")
        ag_news_imbalanced_data_49_to_1 = pd.read_parquet(f"{data_dir}/ag_news_train_imbalanced_49_to_1_ratio.parquet")
        
        # Map numeric labels to text labels
        balanced_data["label"] = balanced_data["label"].map(label_map)
        ag_news_imbalanced_data_99_to_1["label"] = ag_news_imbalanced_data_99_to_1["label"].map(label_map)
        ag_news_imbalanced_data_49_to_1["label"] = ag_news_imbalanced_data_49_to_1["label"].map(label_map)
        
        # Shuffle datasets
        ag_news_imbalanced_data_99_to_1 = ag_news_imbalanced_data_99_to_1.sample(frac=1).reset_index(drop=True)
        balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
        ag_news_imbalanced_data_49_to_1 = ag_news_imbalanced_data_49_to_1.sample(frac=1).reset_index(drop=True)
        
        # Create additional imbalanced datasets
        ag_news_world_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'world', 980, 20)
        ag_news_sports_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'sports', 980, 20)
        ag_news_business_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'business', 980, 20)
        
        return {
            "ag_news_balanced": balanced_data,
            "ag_news_imbalanced_data_99_to_1": ag_news_imbalanced_data_99_to_1,
            "ag_news_imbalanced_data_49_to_1": ag_news_imbalanced_data_49_to_1,
            "ag_news_world_majority_99": ag_news_world_majority_99,
            "ag_news_sports_majority_99": ag_news_sports_majority_99,
            "ag_news_business_majority_99": ag_news_business_majority_99
        }
    
    def _split_ratio_for_ag_news(self, df, majority_label, majority_count, minority_count):
        parts = []
        labels = df['label'].unique().tolist()
        for lab in labels:
            if lab == majority_label:
                parts.append(df[df['label'] == lab].sample(majority_count, random_state=42))
            else:
                parts.append(df[df['label'] == lab].sample(minority_count, random_state=42))
        out = pd.concat(parts, ignore_index=True, sort=False)
        return out.sample(frac=1).reset_index(drop=True)
    
    def load_toxic_text_data(self, data_dir="Data/toxic_text"):
        """Load and prepare toxic text datasets."""
        toxic_label_map = self.label_maps['toxic_text']
        
        toxic_text = pd.read_csv(f"{data_dir}/train.csv")
        toxic_text = toxic_text[["comment_text", "toxic"]]
        toxic_text = toxic_text.rename(columns={"comment_text": "text", "toxic": "label"})
        toxic_text["label"] = toxic_text["label"].map(toxic_label_map)
        
        # Create different imbalanced datasets
        toxic_balanced = self._split_ratio_for_toxic_dataset(toxic_text, 'nontoxic', 500, 500)
        toxic_99_to_1 = self._split_ratio_for_toxic_dataset(toxic_text, 'nontoxic', 980, 20)
        toxic_49_to_1 = self._split_ratio_for_toxic_dataset(toxic_text, 'nontoxic', 940, 20)
        toxic_toxic_majority_99 = self._split_ratio_for_toxic_dataset(toxic_text, 'toxic', 980, 20)
        
        return {
            "toxic_text": toxic_balanced,
            "toxic_99_to_1": toxic_99_to_1,
            "toxic_49_to_1": toxic_49_to_1,
            "toxic_toxic_majority_99": toxic_toxic_majority_99
        }
    
    def _split_ratio_for_toxic_dataset(self, df, majority_label='nontoxic', majority_count=500, minority_count=20):
        parts = []
        for lab in df['label'].unique():
            if lab == majority_label:
                parts.append(df[df['label'] == lab].sample(majority_count, random_state=42))
            else:
                parts.append(df[df['label'] == lab].sample(minority_count, random_state=42))
        out = pd.concat(parts, ignore_index=True, sort=False)
        return out.sample(frac=1).reset_index(drop=True)
    
    def load_twitter_emotion_data(self, data_dir="Data/twitter_emotion"):
        """Load and prepare Twitter emotion datasets."""
        emotion_map = self.label_maps['twitter_emotion']
        
        emotion_df = pd.read_parquet(f"{data_dir}/twitter_emotion.parquet")
        emotion_df["label"] = emotion_df["label"].map(emotion_map)
        
        # Create different imbalanced datasets
        emotion_balanced = self._split_ratio_for_emotion_dataset(emotion_df, 'sadness', 200, 200)
        emotion_imbalanced_99_to_1 = self._split_ratio_for_emotion_dataset(emotion_df, 'sadness', 950, 20)
        emotion_imbalanced_49_to_1 = self._split_ratio_for_emotion_dataset(emotion_df, 'sadness', 202, 20)
        emotion_joy_majority_99 = self._split_ratio_for_emotion_dataset(emotion_df, 'joy', 950, 20)
        emotion_love_majority_99 = self._split_ratio_for_emotion_dataset(emotion_df, 'love', 950, 20)
        
        return {
            "emotion_df": emotion_balanced,
            "emotion_imbalanced_99_to_1": emotion_imbalanced_99_to_1,
            "emotion_imbalanced_49_to_1": emotion_imbalanced_49_to_1,
            "emotion_joy_majority_99": emotion_joy_majority_99,
            "emotion_love_majority_99": emotion_love_majority_99
        }
    
    def _split_ratio_for_emotion_dataset(self, df, majority_label='sadness', majority_count=200, minority_count=20):
        parts = []
        labels = df['label'].unique().tolist()
        for lab in labels:
            if lab == majority_label:
                parts.append(df[df['label'] == lab].sample(majority_count, random_state=42))
            else:
                parts.append(df[df['label'] == lab].sample(minority_count, random_state=42))
        out = pd.concat(parts, ignore_index=True, sort=False)
        return out.sample(frac=1).reset_index(drop=True)
    
    def build_prompt(self, df, text, label_map, shots_minority=0, shots_majority=0, forced_maj_label=None):
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

        # Infer majority label from df (if possible)
        maj_label = None
        try:
            if forced_maj_label is not None:
                maj_label = forced_maj_label
            else:
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

        prompt += f"Review: \"{text}\"\So now what is this label for this text, reason why.:"
        return prompt

    def normalize_label(self, label, label_map):
        # Use a cached SentenceTransformer instance (created in __init__) to avoid
        # re-loading the model from disk/network on every call.
        # Cache embeddings for each unique label_map to speed up repeated lookups.
        key = tuple(list(label_map.values()))
        if key not in self.valid_embeddings:
            # compute and cache embeddings for the canonical labels
            self.valid_embeddings[key] = self.embedding_model.encode(list(label_map.values()), convert_to_tensor=True)
            self.valid_labels[key] = list(label_map.values())

        valid_labels_emb = self.valid_embeddings[key]
        pred_emb = self.embedding_model.encode(label, convert_to_tensor=True)
        cos_scores = util.cos_sim(pred_emb, valid_labels_emb)[0]
        closest_idx = int(cos_scores.argmax().item())
        return self.valid_labels[key][closest_idx]

    def classify(self, df, label_map, shots_minority=0, shots_majority=0, batch_size=16, max_new_tokens=3, forced_maj_label=None, use_self_consistency=False, sc_num_samples=5, temp=0.7):
        
        # Convert device for pipeline
        device_arg = -1
        if self.device == 'cuda':
            device_arg = 0
        elif self.device == 'mps':
            device_arg = 'mps'
        
        pipe = pipeline("text-generation", model=self.model_name, device=device_arg)

        # Generate prompts for all rows
        prompts = [
            self.build_prompt(df, text, label_map, shots_minority=shots_minority, shots_majority=shots_majority, forced_maj_label=forced_maj_label)
            for text in df["text"]
        ]

        pred_arr = []
        start_time = time()

        if use_self_consistency:
            # Self-consistency mode: sample multiple times and aggregate
            print(f"Using self-consistency with {sc_num_samples} samples, temp={temp}")
            sc = SelfConsistency(num_samples=sc_num_samples, temperature=temp)
            
            # Ensure non-zero temperature for diversity
            sc_temp = max(temp, 0.7)
            
            for prompt in tqdm(prompts, desc="Self-consistency inference"):
                # Define generate function for self-consistency
                def generate_fn(p, **kwargs):
                    result = pipe(
                        p,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=sc_temp,
                        return_full_text=False
                    )
                    if result and len(result) > 0:
                        generated = result[0].get('generated_text', '')
                        return generated if generated else ""
                    return ""
                
                # Use self-consistency sampling and aggregation
                pred = sc.sample_and_aggregate(
                    generate_fn=generate_fn,
                    prompt=prompt,
                    valid_labels=list(label_map.values()),
                    normalize_fn=lambda x: self.normalize_label(x, label_map) if x and x.strip() else "unknown"
                )
                pred_arr.append(pred)
        else:
            # Greedy mode: batch inference
            print(f"Using greedy decoding with temp={temp}")
            
            for i in tqdm(range(0, len(prompts), batch_size), desc="Greedy inference"):
                batch = prompts[i:i + batch_size]
                results = pipe(
                    batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temp > 0),
                    temperature=temp if temp > 0 else 1.0
                )

                for prompt, res in zip(batch, results):
                    # generated_text usually contains prompt + completion
                    generated = res[0].get('generated_text', '')
                    
                    # Handle empty generation
                    if not generated or generated.strip() == "":
                        pred_arr.append("unknown")
                        continue
                    
                    completion = generated[len(prompt):].strip().lower().split()
                    if completion:
                        first_tok = completion[0]
                        if first_tok not in [lab.lower() for lab in label_map.values()]:
                            normalized_pred = self.normalize_label(first_tok, label_map)
                            pred_arr.append(normalized_pred)
                        else:
                            pred_arr.append(first_tok)
                    else:
                        pred_arr.append("unknown")

        end_time = time()
        total_time = self.clean_time(end_time - start_time)
        print(f"Total running time: {total_time}")

        return pred_arr

    
    def eval_llm(self, y_true, y_pred, label_map):
        """Evaluate LLM predictions using various metrics."""
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
    
    def infer_metadata(self, ds_name, df):
        """Infer dataset metadata from name and content."""
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
    
    def run_experiments(self, datasets_dict, dataset_name, label_map, shots_minority=0, batch_size=16, shots_majority=8, max_new_tokens=3,forced_maj_label=None, use_self_consistency=False, sc_num_samples=5, temp=0.7):
        """
        Run experiments where `shots_list` contains majority shot counts to sweep,
        and `shots_minority` is used for the other classes.
        """
        results = []

        out_dir = os.path.join("results", dataset_name)
        os.makedirs(out_dir, exist_ok=True)

        min_range = [0] if shots_minority == 0 or use_self_consistency else list(range(0, shots_minority + 1, 2))
        maj_range = [0] if shots_majority == 0 or use_self_consistency else list(range(0, shots_majority + 1, 2))

        for ds_name, df in datasets_dict.items():
            print(f"=== RUNNING DATASET {ds_name} ===")
            test_df = df.sample(frac=1).reset_index(drop=True)

            for shot_min in min_range:
                for shot_maj in maj_range:
                    print(f"    === SHOTS (majority={shot_maj}, minority={shot_min}) ===")
                    preds = self.classify(
                        test_df, label_map,
                        shots_minority=shot_min,
                        shots_majority=shot_maj,
                        batch_size=batch_size,
                        max_new_tokens=max_new_tokens,
                        forced_maj_label=forced_maj_label,
                        use_self_consistency=use_self_consistency,
                        sc_num_samples=sc_num_samples,
                        temp=temp
                    )

                    metrics = self.eval_llm(test_df['label'].tolist(), preds, label_map=label_map)

                    # Infer dataset metadata
                    ratio, maj_label = self.infer_metadata(ds_name, df)

                    row = {
                        "model": self.model_name,
                        "dataset": ds_name,
                        "shots_majority": int(shot_min),
                        "shots_minority": int(shot_maj),
                        "dataset_ratio": ratio,
                        "majority_label": maj_label,
                        **metrics
                    }
                    results.append(row)

                    # Save results incrementally
                    self._save_results(results, out_dir)

        return pd.DataFrame(results)

    
    def _save_results(self, results, out_dir):
        """Save results to CSV files with timestamp and shot metadata."""
        os.makedirs(out_dir, exist_ok=True)

        # Create aggregated DataFrame
        df_agg = pd.json_normalize(results)
        df_agg.columns = [c.replace('.', '_') for c in df_agg.columns]

        # Add timestamp
        timestamp = datetime.now().strftime("%m_%d_%Y")
        df_agg["saved_timestamp"] = timestamp

        # Save aggregated results (timestamped so old runs are kept)
        agg_name = f"few_shot_results_{self.model_name.replace('/', '_')}_{timestamp}.csv"
        agg_path = os.path.join(out_dir, agg_name)
        df_agg.to_csv(agg_path, index=False)


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate LLMs on imbalanced datasets")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", 
                       help="HuggingFace model name")
    parser.add_argument("--device", type=str, choices=['cuda', 'mps', 'cpu'], 
                       help="Device to use (auto-detect if not specified)")
    parser.add_argument("--datasets", nargs='+', choices=['ag_news', 'toxic_text', 'twitter_emotion'], 
                       default=['ag_news', 'toxic_text', 'twitter_emotion'],
                       help="Datasets to evaluate on")
    parser.add_argument(
        "--different-shots",
        action="store_true",
        help="Use different number of shots for majority and minority classes. If not set, both will be equal."
    )
    parser.add_argument(
        "--shots-minority",
        type=int,
        default=8,
        help="Number of shots for minority class (used if --different-shots is set)."
    )
    parser.add_argument(
        "--shots-majority",
        type=int,
        default=8,
        help="Number of shots for majority class (used if --different-shots is set)."
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4,
        help="Number of shots for both classes (used if --different-shots is not set)."
    )
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for inference")
    parser.add_argument("--data-dir", type=str, default="Data",
                       help="Directory containing the datasets")
    parser.add_argument("--majority-label", type=str, default="world")
    parser.add_argument("--max-tokens", type=int, default=3)
    parser.add_argument("--use-self-consistency", action='store_true',
                       help="Use self-consistency prompting (samples multiple diverse outputs and aggregates via majority vote)")
    parser.add_argument("--sc-samples", type=int, default=5,
                       help="Number of samples for self-consistency (default: 5)")
    parser.add_argument("--sc-temperature", type=float, default=0.7,
                       help="Temperature for self-consistency sampling (default: 0.7)")
    
    args = parser.parse_args()
    
    evaluator = LLMEvaluator(args.model, args.device)
    
    # Authenticate with HuggingFace
    evaluator.authenticate_hf()
    
    # Run experiments on specified datasets
    for dataset_name in args.datasets:
        print(f"\n{'='*50}")
        print(f"EVALUATING ON {dataset_name.upper()}")
        print(f"{'='*50}")
        
        # Load dataset
        if dataset_name == 'ag_news':
            datasets_dict = evaluator.load_ag_news_data(f"{args.data_dir}/ag_news")
            label_map = evaluator.label_maps['ag_news']
        elif dataset_name == 'toxic_text':
            datasets_dict = evaluator.load_toxic_text_data(f"{args.data_dir}/toxic_text")
            label_map = evaluator.label_maps['toxic_text']
        elif dataset_name == 'twitter_emotion':
            datasets_dict = evaluator.load_twitter_emotion_data(f"{args.data_dir}/twit")
            label_map = evaluator.label_maps['twitter_emotion']
        
        # Determine shot configuration and display mode
        if args.use_self_consistency:
            print(f"\n{'='*60}")
            print(f"SELF-CONSISTENCY MODE ENABLED")
            print(f"  - Samples per prediction: {args.sc_samples}")
            print(f"  - Sampling temperature: {args.sc_temperature}")
            print(f"  - Aggregation: majority vote")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"GREEDY DECODING MODE")
            print(f"{'='*60}\n")

        print(f"Using different shots → minority: {args.shots_minority}, majority: {args.shots_majority}")
        evaluator.run_experiments(
            datasets_dict, dataset_name, label_map,
            batch_size=args.batch_size,
            shots_minority=args.shots_minority,
            shots_majority=args.shots_majority,
            forced_maj_label=args.majority_label,
            use_self_consistency=args.use_self_consistency,
            sc_num_samples=args.sc_samples,
            temp=args.sc_temperature
        )
        
        print(f"Completed evaluation on {dataset_name}")
        print(f"Results saved to results/{dataset_name}/")


if __name__ == "__main__":
    main()
