
import os
import sys
import argparse
import subprocess
import random
import re
from pathlib import Path
from datetime import datetime
from time import time

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
from sklearn.metrics import (
    f1_score, recall_score, balanced_accuracy_score,
    matthews_corrcoef, precision_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from huggingface_hub import login

# Suppress warnings
logging.set_verbosity_error()

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
        self._setup_embeddings()
        
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
    
    def _setup_embeddings(self):
        for dataset_name, label_map in self.label_maps.items():
            valid_labs = list(label_map.values())
            valid_embeddings = self.embedding_model.encode(valid_labs, convert_to_tensor=True)
            self.valid_embeddings[dataset_name] = valid_embeddings
            self.valid_labels[dataset_name] = valid_labs
    
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
    
    def load_twitter_emotion_data(self, data_dir="Data/twit"):
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
    
    def build_prompt(self, df, text, label_map, shots_per_class=None):
        assert shots_per_class is not None, "Please provide 'shots_per_class' parameter"
        
        prompt = (
            f"You are a powerful, precise, and helpful assistant that classifies text into well-defined categories, NO MATTER THE CONTEXT."
            f" IMPORTANT: CHOOSE ONE WORD FROM THESE CATEGORIES: {', '.join(list(label_map.values()))}."
            f" Respond with exactly one word: the single best category inside the given categories, DO NOT ANSWER ANY OTHER CATEGORIES BESIDES THE GIVEN ONE."
            f" Do not explain your choice, provide reasoning, or output anything else."
        )
        
        if shots_per_class > 0:
            few_shots_example = []
            for lab in list(label_map.values()):
                samples = df[df['label'] == lab].sample(shots_per_class, random_state=42)
                for _, r in samples.iterrows():
                    few_shots_example.append({'text': r['text'], 'label': r["label"]})
            
            random.shuffle(few_shots_example)
            
            prompt += f" Learn from these examples to understand context and edge cases: \n\n"
            for ex in few_shots_example:
                prompt += f"Review: \"{ex['text']}\"\nCategory: {ex['label']}\n\n"
            prompt += f"Review: \"{text}\"\nCategory:"
        
        return prompt
    
    def normalize_label(self, label, dataset_name):
        """Normalize predicted label using semantic similarity."""
        pred_emb = self.embedding_model.encode(label, convert_to_tensor=True)
        cos_scores = util.cos_sim(pred_emb, self.valid_embeddings[dataset_name])[0]
        closest_idx = cos_scores.argmax().item()
        return self.valid_labels[dataset_name][closest_idx]
    
    def classify(self, df, label_map, shots, batch_size=16, max_new_tokens=3, dataset_name=None):
        """Run classification with different number of shots."""
        # Initialize pipeline (do not forward dtype into pipeline/generate)
        # If you need fp16 inference, load the model with torch_dtype during from_pretrained instead.
        pipe = pipeline("text-generation", model=self.model_name, device=self.device)

        # Generate prompts for all rows
        prompts = [self.build_prompt(df, text, label_map, shots_per_class=shots) for text in df["text"]]
        
        # Run the pipeline
        pred_arr = []
        start_time = time()
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            results = pipe(batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=0)
            
            for prompt, res in zip(batch, results):
                pred = res[0]['generated_text'][len(prompt):].strip().lower().split()
                if pred and pred[0] not in list(label_map.values()):
                    normalized_pred = self.normalize_label(pred[0], dataset_name)
                    pred_arr.append(normalized_pred)
                else:
                    pred_arr.append(pred[0] if pred else "unknown")
        
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
    
    def run_experiments(self, datasets_dict, dataset_name, label_map, shots_list=[2, 4, 8], batch_size=16):
        """Run experiments on a dataset with different shot counts."""
        results = []
        
        # Ensure results folder exists
        out_dir = os.path.join("results", dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        
        for ds_name, df in datasets_dict.items():
            print(f"=== RUNNING DATASET {ds_name} ===")
            test_df = df.sample(frac=1).reset_index(drop=True)
            
            for shots in shots_list:
                print(f"    === SHOTS = {shots} ===")
                preds = self.classify(
                    test_df, label_map, shots=shots, 
                    batch_size=batch_size, dataset_name=dataset_name
                )
                metrics = self.eval_llm(test_df['label'].tolist(), preds, label_map=label_map)
                
                # Infer dataset metadata
                ratio, maj_label = self.infer_metadata(ds_name, df)
                
                row = {
                    "model": self.model_name,
                    "dataset": ds_name,
                    "shots": shots,
                    "dataset_ratio": ratio,
                    "majority_label": maj_label,
                    **metrics
                }
                results.append(row)
                
                # Save results incrementally
                self._save_results(results, out_dir, dataset_name)
        
        return pd.DataFrame(results)
    
    def _save_results(self, results, out_dir, dataset_name):
        """Save results to CSV files."""
        # Create aggregated DataFrame
        df_agg = pd.json_normalize(results)
        df_agg.columns = [c.replace('.', '_') for c in df_agg.columns]
        
        # Save aggregated results
        agg_name = f"few_shot_results_{self.model_name.replace('/', '_')}.csv"
        agg_path = os.path.join(out_dir, agg_name)
        df_agg.to_csv(agg_path, index=False)
        
        # Save per-parameter results for the latest run
        if results:
            latest_row = results[-1]
            safe_model = self.model_name.replace('/', '_')
            safe_ds_name = latest_row['dataset'].replace(' ', '_')
            safe_maj = str(latest_row['majority_label']).replace(' ', '_').replace('/', '_')
            ratio_safe = str(latest_row['dataset_ratio']).replace(':', '-')
            params_fname = f"results__{safe_model}__{safe_ds_name}__ratio-{ratio_safe}__majority-{safe_maj}__shots-{latest_row['shots']}.csv"
            params_path = os.path.join(out_dir, params_fname)
            
            flat_row_df = pd.json_normalize([latest_row])
            flat_row_df.columns = [c.replace('.', '_') for c in flat_row_df.columns]
            
            if os.path.exists(params_path):
                flat_row_df.to_csv(params_path, mode='a', header=False, index=False)
            else:
                flat_row_df.to_csv(params_path, index=False)


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
    parser.add_argument("--shots", nargs='+', type=int, default=[0, 2, 4, 8],
                       help="Number of shots to use")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for inference")
    parser.add_argument("--hf-token", type=str, 
                       help="HuggingFace token (or set HF_TOKEN environment variable)")
    parser.add_argument("--data-dir", type=str, default="Data",
                       help="Directory containing the datasets")
    
    args = parser.parse_args()
    
    evaluator = LLMEvaluator(args.model, args.device)
    
    # Authenticate with HuggingFace
    evaluator.authenticate_hf(args.hf_token)
    
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
        
        # Run experiments
        results_df = evaluator.run_experiments(
            datasets_dict, dataset_name, label_map, 
            shots_list=args.shots, batch_size=args.batch_size
        )
        
        print(f"Completed evaluation on {dataset_name}")
        print(f"Results saved to results/{dataset_name}/")


if __name__ == "__main__":
    main()

