
import os
import sys
import argparse
import random
import re
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import time
from typing import List, Optional

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
from openai import OpenAI

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from packages.dataset_loader import DatasetLoader
from packages.self_consistency import SelfConsistency
from packages.min_first_voting import choose_with_threshold_override, summarise_votes
from evals import infer_preference as preference

# Suppress warnings
logging.set_verbosity_info()

class LLMEvaluator:
    """Main class for evaluating LLMs on imbalanced datasets."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        *,
        backend: str = "huggingface",
        openrouter_api_key: Optional[str] = None,
        openrouter_app_url: Optional[str] = None,
        openrouter_app_title: Optional[str] = None,
        openrouter_max_tokens: Optional[int] = None,
    ):
        self.model_name = model_name
        self.backend = (backend or "huggingface").lower()
        if self.backend not in {"huggingface", "openrouter"}:
            raise ValueError(f"Unsupported backend '{backend}'. Choose 'huggingface' or 'openrouter'.")

        self.device = self._setup_device(device) if self.backend == "huggingface" else "api"
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._hf_pipeline = None
        self.openrouter_client: Optional[OpenAI] = None
        self.openrouter_default_max_tokens = int(openrouter_max_tokens) if openrouter_max_tokens else 256

        if self.backend == "openrouter":
            self._configure_openrouter_client(
                api_key=openrouter_api_key,
                referer=openrouter_app_url,
                title=openrouter_app_title,
            )
        
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
        self.dataset_loader = DatasetLoader(self.label_maps)
        
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

    def _configure_openrouter_client(self, api_key=None, referer=None, title=None):
        """Initialise OpenRouter client using OpenAI SDK."""
        load_dotenv()
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter backend selected but no API key provided. Set OPENROUTER_API_KEY or pass openrouter_api_key."
            )

        default_headers = {}
        referer = referer or os.getenv("OPENROUTER_APP_URL")
        title = title or os.getenv("OPENROUTER_APP_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title

        self.openrouter_client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=default_headers or None,
        )
    
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
        return self.dataset_loader.load_ag_news_data(data_dir)

    def load_toxic_text_data(self, data_dir="Data/toxic_text"):
        return self.dataset_loader.load_toxic_text_data(data_dir)

    def load_twitter_emotion_data(self, data_dir="Data/twitter_emotion"):
        return self.dataset_loader.load_twitter_emotion_data(data_dir)
    
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

        prompt += f"Review: \"{text}\"\nSo now what is the label for this text? Provide the label name only."
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

    def _build_text_generation_pipeline(self):
        if self.backend != "huggingface":
            raise RuntimeError("Text-generation pipeline is only available for the HuggingFace backend.")

        if self._hf_pipeline is None:
            device_arg = -1
            if self.device == 'cuda':
                device_arg = 0
            elif self.device == 'mps':
                device_arg = 'mps'

            self._hf_pipeline = pipeline("text-generation", model=self.model_name, device=device_arg)
            tokenizer = getattr(self._hf_pipeline, "tokenizer", None)
            if tokenizer is not None:
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if pad_token_id is None and eos_token_id is not None:
                    tokenizer.pad_token_id = eos_token_id

        return self._hf_pipeline

    @staticmethod
    def _clean_generated_text(output) -> str:
        """Normalise pipeline outputs into plain text."""
        if output is None:
            return ""
        if isinstance(output, dict):
            text = output.get("generated_text") or output.get("text") or ""
            return str(text).strip()
        return str(output).strip()

    @staticmethod
    def _normalize_openrouter_message(message) -> str:
        """Convert OpenRouter ChatCompletionMessage into plain text, falling back to reasoning when needed."""

        def _coalesce_content(value) -> List[str]:
            if not value:
                return []
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                parts: List[str] = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text_val = item.get("text")
                        if text_val:
                            parts.append(str(text_val))
                    else:
                        text_val = getattr(item, "text", None)
                        if text_val:
                            parts.append(str(text_val))
                return parts
            return [str(value)]

        if message is None:
            return ""

        text_fragments: List[str] = []

        content = getattr(message, "content", None)
        text_fragments.extend(_coalesce_content(content))

        if not text_fragments:
            reasoning = getattr(message, "reasoning", None)
            text_fragments.extend(_coalesce_content(reasoning))

        if not text_fragments:
            reasoning_details = getattr(message, "reasoning_details", None)
            text_fragments.extend(_coalesce_content(reasoning_details))

        if not text_fragments and isinstance(message, dict):
            text_fragments.extend(_coalesce_content(message.get("content")))
            if not text_fragments and "reasoning" in message:
                text_fragments.extend(_coalesce_content(message.get("reasoning")))

        return " ".join(fragment.strip() for fragment in text_fragments if fragment).strip()

    def _generate_openrouter(
        self,
        prompt: str,
        *,
    max_new_tokens: int,
        temperature: float,
        top_p: Optional[float],
        num_return_sequences: int,
        do_sample: bool,
    ) -> List[str]:
        if self.openrouter_client is None:
            self._configure_openrouter_client()

        effective_temperature = float(temperature) if do_sample else 0.0
        effective_temperature = max(effective_temperature, 0.0)

        if max_new_tokens is None or max_new_tokens <= 0:
            max_new_tokens = self.openrouter_default_max_tokens
        else:
            max_new_tokens = max(max_new_tokens, self.openrouter_default_max_tokens)

        params = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": effective_temperature,
            "n": max(1, num_return_sequences),
        }
        if top_p is not None and do_sample:
            params["top_p"] = top_p

        try:
            response = self.openrouter_client.chat.completions.create(**params)
        except Exception as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        outputs: List[str] = []
        for choice in getattr(response, "choices", []) or []:
            message = getattr(choice, "message", None)
            outputs.append(self._normalize_openrouter_message(message))

        if not outputs:
            outputs.append("")
        return outputs

    def _generate_prompt(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        do_sample: bool = False,
    ) -> List[str]:
        if self.backend == "huggingface":
            pipe = self._build_text_generation_pipeline()
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
                "num_return_sequences": max(1, num_return_sequences),
            }
            if do_sample:
                kwargs["do_sample"] = True
                kwargs["temperature"] = max(float(temperature), 1e-6)
                if top_p is not None:
                    kwargs["top_p"] = top_p
            else:
                kwargs["do_sample"] = False

            outputs = pipe(prompt, **kwargs)
            sequences = outputs if isinstance(outputs, list) else [outputs]
            return [self._clean_generated_text(seq) for seq in sequences]

        return self._generate_openrouter(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
        )

    def _generate_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        batch_size: int = 16,
    ) -> List[str]:
        if not prompts:
            return []

        if self.backend == "huggingface":
            pipe = self._build_text_generation_pipeline()
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
                "batch_size": batch_size,
            }
            if do_sample:
                kwargs["do_sample"] = True
                kwargs["temperature"] = max(float(temperature), 1e-6)
            else:
                kwargs["do_sample"] = False

            outputs = pipe(prompts, **kwargs)
            completions: List[str] = []
            for out in outputs:
                if isinstance(out, list) and out:
                    completions.append(self._clean_generated_text(out[0]))
                else:
                    completions.append(self._clean_generated_text(out))
            return completions

        # OpenRouter (no batch endpoint): iterate sequentially
        completions = []
        for prompt in prompts:
            texts = self._generate_prompt(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=do_sample,
            )
            completions.append(texts[0] if texts else "")
        return completions

    @staticmethod
    def _clean_candidate_token(token: str) -> str:
        return token.strip().strip(".?,;:!\"'()[]{}").lower()


    def _extract_label_from_completion(
        self,
        generated_text: str,
        label_map,
        valid_labels_lower: List[str],
    ) -> str:
        text = (generated_text or "").strip()
        if not text:
            return "unknown"

        valid_set = set(valid_labels_lower)
        tokens = re.split(r"\s+", text)

        skip_indices = set()
        idx = 0
        total_tokens = len(tokens)
        while idx < total_tokens:
            token_clean = self._clean_candidate_token(tokens[idx])
            if token_clean == "categories":
                lookahead = idx + 1
                collected = 0
                while lookahead < total_tokens and collected < len(valid_labels_lower):
                    la_clean = self._clean_candidate_token(tokens[lookahead])
                    if la_clean in valid_set:
                        skip_indices.add(lookahead)
                        collected += 1
                        lookahead += 1
                    elif la_clean:
                        break
                    else:
                        lookahead += 1
                idx = lookahead
                continue
            idx += 1

        for idx in range(total_tokens - 1, -1, -1):
            tok = tokens[idx]
            cleaned = self._clean_candidate_token(tok)
            if cleaned in valid_set:
                if idx in skip_indices:
                    continue
                return cleaned

        fallback_token = self._clean_candidate_token(tokens[0]) if tokens else self._clean_candidate_token(text)
        if fallback_token:
            normalized = self.normalize_label(fallback_token, label_map)
            if normalized:
                return normalized.lower()

        return "unknown"

    def _prepare_pref_dataset(self, df: pd.DataFrame, label_map):
        labels = list(dict.fromkeys(label_map.values()))
        records = []
        for idx, row in enumerate(df.itertuples()):
            text = getattr(row, 'text', '')
            label = getattr(row, 'label', None)
            records.append({
                "id": idx,
                "text": text,
                "label": label,
            })

        counts = Counter(record["label"] for record in records if record.get("label") in labels)
        total = sum(counts.values())
        if total == 0:
            prior = {label: 1.0 / max(len(labels), 1) for label in labels}
        else:
            prior = {label: counts.get(label, 0) / total for label in labels}

        return preference.Dataset(records=records, labels=labels, dataset_prior=prior)

    def _detect_preferred_labels(
        self,
        df: pd.DataFrame,
        label_map,
        shots_minority: int,
        shots_majority: int,
        max_new_tokens: int,
        temp: float,
    ):
        print("\n[INFO] Detecting preferred labels from balanced sample...")

        value_counts = df['label'].value_counts()
        if value_counts.empty:
            return []

        sample_size = min(10, value_counts.min())
        balanced_dfs = []
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            sampled = subset.sample(n=sample_size, replace=len(subset) < sample_size, random_state=42)
            balanced_dfs.append(sampled)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"[INFO] Using balanced sample: {len(balanced_df)} rows ({sample_size} per class)")

        preds = self.classify(
            balanced_df,
            label_map,
            shots_minority=shots_minority,
            shots_majority=shots_majority,
            batch_size=16,
            max_new_tokens=max_new_tokens,
            forced_maj_label=None,
            use_self_consistency=False,
            sc_num_samples=1,
            temp=temp,
        )

        metrics = self.eval_llm(balanced_df['label'].tolist(), preds, label_map=label_map)

        preference_scores = {}
        precision_per_class = metrics.get("precision_per_class", {})
        recall_per_class = metrics.get("recall_per_class", {})

        for label in label_map.values():
            label_lower = label.lower()
            prec = precision_per_class.get(label_lower, 0.0)
            rec = recall_per_class.get(label_lower, 0.0)
            preference_scores[label] = (rec / prec) if prec > 0 else 0.0

        print(f"[INFO] Preference scores (recall/precision): {preference_scores}")
        preferred_labels = [label for label, score in preference_scores.items() if score >= 1.0]

        if preferred_labels:
            for label in preferred_labels:
                print(f"[INFO] Detected preferred label: '{label}' (score={preference_scores[label]:.3f})")
        else:
            top_label = max(preference_scores.items(), key=lambda item: item[1])[0]
            print(f"[INFO] No label exceeded threshold; defaulting to '{top_label}' as preferred candidate")
            preferred_labels = [top_label]

        return preferred_labels

    def classify(self, df, label_map, shots_minority=0, shots_majority=0, batch_size=16, max_new_tokens=3, forced_maj_label=None, use_self_consistency=False, sc_num_samples=5, temp=0.7):
        # Generate prompts for all rows
        prompts = [
            self.build_prompt(df, text, label_map, shots_minority=shots_minority, shots_majority=shots_majority, forced_maj_label=forced_maj_label)
            for text in df["text"]
        ]

        pred_arr = []
        start_time = time()
        valid_labels_lower = [lab.lower() for lab in label_map.values()]

        if use_self_consistency:
            # Self-consistency mode: sample multiple times and aggregate
            print(f"Using self-consistency with {sc_num_samples} samples, temp={temp}")
            sc = SelfConsistency(num_samples=sc_num_samples, temperature=temp)
            
            # Ensure non-zero temperature for diversity
            sc_temp = max(temp, 0.7)
            
            for prompt in tqdm(prompts, desc="Self-consistency inference"):
                # Define generate function for self-consistency
                def generate_fn(p, **kwargs):
                    outputs = self._generate_prompt(
                        p,
                        max_new_tokens=max_new_tokens,
                        temperature=sc_temp,
                        num_return_sequences=1,
                        do_sample=True,
                    )
                    return outputs[0] if outputs else ""
                
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
                results = self._generate_batch(
                    batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=(temp > 0),
                    batch_size=batch_size,
                )

                for generated in results:
                    label = self._extract_label_from_completion(generated, label_map, valid_labels_lower)
                    pred_arr.append(label)

        end_time = time()
        total_time = self.clean_time(end_time - start_time)
        print(f"Total running time: {total_time}")

        return pred_arr

    def classify_minority_first(
        self,
        df,
        label_map,
        *,
        shots_minority=0,
        shots_majority=0,
        max_new_tokens=32,
        temp=0.7,
        top_p=0.9,
        samples_per_example=5,
        threshold=20,
        seed=0,
        preferred_label=None,
        include_text=True,
    ):
        preference.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        dataset = self._prepare_pref_dataset(df, label_map)
        if not dataset.records:
            raise ValueError("Dataset is empty; cannot run minority-first voting")

        if preferred_label is not None:
            preferred_candidates = [preferred_label]
        else:
            preferred_candidates = self._detect_preferred_labels(
                df,
                label_map,
                shots_minority=shots_minority,
                shots_majority=shots_majority,
                max_new_tokens=max_new_tokens,
                temp=temp,
            )

        derived_preferred = preferred_candidates[0] if preferred_candidates else None
        print(f"[INFO] Applying threshold-based voting (threshold={threshold}, preferred={derived_preferred})")

        effective_temp = temp if temp and temp > 0 else 0.7
        effective_top_p = top_p if top_p and top_p > 0 else 0.9

        predictions = []
        vote_rows = []
        valid_labels_lower = [lab.lower() for lab in label_map.values()]

        for idx, record in enumerate(tqdm(dataset.records, desc="Minority-first voting")):
            text = record.get("text", "")
            prompt = self.build_prompt(
                df,
                text,
                label_map,
                shots_minority=shots_minority,
                shots_majority=shots_majority,
                forced_maj_label=derived_preferred,
            ) + f"\nText: {text}\nCategory:"

            outputs = self._generate_prompt(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=effective_temp,
                top_p=effective_top_p,
                num_return_sequences=max(1, samples_per_example),
                do_sample=True,
            )

            votes = []
            for out in outputs:
                label = self._extract_label_from_completion(out, label_map, valid_labels_lower)
                votes.append(label)

            final_label, decision_mode = choose_with_threshold_override(votes, derived_preferred, threshold)
            if final_label is None:
                final_label = "unknown"

            majority_label = Counter(votes).most_common(1)[0][0] if votes else None

            record_row = {
                "id": record.get("id", idx),
                "samples": votes,
                "majority": majority_label,
                "final": final_label,
                "threshold_applied": decision_mode == "threshold_override",
                "decision_mode": decision_mode,
            }
            if include_text:
                record_row["text"] = text
            if record.get("label") is not None:
                record_row["label"] = record.get("label")

            vote_rows.append(record_row)
            predictions.append(final_label)

        summary = summarise_votes(vote_rows, dataset.labels, derived_preferred)
        return predictions, vote_rows, summary

    
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
    
    def run_experiments(
        self,
        datasets_dict,
        dataset_name,
        label_map,
        shots_minority=0,
        batch_size=16,
        shots_majority=8,
        max_new_tokens=3,
        forced_maj_label=None,
        use_self_consistency=False,
        sc_sample_counts=None,
        temp=0.7,
        minority_first=False,
        mf_samples=None,
        mf_threshold=20,
        mf_top_p=0.9,
        seed=0,
        mf_preferred_label=None,
    ):
        """
        Run experiments where `shots_list` contains majority shot counts to sweep,
        and `shots_minority` is used for the other classes.
        """
        results = []

        out_dir = os.path.join("results", dataset_name)
        os.makedirs(out_dir, exist_ok=True)

        sc_sample_counts = sc_sample_counts or []

        if minority_first:
            default_samples = mf_samples if mf_samples and mf_samples > 0 else (sc_sample_counts[0] if sc_sample_counts else 5)
            for ds_name, df in datasets_dict.items():
                print(f"=== RUNNING DATASET {ds_name} (minority-first) ===")
                preds, vote_rows, summary = self.classify_minority_first(
                    df,
                    label_map,
                    shots_minority=shots_minority,
                    shots_majority=shots_majority,
                    max_new_tokens=max_new_tokens,
                    temp=temp,
                    top_p=mf_top_p,
                    samples_per_example=default_samples,
                    threshold=mf_threshold,
                    seed=seed,
                    preferred_label=mf_preferred_label,
                )

                metrics = self.eval_llm(df['label'].tolist(), preds, label_map=label_map)
                ratio, maj_label = self.infer_metadata(ds_name, df)

                row = {
                    "model": self.model_name,
                    "backend": self.backend,
                    "dataset": ds_name,
                    "shots_minority": int(shots_minority),
                    "shots_majority": int(shots_majority),
                    "self_consistency_samples": None,
                    "minority_first": True,
                    "mf_samples": int(default_samples),
                    "mf_threshold": int(mf_threshold),
                    "mf_threshold_fraction": summary.get("threshold_fraction"),
                    "preferred_label": summary.get("preferred_label"),
                    "dataset_ratio": ratio,
                    "majority_label": maj_label,
                    "mf_final_counts": summary.get("final_counts"),
                    "mf_majority_counts": summary.get("majority_counts"),
                    **metrics,
                }
                results.append(row)
                self._save_results(results, out_dir)

            return pd.DataFrame(results)

        if use_self_consistency:
            min_range = [shots_minority]
            maj_range = [shots_majority]
            sample_counts = sc_sample_counts or [5]
        else:
            min_range = [0] if shots_minority == 0 else list(range(0, shots_minority + 1, 2))
            maj_range = [0] if shots_majority == 0 else list(range(0, shots_majority + 1, 2))
            sample_counts = [None]

        for ds_name, df in datasets_dict.items():
            print(f"=== RUNNING DATASET {ds_name} ===")
            test_df = df.sample(frac=1).reset_index(drop=True)

            if use_self_consistency:
                shot_min = min_range[0] if min_range else 0
                shot_maj = maj_range[0] if maj_range else 0
                for sc_samples in sample_counts:
                    print(f"    === SELF-CONSISTENCY SAMPLES={sc_samples} ===")
                    preds = self.classify(
                        test_df,
                        label_map,
                        shots_minority=shot_min,
                        shots_majority=shot_maj,
                        batch_size=batch_size,
                        max_new_tokens=max_new_tokens,
                        forced_maj_label=forced_maj_label,
                        use_self_consistency=True,
                        sc_num_samples=sc_samples,
                        temp=temp,
                    )

                    metrics = self.eval_llm(test_df['label'].tolist(), preds, label_map=label_map)

                    ratio, maj_label = self.infer_metadata(ds_name, df)

                    row = {
                        "model": self.model_name,
                        "backend": self.backend,
                        "dataset": ds_name,
                        "shots_minority": int(shot_min) if shot_min is not None else None,
                        "shots_majority": int(shot_maj) if shot_maj is not None else None,
                        "self_consistency_samples": int(sc_samples),
                        "dataset_ratio": ratio,
                        "majority_label": maj_label,
                        **metrics,
                    }
                    results.append(row)
                    self._save_results(results, out_dir)
            else:
                for shot_min in min_range:
                    for shot_maj in maj_range:
                        print(f"    === SHOTS (majority={shot_maj}, minority={shot_min}) ===")
                        preds = self.classify(
                            test_df,
                            label_map,
                            shots_minority=shot_min,
                            shots_majority=shot_maj,
                            batch_size=batch_size,
                            max_new_tokens=max_new_tokens,
                            forced_maj_label=forced_maj_label,
                            use_self_consistency=False,
                            sc_num_samples=sample_counts[0],
                            temp=temp,
                        )

                        metrics = self.eval_llm(test_df['label'].tolist(), preds, label_map=label_map)

                        ratio, maj_label = self.infer_metadata(ds_name, df)

                        row = {
                            "model": self.model_name,
                            "backend": self.backend,
                            "dataset": ds_name,
                            "shots_minority": int(shot_min),
                            "shots_majority": int(shot_maj),
                            "self_consistency_samples": None,
                            "dataset_ratio": ratio,
                            "majority_label": maj_label,
                            **metrics,
                        }
                        results.append(row)
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

        last_row = results[-1] if results else {}
        if last_row.get("minority_first"):
            mode_prefix = "MF"
        else:
            use_sc = bool(last_row.get("self_consistency_samples"))
            mode_prefix = "SC" if use_sc else "ICL"

        # Save aggregated results (timestamped so old runs are kept)
        agg_name = f"{mode_prefix}_results_{self.model_name.replace('/', '_')}_{timestamp}.csv"
        agg_path = os.path.join(out_dir, agg_name)
        df_agg.to_csv(agg_path, index=False)


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate LLMs on imbalanced datasets")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", 
                       help="Model identifier (HuggingFace repo or OpenRouter model name)")
    parser.add_argument("--backend", type=str, choices=['huggingface', 'openrouter'], default='huggingface',
                       help="Backend to use for inference (default: huggingface)")
    parser.add_argument("--openrouter-max-tokens", type=int,
                       help="Override default max tokens for OpenRouter completions (default: 256)")
    parser.add_argument("--device", type=str, choices=['cuda', 'mps', 'cpu'], 
                       help="Device to use (auto-detect if not specified)")
    parser.add_argument("--datasets", nargs='+', choices=['ag_news', 'toxic_text', 'twitter_emotion', 'all'], 
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
    parser.add_argument(
        "--sc-samples",
        type=int,
        nargs='+',
        default=[5],
        help="Number of samples for self-consistency. Provide one or more integers (default: 5)."
    )
    parser.add_argument("--sc-temperature", type=float, default=0.7,
                       help="Temperature for self-consistency sampling (default: 0.7)")
    parser.add_argument("--save-all-results", action='store_true',
                       help="Save combined results across all evaluated datasets to results/all_datasets")
    parser.add_argument("--all-output-dir", type=str, default="results/all_datasets",
                       help="Directory for combined results when --save-all-results or --datasets all is used")
    parser.add_argument("--minority-first", action='store_true',
                       help="Enable threshold-based minority-first voting with multiple samples per example")
    parser.add_argument("--mf-samples", type=int, default=0,
                       help="Samples per example for minority-first voting (default: auto)")
    parser.add_argument("--mf-threshold", type=int, default=20,
                       help="Threshold for suppressing preferred label during minority-first voting")
    parser.add_argument("--mf-top-p", type=float, default=0.9,
                       help="Top-p value to use when sampling for minority-first voting")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling-based evaluation modes")
    parser.add_argument("--mf-preferred-label", type=str,
                       help="Explicit preferred label to mitigate during minority-first voting")
    
    args = parser.parse_args()

    
    evaluator = LLMEvaluator(
        args.model,
        args.device,
        backend=args.backend,
        openrouter_max_tokens=args.openrouter_max_tokens,
    )

    print(f"Selected backend: {evaluator.backend}")

    # Authenticate with HuggingFace when required
    if evaluator.backend == "huggingface":
        evaluator.authenticate_hf()
    if args.minority_first and args.use_self_consistency:
        parser.error("Minority-first voting cannot be combined with self-consistency mode.")
    
    sc_sample_counts = (
        list(args.sc_samples)
        if isinstance(args.sc_samples, (list, tuple))
        else [int(args.sc_samples)]
    )

    # Run experiments on specified datasets
    dataset_args = args.datasets
    if 'all' in dataset_args:
        dataset_list = ['ag_news', 'toxic_text', 'twitter_emotion']
    else:
        dataset_list = dataset_args

    save_combined = args.save_all_results or ('all' in dataset_args)
    combined_results = []

    for dataset_name in dataset_list:
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
            print(f"  - Samples per prediction: {', '.join(str(x) for x in sc_sample_counts)}")
            print(f"  - Sampling temperature: {args.sc_temperature}")
            print(f"  - Aggregation: majority vote")
            print(f"{'='*60}\n")
            print(f"Few-shot context → minority: {args.shots_minority}, majority: {args.shots_majority}")
        elif args.minority_first:
            print(f"\n{'='*60}")
            print(f"MINORITY-FIRST VOTING ENABLED")
            print(f"  - Samples per example: {args.mf_samples or 'auto'}")
            print(f"  - Threshold: {args.mf_threshold}")
            print(f"  - Top-p: {args.mf_top_p}")
            if args.mf_preferred_label:
                print(f"  - Preferred label override: {args.mf_preferred_label}")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"GREEDY DECODING MODE")
            print(f"{'='*60}\n")
            print(f"Using different shots → minority: {args.shots_minority}, majority: {args.shots_majority}")
        
        df_results = evaluator.run_experiments(
            datasets_dict, dataset_name, label_map,
            batch_size=args.batch_size,
            shots_minority=args.shots_minority,
            shots_majority=args.shots_majority,
            forced_maj_label=args.majority_label,
            max_new_tokens=args.max_tokens,
            use_self_consistency=args.use_self_consistency,
            sc_sample_counts=sc_sample_counts,
            temp=args.sc_temperature,
            minority_first=args.minority_first,
            mf_samples=args.mf_samples,
            mf_threshold=args.mf_threshold,
            mf_top_p=args.mf_top_p,
            seed=args.seed,
            mf_preferred_label=args.mf_preferred_label,
        )

        if save_combined and df_results is not None and not df_results.empty:
            combined_results.append(df_results)
        
        print(f"Completed evaluation on {dataset_name}")
        print(f"Results saved to results/{dataset_name}/")



if __name__ == "__main__":
    main()
