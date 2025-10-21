import os
import re
import json
import argparse
import random
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

from dataset_loader import DatasetLoader
from self_consistency import SelfConsistency


class PromptCache:
    """Minimal disk cache for OpenAI completions."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def key_for(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def load(self, key: str) -> Optional[Dict[str, object]]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, key: str, data: Dict[str, object]) -> None:
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True)


LABEL_MAPS: Dict[str, Dict[int, str]] = {
    "ag_news": {0: "world", 1: "sports", 2: "business", 3: "sci/tech"},
    "toxic_text": {0: "nontoxic", 1: "toxic"},
    "twitter_emotion": {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
    },
}


class OpenAIEvaluator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 16,
        base_temperature: float = 0.0,
        sc_temperature: float = 0.7,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.base_temperature = base_temperature
        self.sc_temperature = sc_temperature
        self.client = OpenAI()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.valid_embeddings: Dict[tuple, np.ndarray] = {}
        self.valid_labels: Dict[tuple, List[str]] = {}
        self.cache: Optional[PromptCache] = None
        if use_cache and cache_dir:
            self.cache = PromptCache(cache_dir)

    def build_prompt(
        self,
        df: pd.DataFrame,
        text: str,
        label_map: Dict[int, str],
        shots_minority: int,
        shots_majority: int,
        forced_maj_label: Optional[str] = None,
    ) -> str:
        labels = list(label_map.values())
        prompt = (
            "You are a precise assistant that classifies text into strict categories. "
            f"Choose exactly one label from: {', '.join(labels)}. "
            "Respond with a single word — the chosen label — and nothing else."
        )

        maj_label = None
        try:
            if forced_maj_label:
                maj_label = forced_maj_label
            else:
                counts = df["label"].value_counts()
                if len(counts) > 0:
                    maj_label = counts.idxmax()
        except Exception:
            maj_label = None

        few_shots: List[Dict[str, str]] = []
        for lab in labels:
            n_shots = shots_majority if maj_label and lab == maj_label else shots_minority
            if n_shots <= 0:
                continue
            candidates = df[df["label"] == lab]
            k = min(n_shots, len(candidates))
            if k <= 0:
                continue
            samples = candidates.sample(k, random_state=42)
            for _, row in samples.iterrows():
                few_shots.append({"text": row["text"], "label": row["label"]})

        if few_shots:
            random.shuffle(few_shots)
            prompt += "\n\nLearn from these labeled examples:\n"
            for example in few_shots:
                prompt += f"\nText: \"{example['text']}\"\nLabel: {example['label']}\n"

        prompt += f"\nNow classify the following text:\nText: \"{text}\"\nLabel:"
        return prompt

    def normalize_label(self, label: str, label_map: Dict[int, str]) -> str:
        key = tuple(label_map.values())
        if key not in self.valid_embeddings:
            embeddings = self.embedding_model.encode(list(label_map.values()), convert_to_tensor=True)
            self.valid_embeddings[key] = embeddings
            self.valid_labels[key] = list(label_map.values())

        valid_emb = self.valid_embeddings[key]
        pred_emb = self.embedding_model.encode(label, convert_to_tensor=True)
        cos_scores = util.cos_sim(pred_emb, valid_emb)[0]
        closest_idx = int(cos_scores.argmax().item())
        return self.valid_labels[key][closest_idx]

    def _call_openai(self, prompt: str, temperature: float, num_samples: int = 1) -> List[str]:
        messages = [
            {"role": "system", "content": "You are a concise classification assistant."},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=0.9,
                    n=num_samples,
                    max_tokens=self.max_new_tokens,
                )
                outputs: List[str] = []
                for choice in getattr(response, "choices", []):
                    message = getattr(choice, "message", None)
                    if message is None:
                        continue
                    content = getattr(message, "content", None)
                    if content:
                        outputs.append(content.strip())
                if len(outputs) < num_samples:
                    outputs.extend(["" for _ in range(num_samples - len(outputs))])
                return outputs[:num_samples]
            except Exception as exc:
                wait = (attempt + 1) * 2
                print(f"OpenAI API error: {exc}. Retrying in {wait}s...")
                time.sleep(wait)

        return ["" for _ in range(num_samples)]

    def _get_samples(self, prompt: str, temperature: float, num_samples: int) -> List[str]:
        if num_samples <= 0:
            return []

        cached_data: Optional[Dict[str, object]] = None
        cached_samples: List[str] = []
        cache_key: Optional[str] = None

        if self.cache is not None:
            cache_key = self.cache.key_for(self.model_name, prompt, temperature, self.max_new_tokens)
            cached_data = self.cache.load(cache_key)
            if cached_data is not None:
                cached_samples = list(cached_data.get("completions", []))

        samples = list(cached_samples[:num_samples])
        needed = num_samples - len(samples)

        if needed > 0:
            if cached_data is None:
                cached_data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": self.max_new_tokens,
                    "completions": list(cached_samples),
                }

            new_samples = self._call_openai(prompt, temperature, needed)
            for sample in new_samples:
                samples.append(sample)
                cached_samples.append(sample)

            cached_data["completions"] = cached_samples
            cached_data["updated_at"] = datetime.utcnow().isoformat()

            if self.cache is not None and cache_key is not None:
                self.cache.save(cache_key, cached_data)

        return samples[:num_samples]

    def _extract_prediction(self, generated: str, label_map: Dict[int, str]) -> str:
        if not generated:
            return "unknown"
        tokens = generated.strip().split()
        labels = list(label_map.values())
        for token in tokens:
            cleaned = token.strip('.,!?;:"\'').lower()
            for label in labels:
                if cleaned == label.lower():
                    return label
        return self.normalize_label(tokens[0], label_map)

    def classify(
        self,
        df: pd.DataFrame,
        label_map: Dict[int, str],
        shots_minority: int,
        shots_majority: int,
        forced_maj_label: Optional[str],
        use_self_consistency: bool,
        sc_samples: int,
    ) -> List[str]:
        prompts = [
            self.build_prompt(
                df,
                text,
                label_map,
                shots_minority=shots_minority,
                shots_majority=shots_majority,
                forced_maj_label=forced_maj_label,
            )
            for text in df["text"].tolist()
        ]

        predictions: List[str] = []

        if use_self_consistency:
            print(f"Using self-consistency with {sc_samples} samples and temperature {self.sc_temperature}")
            sc = SelfConsistency(num_samples=sc_samples, temperature=self.sc_temperature)
            valid_labels = list(label_map.values())

            for prompt in tqdm(prompts, desc="Self-consistency inference"):
                samples = self._get_samples(prompt, self.sc_temperature, sc_samples)
                iterator = iter(samples)

                def generate_fn(_: str, **kwargs: object) -> str:
                    return next(iterator)

                pred = sc.sample_and_aggregate(
                    generate_fn=generate_fn,
                    prompt=prompt,
                    valid_labels=valid_labels,
                    normalize_fn=lambda x: self.normalize_label(x, label_map) if x else "unknown",
                )
                predictions.append(pred)
        else:
            print(f"Using in-context learning with temperature {self.base_temperature}")
            for prompt in tqdm(prompts, desc="OpenAI inference"):
                generated = self._get_samples(prompt, self.base_temperature, 1)[0]
                predictions.append(self._extract_prediction(generated, label_map))

        return predictions

    def eval_predictions(
        self,
        y_true: List[str],
        y_pred: List[str],
        label_map: Dict[int, str],
    ) -> Dict[str, object]:
        y_true_arr = np.array([x.lower().strip() for x in y_true])
        y_pred_arr = np.array([x.lower().strip() for x in y_pred])
        labels = [label.lower() for label in label_map.values()]

        macro_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels, zero_division=0, average="macro")
        macro_recall = recall_score(y_true_arr, y_pred_arr, labels=labels, zero_division=0, average="macro")
        bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
        mcc = matthews_corrcoef(y_true_arr, y_pred_arr)

        precision_per_class_vals = precision_score(
            y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
        )
        recall_per_class_vals = recall_score(
            y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
        )
        f1_per_class_vals = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)

        precision_per_class = {
            labels[idx]: float(value) for idx, value in enumerate(precision_per_class_vals)
        }
        recall_per_class = {
            labels[idx]: float(value) for idx, value in enumerate(recall_per_class_vals)
        }
        f1_per_class = {
            labels[idx]: float(value) for idx, value in enumerate(f1_per_class_vals)
        }

        y_true_bin = label_binarize(y_true_arr, classes=labels)
        y_pred_bin = label_binarize(y_pred_arr, classes=labels)
        if len(labels) == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            y_pred_bin = np.hstack([1 - y_pred_bin, y_pred_bin])

        auprc_per_class: Dict[str, float] = {}
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
        }

    @staticmethod
    def infer_metadata(ds_name: str, df: pd.DataFrame) -> Tuple[str, str]:
        ratio = "unknown"
        maj_label = "unknown"

        if "balanced" in ds_name:
            ratio = "balanced"
        ratio_match = re.search(r"(\d+)_to_(\d+)", ds_name)
        if ratio_match:
            ratio = f"{ratio_match.group(1)}:{ratio_match.group(2)}"

        majority_match = re.search(r"([A-Za-z0-9]+)_majority", ds_name)
        if majority_match:
            maj_label = majority_match.group(1)

        try:
            counts = df["label"].value_counts()
            if len(counts) > 0:
                majority_from_df = counts.idxmax()
                if maj_label == "unknown":
                    maj_label = str(majority_from_df)
                maj_count = int(counts.max())
                others = counts.drop(majority_from_df)
                min_count = int(others.min()) if len(others) > 0 else 0
                ratio = f"{maj_count}:{min_count}"
        except Exception:
            pass

        return ratio, maj_label

    def _save_results(self, results: List[Dict[str, object]], out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        if not results:
            return

        df_agg = pd.json_normalize(results)
        df_agg.columns = [col.replace(".", "_") for col in df_agg.columns]

        timestamp = datetime.now().strftime("%m_%d_%Y")
        df_agg["saved_timestamp"] = timestamp

        use_sc = bool(results[-1].get("self_consistency_samples"))
        prefix = "SC" if use_sc else "ICL"
        filename = f"{prefix}_results_{self.model_name.replace('/', '_')}_{timestamp}.csv"
        path = os.path.join(out_dir, filename)
        df_agg.to_csv(path, index=False)

    def run_experiments(
        self,
        datasets_dict: Dict[str, pd.DataFrame],
        dataset_name: str,
        label_map: Dict[int, str],
        shots_minority: int,
        shots_majority: int,
        forced_maj_label: Optional[str],
        use_self_consistency: bool,
        sc_sample_counts: List[int],
        output_root: str,
    ) -> pd.DataFrame:
        results: List[Dict[str, object]] = []
        out_dir = os.path.join(output_root, dataset_name)
        os.makedirs(out_dir, exist_ok=True)

        if use_self_consistency:
            min_range = [shots_minority]
            maj_range = [shots_majority]
            sample_counts = sc_sample_counts or [5]
        else:
            min_range = [0] if shots_minority == 0 else list(range(0, shots_minority + 1, 2))
            maj_range = [0] if shots_majority == 0 else list(range(0, shots_majority + 1, 2))
            sample_counts = [None]

        for variant_name, df in datasets_dict.items():
            print(f"=== Dataset variant: {variant_name} ===")
            test_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            if use_self_consistency:
                shot_min = min_range[0]
                shot_maj = maj_range[0]
                for sc_samples in sample_counts:
                    predictions = self.classify(
                        test_df,
                        label_map,
                        shots_minority=shot_min,
                        shots_majority=shot_maj,
                        forced_maj_label=forced_maj_label,
                        use_self_consistency=True,
                        sc_samples=int(sc_samples),
                    )
                    metrics = self.eval_predictions(test_df["label"].tolist(), predictions, label_map)
                    ratio, maj_label = self.infer_metadata(variant_name, df)
                    results.append(
                        {
                            "model": self.model_name,
                            "dataset": variant_name,
                            "shots_minority": int(shot_min),
                            "shots_majority": int(shot_maj),
                            "self_consistency_samples": int(sc_samples),
                            "dataset_ratio": ratio,
                            "majority_label": maj_label,
                            **metrics,
                        }
                    )
                    self._save_results(results, out_dir)
            else:
                for shot_min in min_range:
                    for shot_maj in maj_range:
                        print(f"Shots configuration → minority: {shot_min}, majority: {shot_maj}")
                        predictions = self.classify(
                            test_df,
                            label_map,
                            shots_minority=int(shot_min),
                            shots_majority=int(shot_maj),
                            forced_maj_label=forced_maj_label,
                            use_self_consistency=False,
                            sc_samples=sample_counts[0] or 1,
                        )
                        metrics = self.eval_predictions(test_df["label"].tolist(), predictions, label_map)
                        ratio, maj_label = self.infer_metadata(variant_name, df)
                        results.append(
                            {
                                "model": self.model_name,
                                "dataset": variant_name,
                                "shots_minority": int(shot_min),
                                "shots_majority": int(shot_maj),
                                "self_consistency_samples": None,
                                "dataset_ratio": ratio,
                                "majority_label": maj_label,
                                **metrics,
                            }
                        )
                        self._save_results(results, out_dir)

        return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenAI models on imbalanced datasets")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--data-dir", type=str, default="Data", help="Root directory for datasets")
    parser.add_argument("--output-dir", type=str, default="results/openai", help="Directory for results")
    parser.add_argument("--datasets", nargs="+", choices=list(LABEL_MAPS.keys()),
                        default=list(LABEL_MAPS.keys()), help="Datasets to evaluate")
    parser.add_argument("--different-shots", action="store_true",
                        help="Use different shot counts for majority/minority labels")
    parser.add_argument("--shots", type=int, default=4, help="Shot count when not using --different-shots")
    parser.add_argument("--shots-minority", type=int, default=4,
                        help="Minority shot count when --different-shots is set")
    parser.add_argument("--shots-majority", type=int, default=4,
                        help="Majority shot count when --different-shots is set")
    parser.add_argument("--max-output-tokens", type=int, default=16, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for ICL mode")
    parser.add_argument("--use-self-consistency", action="store_true",
                        help="Enable self-consistency decoding")
    parser.add_argument("--sc-samples", type=int, nargs="+", default=[2],
                        help="Samples to draw per prediction in self-consistency mode")
    parser.add_argument("--sc-temperature", type=float, default=0.7,
                        help="Sampling temperature for self-consistency")
    parser.add_argument("--majority-label", type=str, default=None,
                        help="Force a specific majority label in prompts")
    parser.add_argument(
        "--rows-per-class",
        type=int,
        default=100,
        help="Subsample each dataset variant to N rows per class (default: 100; set <=0 to disable)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="results/openai_cache",
        help="Directory for caching OpenAI completions (default: results/openai_cache)",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Skip reading/writing cached OpenAI completions",
    )

    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set; OpenAI client may fail to authenticate.")

    shots_minority = args.shots_minority if args.different_shots else args.shots
    shots_majority = args.shots_majority if args.different_shots else args.shots
    sc_sample_counts = list(args.sc_samples)

    evaluator = OpenAIEvaluator(
        model_name=args.model,
        max_new_tokens=args.max_output_tokens,
        base_temperature=args.temperature,
        sc_temperature=args.sc_temperature,
        cache_dir=None if args.disable_cache else args.cache_dir,
        use_cache=not args.disable_cache,
    )

    loader = DatasetLoader(LABEL_MAPS)
    combined_results: List[pd.DataFrame] = []

    start_time = time.time()

    for dataset_name in args.datasets:
        if dataset_name == "ag_news":
            dataset_variants = loader.load_ag_news_data(os.path.join(args.data_dir, "ag_news"))
        elif dataset_name == "toxic_text":
            dataset_variants = loader.load_toxic_text_data(os.path.join(args.data_dir, "toxic_text"))
        else:
            dataset_variants = loader.load_twitter_emotion_data(os.path.join(args.data_dir, "twitter_emotion"))

        label_map = LABEL_MAPS[dataset_name]
        dataset_variants = loader.reduce_size(dataset_variants, args.rows_per_class)

        df_results = evaluator.run_experiments(
            datasets_dict=dataset_variants,
            dataset_name=dataset_name,
            label_map=label_map,
            shots_minority=shots_minority,
            shots_majority=shots_majority,
            forced_maj_label=args.majority_label,
            use_self_consistency=args.use_self_consistency,
            sc_sample_counts=sc_sample_counts,
            output_root=args.output_dir,
        )

        if not df_results.empty:
            combined_results.append(df_results)

    elapsed = time.time() - start_time
    print(f"Finished evaluation in {elapsed:.2f} seconds")

    if combined_results:
        os.makedirs(args.output_dir, exist_ok=True)
        combined_df = pd.concat(combined_results, ignore_index=True)
        combined_flat = pd.json_normalize(combined_df.to_dict(orient="records"))
        combined_flat.columns = [col.replace(".", "_") for col in combined_flat.columns]
        timestamp = datetime.now().strftime("%m_%d_%Y")
        prefix = "SC" if args.use_self_consistency else "ICL"
        filename = f"{prefix}_results_{args.model.replace('/', '_')}_all_{timestamp}.csv"
        path = os.path.join(args.output_dir, filename)
        combined_flat.to_csv(path, index=False)
        print(f"Combined results saved to {path}")


if __name__ == "__main__":
    main()
