from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evals import infer_preference as preference

__all__ = [
    "ThresholdBasedVoting",
    "apply_threshold_based_voting",
    "choose_with_threshold_override",
    "load_preferred_from_metrics",
    "summarise_votes",
    "write_jsonl",
]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if rows:
        payload += "\n"
    preference.atomic_write(path, payload)


def choose_with_threshold_override(
    samples: Sequence[Optional[str]], 
    preferred_labels: Optional[Sequence[str]],
    threshold: int
) -> Tuple[Optional[str], str]:
    """
    Apply threshold-based preference mitigation.
    
    If the most common label is NOT preferred, use it (normal majority voting).
    Otherwise, if any preferred label exceeds threshold, exclude preferred labels
    and pick the most common non-preferred label.
    If preferred labels don't exceed threshold, use normal majority voting.
    
    Args:
        samples: List of sampled predictions
        preferred_labels: List of labels the model is biased toward (or single label)
        threshold: Minimum count required to trigger mitigation of preferred label
    
    Returns:
        (final_label, decision_mode) where mode describes the decision process
    """
    votes = [label for label in samples if label]
    if not votes:
        return None, "none"
    
    # Normalize preferred_labels to a list
    if preferred_labels is None:
        preferred_set = set()
    elif isinstance(preferred_labels, str):
        preferred_set = {preferred_labels}
    else:
        preferred_set = set(preferred_labels)
    
    # Count all votes
    counter = Counter(votes)
    most_common_label, most_common_count = counter.most_common(1)[0]
    
    # If most common label is NOT preferred, use it directly
    if most_common_label not in preferred_set:
        return most_common_label, "majority_not_preferred"
    
    # Most common label IS preferred - check if it exceeds threshold
    if most_common_count > threshold:
        # Preferred label exceeded threshold - apply mitigation
        remaining_votes = [label for label in votes if label not in preferred_set]
        if remaining_votes:
            # Return most common non-preferred label
            remaining_counter = Counter(remaining_votes)
            majority_label = remaining_counter.most_common(1)[0][0]
            return majority_label, "threshold_override"
        else:
            # Only preferred labels exist, can't mitigate
            return most_common_label, "forced_preferred"
    else:
        # Preferred label didn't exceed threshold - use normal majority voting
        return most_common_label, "majority_below_threshold"


def load_preferred_from_metrics(path: Path, labels: Sequence[str]) -> Optional[str]:
    
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    per_class = data.get("per_class") or data.get("per_class_metrics") or {}
    preferences: List[Tuple[str, float]] = []
    
    for label in labels:
        recall_value = None
        precision_value = None
        
        entry = per_class.get(label)
        if isinstance(entry, Mapping):
            recall_value = entry.get("recall")
            precision_value = entry.get("precision")
        
        if recall_value is None:
            recall_value = data.get("recall", {}).get(label)
        if precision_value is None:
            precision_value = data.get("precision", {}).get(label)
        
        recall = float(recall_value or 0.0)
        precision = float(precision_value or 1.0)
        
        # Calculate preference score: recall / precision
        if precision > 0:
            preference = recall / precision
        else:
            preference = 0.0
            
        preferences.append((label, preference))

    # Sort by preference score (descending) - highest preference = most biased toward
    preferences.sort(key=lambda item: item[1], reverse=True)
    
    if preferences:
        return preferences[0][0]  # Return label with highest preference
    return None


class ThresholdBasedVoting:
    """Coordinator for applying threshold-based preference mitigation during inference."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 32,
        samples_per_example: int = 5,
        preferred_label: Optional[str] = None,
        preferred_from_metrics: Optional[str] = None,
        threshold: int = 20,
        prompt_template: Optional[str] = None,
        seed: int = 0,
        include_text: bool = True,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.samples_per_example = max(1, samples_per_example)
        self.threshold = threshold
        self.prompt_template = prompt_template
        self.seed = seed
        self.include_text = include_text

        self._cli_preferred_label = preferred_label
        self._metrics_path = preferred_from_metrics
        self._warned_no_preferred = False

        self._generator: Optional[preference.BaseGenerator] = None
        self._generator_labels: Optional[Tuple[str, ...]] = None

        preference.set_seed(seed)

    def _derive_preferred_label(self, labels: Sequence[str]) -> Optional[str]:
        """Derive the preferred (biased) label from CLI arg or metrics file."""
        if self._cli_preferred_label:
            return self._cli_preferred_label
        if self._metrics_path:
            path = Path(self._metrics_path)
            if not path.exists():
                raise preference.UserInputError(f"Metrics file not found: {self._metrics_path}")
            return load_preferred_from_metrics(path, labels)
        if not self._warned_no_preferred:
            print("[WARN] no preferred label provided; threshold logic may not work as expected.", file=sys.stderr)
            self._warned_no_preferred = True
        return None

    def _limit_records(self, records: Sequence[Mapping[str, object]], max_examples: Optional[int]) -> List[Mapping[str, object]]:
        if max_examples and max_examples > 0:
            return list(records[: max_examples])
        return list(records)

    def _ensure_generator(self, labels: Sequence[str]) -> preference.BaseGenerator:
        key = tuple(labels)
        if self._generator is not None and self._generator_labels == key:
            return self._generator
        self._generator = preference.build_mlx_generator(
            labels=labels,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            mlx_prompt_template=self.prompt_template,
        )
        self._generator_labels = key
        return self._generator

    def compute_votes(
        self,
        records: Sequence[Mapping[str, object]],
        labels: Sequence[str],
        *,
        preferred_label: Optional[str] = None,
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        derived_preferred = preferred_label if preferred_label is not None else self._derive_preferred_label(labels)
        generator = self._ensure_generator(labels)
        rows = apply_threshold_based_voting(
            records,
            generator,
            samples_per_example=self.samples_per_example,
            preferred_label=derived_preferred,
            threshold=self.threshold,
            include_text=self.include_text,
        )
        summary = summarise_votes(rows, labels, derived_preferred)
        return rows, summary

    def run(
        self,
        dataset: preference.Dataset,
        *,
        max_examples: Optional[int] = None,
        out: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, object]:
        records = self._limit_records(dataset.records, max_examples)
        if not records:
            raise preference.UserInputError("Input dataset is empty after applying max_examples limit.")

        rows, summary = self.compute_votes(records, dataset.labels)
        summary.update(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "labels": dataset.labels,
                "model": self.model,
                "samples_per_example": self.samples_per_example,
            }
        )

        if out:
            out_path = preference._prepare_output_path(out, self.model, ".jsonl", overwrite)
            if out_path is None:
                raise preference.UserInputError("Unable to resolve output path.")
            write_jsonl(out_path, rows)
            summary["out_path"] = str(out_path)

        return summary


def _majority_label(votes: Sequence[Optional[str]]) -> Optional[str]:
    filtered = [vote for vote in votes if vote]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def apply_threshold_based_voting(
    records: Sequence[Mapping[str, object]],
    generator: preference.BaseGenerator,
    *,
    samples_per_example: int,
    preferred_label: Optional[str],
    threshold: int,
    include_text: bool = True,
) -> List[Dict[str, object]]:
    """Apply threshold-based voting: only output preferred label if count > threshold."""
    results: List[Dict[str, object]] = []
    for index, record in enumerate(records):
        text_raw = record.get("text", "")
        text = str(text_raw).strip()
        if not text:
            raise preference.UserInputError(f"Record {index} missing 'text' field.")

        votes = generator.sample(text, max(1, samples_per_example))
        final_label, mode = choose_with_threshold_override(votes, preferred_label, threshold)
        output: Dict[str, object] = {
            "id": record.get("id", index),
            "samples": votes,
            "majority": _majority_label(votes),
            "final": final_label,
            "threshold_applied": mode == "threshold_override",
            "decision_mode": mode,
        }
        if include_text:
            output["text"] = text
        if "label" in record:
            output["label"] = record["label"]
        results.append(output)
    return results


def summarise_votes(
    rows: Sequence[Mapping[str, object]],
    labels: Sequence[str],
    preferred_label: Optional[str],
) -> Dict[str, object]:
    """Summarize voting results with threshold-based statistics."""
    finals = Counter(row.get("final") for row in rows if row.get("final"))
    majorities = Counter(row.get("majority") for row in rows if row.get("majority"))
    threshold_applied = sum(1 for row in rows if row.get("threshold_applied"))
    total = len(rows)
    return {
        "total": total,
        "threshold_fraction": (threshold_applied / total) if total else 0.0,
        "final_counts": {label: finals.get(label, 0) for label in labels},
        "majority_counts": {label: majorities.get(label, 0) for label in labels},
        "preferred_label": preferred_label,
    }


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    dataset = preference.load_dataset(args.inputs, args.labels)
    pipeline = ThresholdBasedVoting(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        samples_per_example=args.vote_samples,
        preferred_label=args.preferred_label,
        preferred_from_metrics=args.preferred_from_metrics,
        threshold=args.threshold,
        prompt_template=args.prompt_template,
        seed=args.seed,
        include_text=not args.strip_text,
    )

    summary = pipeline.run(
        dataset,
        max_examples=(args.max_examples if args.max_examples > 0 else None),
        out=args.out,
        overwrite=args.overwrite,
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold-based voting utility for MLX inference.")
    parser.add_argument("--inputs", required=True, help="Path to JSON/JSONL/CSV dataset with a 'text' field.")
    parser.add_argument("--labels", nargs="*", help="Explicit label list; inferred when omitted.")
    parser.add_argument("--out", required=True, help="Output JSONL destination or directory.")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit", help="MLX model identifier.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--vote_samples", type=int, default=5, help="Samples per example.")
    parser.add_argument("--preferred_label", help="Explicit preferred (biased) label.")
    parser.add_argument("--preferred_from_metrics", help="Metrics JSON to derive preferred label.")
    parser.add_argument("--threshold", type=int, default=20, help="Threshold for accepting preferred label.")
    parser.add_argument("--max_examples", type=int, default=0, help="Limit processed records (0 = all).")
    parser.add_argument("--prompt_template", help="Optional prompt template for MLX (uses {text} and {label_list}).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs instead of auto-renaming.")
    parser.add_argument("--strip_text", action="store_true", help="Omit text field from JSONL output.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_pipeline(args)
    except preference.UserInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    print(
        "[ OK ] wrote {out} (total={total}, threshold_applied={thresh:.1%})".format(
            out=result["out_path"],
            total=result["total"],
            thresh=result.get("threshold_fraction", 0.0),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
