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
    "MinorityFirstVotingPipeline",
    "apply_minority_first_voting",
    "choose_with_fair_override",
    "load_weak_from_metrics",
    "summarise_votes",
    "write_jsonl",
]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if rows:
        payload += "\n"
    preference.atomic_write(path, payload)


def choose_with_fair_override(samples: Sequence[Optional[str]], weak_set: set[str]) -> Tuple[Optional[str], str]:
    votes = [label for label in samples if label]
    if not votes:
        return None, "none"

    majority_label = Counter(votes).most_common(1)[0][0]
    weak_votes = [label for label in votes if label in weak_set]
    if weak_votes:
        weak_counter = Counter(weak_votes)
        max_count = max(weak_counter.values())
        candidates = sorted(label for label, count in weak_counter.items() if count == max_count)
        return candidates[0], "override"
    return majority_label, "majority"


def load_weak_from_metrics(path: Path, labels: Sequence[str], percent: int) -> List[str]:
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
        precision = float(precision_value or 1.0)  # Avoid division by zero
        
        # Calculate preference score: recall / precision
        # Higher preference = more false positives relative to true positives (underrepresented)
        if precision > 0:
            preference = recall / precision
        else:
            preference = 0.0
            
        preferences.append((label, preference))

    # Sort by preference score (ascending) - lowest preference = most underrepresented
    preferences.sort(key=lambda item: item[1])
    count = max(1, int(len(labels) * percent / 100))
    return [label for label, _ in preferences[:count]]


class MinorityFirstVotingPipeline:
    """Coordinator for applying minority-first voting during inference."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 32,
        samples_per_example: int = 5,
        weak_labels: Optional[Sequence[str]] = None,
        weak_from_metrics: Optional[str] = None,
        weak_percent: int = 25,
        prompt_template: Optional[str] = None,
        seed: int = 0,
        include_text: bool = True,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.samples_per_example = max(1, samples_per_example)
        self.prompt_template = prompt_template
        self.seed = seed
        self.include_text = include_text

        self._cli_weak_labels = self._normalize_cli_labels(weak_labels)
        self._metrics_path = weak_from_metrics
        self._weak_percent = weak_percent
        self._warned_no_weak = False

        self._generator: Optional[preference.BaseGenerator] = None
        self._generator_labels: Optional[Tuple[str, ...]] = None

        preference.set_seed(seed)

    @staticmethod
    def _normalize_cli_labels(values: Optional[Sequence[str]]) -> List[str]:
        if not values:
            return []
        items: List[str] = []
        for value in values:
            for part in str(value).split(','):
                item = part.strip()
                if item and item not in items:
                    items.append(item)
        return items

    def _derive_weak_labels(self, labels: Sequence[str]) -> set[str]:
        if self._cli_weak_labels:
            return set(self._cli_weak_labels)
        if self._metrics_path:
            path = Path(self._metrics_path)
            if not path.exists():
                raise preference.UserInputError(f"Metrics file not found: {self._metrics_path}")
            return set(load_weak_from_metrics(path, labels, self._weak_percent))
        if not self._warned_no_weak:
            print("[WARN] no weak labels provided; overrides disabled.", file=sys.stderr)
            self._warned_no_weak = True
        return set()

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
        weak_labels: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        derived = set(weak_labels) if weak_labels is not None else self._derive_weak_labels(labels)
        generator = self._ensure_generator(labels)
        rows = apply_minority_first_voting(
            records,
            generator,
            samples_per_example=self.samples_per_example,
            weak_labels=derived,
            include_text=self.include_text,
        )
        summary = summarise_votes(rows, labels, derived)
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


def apply_minority_first_voting(
    records: Sequence[Mapping[str, object]],
    generator: preference.BaseGenerator,
    *,
    samples_per_example: int,
    weak_labels: Iterable[str],
    include_text: bool = True,
) -> List[Dict[str, object]]:
    weak_set = {label for label in weak_labels if label}
    results: List[Dict[str, object]] = []
    for index, record in enumerate(records):
        text_raw = record.get("text", "")
        text = str(text_raw).strip()
        if not text:
            raise preference.UserInputError(f"Record {index} missing 'text' field.")

        votes = generator.sample(text, max(1, samples_per_example))
        final_label, mode = choose_with_fair_override(votes, weak_set)
        output: Dict[str, object] = {
            "id": record.get("id", index),
            "samples": votes,
            "majority": _majority_label(votes),
            "final": final_label,
            "override_applied": mode == "override",
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
    weak_labels: Iterable[str],
) -> Dict[str, object]:
    finals = Counter(row.get("final") for row in rows if row.get("final"))
    majorities = Counter(row.get("majority") for row in rows if row.get("majority"))
    overrides = sum(1 for row in rows if row.get("override_applied"))
    total = len(rows)
    return {
        "total": total,
        "override_fraction": (overrides / total) if total else 0.0,
        "final_counts": {label: finals.get(label, 0) for label in labels},
        "majority_counts": {label: majorities.get(label, 0) for label in labels},
        "weak_labels": sorted({label for label in weak_labels if label}),
    }


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    dataset = preference.load_dataset(args.inputs, args.labels)
    pipeline = MinorityFirstVotingPipeline(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        samples_per_example=args.vote_samples,
        weak_labels=args.weak_labels,
        weak_from_metrics=args.weak_from_metrics,
        weak_percent=args.weak_percent,
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
    parser = argparse.ArgumentParser(description="Minority-first voting utility for MLX inference.")
    parser.add_argument("--inputs", required=True, help="Path to JSON/JSONL/CSV dataset with a 'text' field.")
    parser.add_argument("--labels", nargs="*", help="Explicit label list; inferred when omitted.")
    parser.add_argument("--out", required=True, help="Output JSONL destination or directory.")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit", help="MLX model identifier.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--vote_samples", type=int, default=5, help="Samples per example.")
    parser.add_argument("--weak_labels", nargs="*", help="Weak label list (space or comma separated).")
    parser.add_argument("--weak_from_metrics", help="Metrics JSON to derive weak labels.")
    parser.add_argument("--weak_percent", type=int, default=25, help="Bottom percent of labels to mark weak.")
    parser.add_argument("--max_examples", type=int, default=0, help="Limit processed records (0 = all).")
    parser.add_argument("--prompt_template", help="Optional prompt template for MLX (uses {text} and {label_list}).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs instead of auto-renaming.")
    parser.add_argument("--strip_text", action="store_true", help="Omit text field from JSONL output.")
    parser.add_argument("--run_tests", action="store_true", help="Run built-in sanity checks and exit.")
    return parser.parse_args(argv)


def run_tests() -> int:
    decision = choose_with_fair_override(["World", "World", "Sci/Tech"], {"Sci/Tech"})
    assert decision == ("Sci/Tech", "override")

    decision = choose_with_fair_override(["World", "Sports", "World"], {"Sci/Tech"})
    assert decision == ("World", "majority")

    decision = choose_with_fair_override(["B", "A", "B", "A"], {"A", "B"})
    assert decision == ("A", "override")

    class _StubGenerator(preference.BaseGenerator):
        def __init__(self, outputs: List[List[str]]):
            self._outputs = outputs
            self._index = 0

        def sample(self, prompt: str, n: int) -> List[str]:
            result = self._outputs[self._index]
            self._index += 1
            return list(result)

    records = [
        {"text": "Doc", "label": "World"},
        {"text": "Other", "label": "Sci/Tech"},
    ]
    generator = _StubGenerator([
        ["World", "World", "Sci/Tech"],
        ["Sci/Tech", "World", "Sci/Tech"],
    ])
    rows = apply_minority_first_voting(
        records,
        generator,
        samples_per_example=3,
        weak_labels={"Sci/Tech"},
        include_text=False,
    )
    assert rows[0]["final"] == "Sci/Tech"
    assert rows[1]["final"] == "Sci/Tech"

    summary = summarise_votes(rows, ["World", "Sci/Tech"], {"Sci/Tech"})
    assert summary["final_counts"].get("Sci/Tech", 0) == 2

    class _TestPipeline(MinorityFirstVotingPipeline):
        def __init__(self) -> None:
            super().__init__(model="stub-model", samples_per_example=3)
            self._stub = _StubGenerator([
                ["World", "World", "Sci/Tech"],
                ["Sci/Tech", "World", "Sci/Tech"],
            ])

        def _ensure_generator(self, labels: Sequence[str]) -> preference.BaseGenerator:
            return self._stub

        def _derive_weak_labels(self, labels: Sequence[str]) -> set[str]:
            return {"Sci/Tech"}

    dataset = preference.Dataset(
        records=records,
        labels=["World", "Sci/Tech"],
        dataset_prior={"World": 0.5, "Sci/Tech": 0.5},
    )
    pipeline = _TestPipeline()
    pipeline.include_text = False
    pipeline_summary = pipeline.run(dataset, out=None)
    assert pipeline_summary["total"] == 2
    assert pipeline_summary["final_counts"].get("Sci/Tech", 0) == 2

    print("[ OK ] tests passed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.run_tests:
        return run_tests()
    try:
        result = run_pipeline(args)
    except preference.UserInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    print(
        "[ OK ] wrote {out} (total={total}, overrides={overrides:.1%})".format(
            out=result["out_path"],
            total=result["total"],
            overrides=result.get("override_fraction", 0.0),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
