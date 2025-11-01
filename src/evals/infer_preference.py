from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Optional OpenAI dependency (imported lazily when needed)
try:  # pragma: no cover - guard for optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - no hard dependency
    OpenAI = None

EPS = 1e-9
DEFAULT_PRIOR_PROMPTS = [
    "42",
    "none",
    "N/A",
    "unknown",
    "I cannot answer",
]
DEFAULT_DEMO_DATA = [
    {"id": "demo-1", "text": "The product arrived broken and support never replied.", "label": "negative"},
    {"id": "demo-2", "text": "Incredible battery life and solid build quality.", "label": "positive"},
    {"id": "demo-3", "text": "It works, but shipping was slower than expected.", "label": "neutral"},
]


class UserInputError(Exception):
    """Raised for CLI usage problems that should exit with status 2."""


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class Dataset:
    records: List[Mapping[str, str]]
    labels: List[str]
    dataset_prior: Dict[str, float]


class BaseGenerator:
    def sample(self, prompt: str, n: int) -> List[str]:  # pragma: no cover - interface only
        raise NotImplementedError


class LocalMockGenerator(BaseGenerator):
    """Deterministic pseudo-random generator used for demo/tests."""

    def __init__(self, labels: Sequence[str], seed: Optional[int] = None) -> None:
        self.labels = list(labels)
        self.seed = seed or 0

    def sample(self, prompt: str, n: int) -> List[str]:
        mix = hash((prompt, self.seed))
        rng = random.Random(mix)
        if not self.labels:
            return []
        return [rng.choice(self.labels) for _ in range(n)]


class OpenAIGenerator(BaseGenerator):
    def __init__(
        self,
        model: str,
        labels: Sequence[str],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> None:
        if OpenAI is None:
            raise UserInputError(
                "OpenAI SDK not available. Install `openai` or use --backend local/--demo."
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise UserInputError("OPENAI_API_KEY not set; cannot use OpenAI backend.")
        self.client = OpenAI()
        self.model = model
        self.labels = list(labels)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def sample(self, prompt: str, n: int) -> List[str]:
        options = self.labels
        if not options:
            return []
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                n=n,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network errors
            raise UserInputError(f"OpenAI request failed: {exc}") from exc

        outputs: List[str] = []
        for choice in getattr(response, "choices", []):
            message = getattr(choice, "message", None)
            content: Optional[str] = None
            if message is None:
                content = getattr(choice, "content", None)  # newer SDK field
            elif isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
            if not content:
                continue
            outputs.append(_extract_label(content, options))
        if len(outputs) < n:
            outputs.extend([random.choice(options) for _ in range(n - len(outputs))])
        return outputs


def _extract_label(text: str, labels: Sequence[str]) -> str:
    canon = {label.lower(): label for label in labels}
    tokens = str(text).strip().split()
    for token in tokens:
        cleaned = token.strip(".,:;!?").lower()
        if cleaned in canon:
            return canon[cleaned]
    return tokens[0] if tokens else labels[0]


def load_dataset(path: Optional[str], labels_cli: Optional[Sequence[str]], demo: bool) -> Dataset:
    if demo:
        labels = ["negative", "neutral", "positive"]
        return Dataset(records=DEFAULT_DEMO_DATA, labels=labels, dataset_prior=_prior_from_labels(DEFAULT_DEMO_DATA, labels))

    if not path:
        raise UserInputError("--inputs is required unless --demo is supplied.")

    file_path = Path(path)
    if not file_path.exists():
        raise UserInputError(f"Input file not found: {path}")

    suffix = file_path.suffix.lower()
    if suffix == ".json" or suffix == ".jsonl":
        records = _load_json_records(file_path)
    elif suffix == ".csv":
        records = _load_csv_records(file_path)
    else:
        raise UserInputError("Unsupported input format. Use JSON, JSONL, or CSV.")

    if not records:
        raise UserInputError("Input dataset is empty.")

    labels = _resolve_labels(records, labels_cli)
    dataset_prior = _prior_from_labels(records, labels)
    return Dataset(records=records, labels=labels, dataset_prior=dataset_prior)


def _resolve_labels(records: Sequence[Mapping[str, str]], labels_cli: Optional[Sequence[str]]) -> List[str]:
    labels = list(dict.fromkeys(labels_cli or []))
    dataset_labels = [r.get("label") for r in records if r.get("label")]
    for label in dataset_labels:
        if label not in labels:
            labels.append(label)
    if not labels:
        raise UserInputError("Unable to determine label set; provide --labels or labeled data.")
    return labels


def _load_json_records(path: Path) -> List[Mapping[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        if text.startswith("["):
            data = json.loads(text)
            if isinstance(data, list):
                return [dict(item) for item in data]
        records: List[Mapping[str, str]] = []
        for line in text.splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records
    except json.JSONDecodeError as exc:
        raise UserInputError(f"Failed to parse JSON inputs: {exc}") from exc


def _load_csv_records(path: Path) -> List[Mapping[str, str]]:
    import csv

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _prior_from_labels(records: Sequence[Mapping[str, str]], labels: Sequence[str]) -> Dict[str, float]:
    counts = Counter(r.get("label", "") for r in records if r.get("label") in labels)
    total = sum(counts.values())
    if total == 0:
        uniform = 1.0 / max(len(labels), 1)
        return {label: uniform for label in labels}
    return {label: counts.get(label, 0) / total for label in labels}


@dataclass
class VoteSummary:
    mean: Dict[str, float]
    ci: Dict[str, Tuple[float, float]]


@dataclass
class EstimationArtifacts:
    p0: VoteSummary
    vbar: VoteSummary
    overprediction: VoteSummary
    skew: VoteSummary
    vbar_cal: Optional[VoteSummary]
    metrics_cal: Optional[Dict[str, VoteSummary]]


def sample_votes(
    generator: BaseGenerator,
    prompts: Sequence[str],
    labels: Sequence[str],
    samples_per_prompt: int,
) -> Tuple[List[Counter], Dict[str, float]]:
    vote_counters: List[Counter] = []
    aggregate = Counter()
    for prompt in prompts:
        votes = Counter(generator.sample(prompt, samples_per_prompt))
        vote_counters.append(votes)
        aggregate.update(votes)
    total = sum(aggregate.values())
    mean = {label: aggregate.get(label, 0) / total if total else 0.0 for label in labels}
    return vote_counters, mean


def counters_to_matrix(counters: Sequence[Counter], labels: Sequence[str]) -> np.ndarray:
    if not counters:
        return np.zeros((0, len(labels)), dtype=float)
    matrix = np.zeros((len(counters), len(labels)), dtype=float)
    for idx, counter in enumerate(counters):
        total = sum(counter.values())
        for col, label in enumerate(labels):
            val = counter.get(label, 0)
            matrix[idx, col] = (val / total) if total else 0.0
    return matrix


def summarize_distribution(
    avg: Mapping[str, float],
    matrix: np.ndarray,
    labels: Sequence[str],
    bootstrap_iters: int,
    rng: random.Random,
) -> VoteSummary:
    ci_raw = bootstrap_ci(matrix, bootstrap_iters, rng)
    ci_named = rename_ci(ci_raw, labels)
    return VoteSummary(mean=dict(avg), ci=ci_named)


def bootstrap_ci(matrix: np.ndarray, iters: int, rng: random.Random) -> Dict[str, Tuple[float, float]]:
    if matrix.size == 0 or iters <= 0:
        return {}
    n = matrix.shape[0]
    idx = np.arange(n)
    samples = []
    for _ in range(iters):
        resampled = rng.choices(idx, k=n)
        sample_mean = matrix[resampled].mean(axis=0)
        samples.append(sample_mean)
    stacked = np.vstack(samples)
    lower = np.percentile(stacked, 2.5, axis=0)
    upper = np.percentile(stacked, 97.5, axis=0)
    return {f"label_{i}": (float(lower[i]), float(upper[i])) for i in range(matrix.shape[1])}


def rename_ci(ci: Dict[str, Tuple[float, float]], labels: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    return {label: ci.get(f"label_{i}", (math.nan, math.nan)) for i, label in enumerate(labels)}


def mean_from_matrix(matrix: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    if matrix.size == 0:
        return {label: 0.0 for label in labels}
    mean = matrix.mean(axis=0)
    return {label: float(mean[idx]) for idx, label in enumerate(labels)}


def ratio_matrix(matrix: np.ndarray, denom: Mapping[str, float], labels: Sequence[str]) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, len(labels)), dtype=float)
    denom_vec = np.array([denom.get(label, 0.0) + EPS for label in labels], dtype=float)
    denom_vec[denom_vec == 0.0] = EPS
    return matrix / denom_vec


def compute_metrics(
    generator: BaseGenerator,
    dataset: Dataset,
    prior_prompts: Sequence[str],
    votes_per_prompt: int,
    prior_samples: int,
    bootstrap_iters: int,
    rng: random.Random,
) -> EstimationArtifacts:
    labels = dataset.labels

    prior_counters, p0_mean = sample_votes(generator, prior_prompts, labels, prior_samples)
    prior_matrix = counters_to_matrix(prior_counters, labels)
    p0_summary = summarize_distribution(p0_mean, prior_matrix, labels, bootstrap_iters, rng)

    prompts = [record.get("text", "") for record in dataset.records]
    vote_counters, vbar_mean = sample_votes(generator, prompts, labels, votes_per_prompt)
    vote_matrix = counters_to_matrix(vote_counters, labels)
    vbar_summary = summarize_distribution(vbar_mean, vote_matrix, labels, bootstrap_iters, rng)

    dataset_prior = dataset.dataset_prior or {label: 1.0 / len(labels) for label in labels}

    overpred_matrix = ratio_matrix(vote_matrix, dataset_prior, labels)
    overpred_mean = mean_from_matrix(overpred_matrix, labels)
    skew_matrix = ratio_matrix(vote_matrix, p0_mean, labels)
    skew_mean = mean_from_matrix(skew_matrix, labels)

    overpred_summary = summarize_distribution(overpred_mean, overpred_matrix, labels, bootstrap_iters, rng)
    skew_summary = summarize_distribution(skew_mean, skew_matrix, labels, bootstrap_iters, rng)

    return EstimationArtifacts(
        p0=p0_summary,
        vbar=vbar_summary,
        overprediction=overpred_summary,
        skew=skew_summary,
        vbar_cal=None,
        metrics_cal=None,
    )


def apply_calibration(
    artifacts: EstimationArtifacts,
    dataset_prior: Mapping[str, float],
    labels: Sequence[str],
) -> EstimationArtifacts:
    vbar = artifacts.vbar.mean
    p0 = artifacts.p0.mean
    adjusted = {label: vbar.get(label, 0.0) / (p0.get(label, 0.0) + EPS) for label in labels}
    norm = sum(adjusted.values()) or 1.0
    vbar_cal = {label: val / norm for label, val in adjusted.items()}

    vbar_cal_summary = VoteSummary(mean=vbar_cal, ci={label: (math.nan, math.nan) for label in labels})

    overpred_cal = {
        label: vbar_cal[label] / (dataset_prior.get(label, 0.0) + EPS) for label in labels
    }
    skew_cal = {label: vbar_cal[label] / (p0.get(label, 0.0) + EPS) for label in labels}

    metrics_cal = {
        "overprediction": VoteSummary(mean=overpred_cal, ci={label: (math.nan, math.nan) for label in labels}),
        "skew": VoteSummary(mean=skew_cal, ci={label: (math.nan, math.nan) for label in labels}),
    }

    return dataclasses.replace(artifacts, vbar_cal=vbar_cal_summary, metrics_cal=metrics_cal)


def choose_generator(
    backend: str,
    labels: Sequence[str],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: Optional[int],
) -> BaseGenerator:
    backend = backend.lower()
    if backend == "local":
        return LocalMockGenerator(labels, seed)
    if backend not in {"auto", "openai"}:
        raise UserInputError("Unsupported backend; choose from auto, openai, local.")
    if backend == "openai":
        return OpenAIGenerator(model, labels, temperature, top_p, max_tokens)
    # auto: prefer OpenAI when available
    with contextlib.suppress(Exception):
        return OpenAIGenerator(model, labels, temperature, top_p, max_tokens)
    return LocalMockGenerator(labels, seed)


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    dataset = load_dataset(args.inputs, args.labels, args.demo)
    rng = random.Random(args.seed or 0)

    generator = choose_generator(
        backend=args.backend,
        labels=dataset.labels,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    artifacts = compute_metrics(
        generator=generator,
        dataset=dataset,
        prior_prompts=args.prior_prompts or DEFAULT_PRIOR_PROMPTS,
        votes_per_prompt=args.vote_samples,
        prior_samples=args.prior_samples,
        bootstrap_iters=args.bootstrap_iters,
        rng=rng,
    )

    if args.calibrate:
        artifacts = apply_calibration(artifacts, dataset.dataset_prior, dataset.labels)

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labels": dataset.labels,
        "input_size": len(dataset.records),
        "prior_prompts": args.prior_prompts or DEFAULT_PRIOR_PROMPTS,
        "vote_samples": args.vote_samples,
        "prior_samples": args.prior_samples,
        "P0": _summary_to_dict(artifacts.p0),
        "Vbar": _summary_to_dict(artifacts.vbar),
        "metrics": {
            "overprediction": _summary_to_dict(artifacts.overprediction),
            "skew": _summary_to_dict(artifacts.skew),
        },
        "dataset_prior": dataset.dataset_prior,
        "calibration_applied": bool(artifacts.vbar_cal),
    }

    if artifacts.vbar_cal:
        result["Vbar_cal"] = _summary_to_dict(artifacts.vbar_cal)
    if artifacts.metrics_cal:
        result["metrics_cal"] = {
            name: _summary_to_dict(summary) for name, summary in artifacts.metrics_cal.items()
        }

    maybe_write_json(args.out, result)
    maybe_write_markdown(args, dataset.labels, result)
    maybe_generate_plots(args, dataset.labels, result)
    return result


def _summary_to_dict(summary: VoteSummary) -> Dict[str, object]:
    return {"mean": summary.mean, "ci": summary.ci}


def maybe_write_json(path: Optional[str], payload: Dict[str, object]) -> None:
    if not path:
        return
    atomic_write(Path(path), json.dumps(payload, indent=2, sort_keys=True) + "\n")


def maybe_write_markdown(args: argparse.Namespace, labels: Sequence[str], result: Mapping[str, object]) -> None:
    report_path = getattr(args, "report_md", None)
    if not report_path:
        return
    content = render_markdown(labels, result)
    atomic_write(Path(report_path), content)


def render_markdown(labels: Sequence[str], result: Mapping[str, object]) -> str:
    lines = ["# Preference Report", "", f"Generated: {result.get('generated_at', 'unknown')}", ""]
    lines.append("## Prior Preference P₀(y)")
    lines.extend(_metric_table(labels, result["P0"]))
    lines.append("")
    lines.append("## Conditional Preference V̄(y)")
    lines.extend(_metric_table(labels, result["Vbar"]))
    lines.append("")
    lines.append("## Over-Prediction Ratio R(y)")
    lines.extend(_metric_table(labels, result["metrics"]["overprediction"]))
    lines.append("")
    lines.append("## Calibrated Skew S(y)")
    lines.extend(_metric_table(labels, result["metrics"]["skew"]))

    if result.get("calibration_applied") and result.get("Vbar_cal"):
        lines.append("")
        lines.append("## Calibrated Conditional Preference V̄_cal(y)")
        lines.extend(_metric_table(labels, result["Vbar_cal"]))
        metrics_cal = result.get("metrics_cal") or {}
        for key, title in [
            ("overprediction", "Calibrated Over-Prediction R_cal(y)"),
            ("skew", "Calibrated Skew S_cal(y)"),
        ]:
            if key in metrics_cal:
                lines.append("")
                lines.append(f"## {title}")
                lines.extend(_metric_table(labels, metrics_cal[key]))
    lines.append("")
    return "\n".join(lines)


def _metric_table(labels: Sequence[str], metric: Mapping[str, object]) -> List[str]:
    mean = metric.get("mean", {})
    ci = metric.get("ci", {})
    header = "| label | value | 95% CI |"
    sep = "|---|---|---|"
    rows = [header, sep]
    for label in labels:
        val = mean.get(label, math.nan)
        ci_low, ci_high = ci.get(label, (math.nan, math.nan))
        rows.append(f"| {label} | {val:.4f} | [{ci_low:.4f}, {ci_high:.4f}] |")
    return rows


def maybe_generate_plots(args: argparse.Namespace, labels: Sequence[str], result: Mapping[str, object]) -> None:
    plots_dir = getattr(args, "plots_dir", None)
    if not plots_dir:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise UserInputError(f"matplotlib is required for plotting: {exc}") from exc

    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _bar_plot(filename: str, values_a: Mapping[str, float], values_b: Optional[Mapping[str, float]], labels: Sequence[str], title: str):
        x = np.arange(len(labels))
        width = 0.35 if values_b else 0.6
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2 if values_b else x, [values_a.get(label, 0.0) for label in labels], width, label="model")
        if values_b is not None:
            ax.bar(x + width / 2, [values_b.get(label, 0.0) for label in labels], width, label="baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(title)
        ax.legend() if values_b is not None else None
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, format="png")
        plt.close(fig)

    uniform = {label: 1.0 / len(labels) for label in labels} if labels else {}
    dataset_prior = result.get("dataset_prior", {})
    _bar_plot("p0_vs_uniform.png", result["P0"]["mean"], uniform, labels, "Prior Preference vs Uniform")
    _bar_plot(
        "vbar_vs_dataset.png",
        result["Vbar"]["mean"],
        dataset_prior,
        labels,
        "Conditional Preference vs Dataset Prior",
    )
    _bar_plot("overprediction.png", result["metrics"]["overprediction"]["mean"], None, labels, "Over-Prediction Ratio")


def atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate language-model label preferences.")
    parser.add_argument("--inputs", type=str, help="Path to JSONL/JSON/CSV with fields: id,text,label")
    parser.add_argument("--labels", nargs="*", help="Explicit list of labels (optional).")
    parser.add_argument("--out", type=str, help="Path to write JSON results.")
    parser.add_argument("--report_md", type=str, default=None, help="Optional markdown report output path.")
    parser.add_argument("--plots_dir", type=str, default=None, help="Directory for optional PNG bar charts.")
    parser.add_argument("--backend", type=str, default="auto", help="LLM backend: auto, openai, or local.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--vote_samples", type=int, default=5, help="Samples per dataset prompt.")
    parser.add_argument("--prior_samples", type=int, default=20, help="Samples per prior prompt.")
    parser.add_argument("--prior_prompts", nargs="*", default=None, help="Content-free prompts.")
    parser.add_argument("--bootstrap_iters", type=int, default=200, help="Bootstrap iterations for CIs.")
    parser.add_argument("--calibrate", action="store_true", help="Apply calibration using P0.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for deterministic sampling.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_pipeline(args)
    except UserInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"Unexpected error: {tb}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
