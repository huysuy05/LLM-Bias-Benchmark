import math
from collections import Counter
from typing import Callable, Dict, List, Sequence

import numpy as np


def _logsumexp(arr: np.ndarray) -> float:
    """Numerically stable log-sum-exp for 1D arrays."""
    m = float(np.max(arr))
    return m + math.log(float(np.sum(np.exp(arr - m))))


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute log-softmax for 1D logits."""
    return logits - _logsumexp(logits)


def _label_logprobs(
    log_probs: np.ndarray,
    answer_token_ids: Dict[str, Sequence[int]],
) -> Dict[str, float]:
    """Aggregate per-label log-probability from token-level log-probs.

    Supports labels that map to multiple tokens by summing token log-probs.
    """
    scores: Dict[str, float] = {}
    for label, token_ids in answer_token_ids.items():
        if not token_ids:
            scores[label] = float("-inf")
            continue
        total = 0.0
        for tid in token_ids:
            total += float(log_probs[tid])
        scores[label] = total
    return scores


class AntiMajoritySelfConsistency:
    """Self-consistency with dynamic anti-majority penalties."""

    def __init__(
        self,
        num_samples: int = 25,
        temperature: float = 0.7,
        lambda_penalty: float = 0.5,
        aggregation: str = "majority",
    ) -> None:
        self.num_samples = num_samples
        self.temperature = temperature
        self.lambda_penalty = lambda_penalty
        self.aggregation = aggregation

    def sample_and_aggregate(
        self,
        logit_fn: Callable[[str], np.ndarray],
        prompt: str,
        answer_token_ids: Dict[str, Sequence[int]],
    ) -> str:
        """Run anti-majority self-consistency.

        Args:
            logit_fn: Callable returning next-token logits for the prompt.
                      Shape must be (vocab,) or (batch, vocab); if 2D, the
                      last row is used.
            prompt: Input prompt string.
            answer_token_ids: Mapping label -> sequence of token ids that
                               realize that label (pre-tokenized).

        Returns:
            Aggregated label string.
        """
        counts: Counter = Counter({label: 0 for label in answer_token_ids})
        samples: List[str] = []

        for _ in range(self.num_samples):
            logits = logit_fn(prompt)
            if logits.ndim == 2:
                logits = logits[-1]
            if logits.ndim != 1:
                raise ValueError("logit_fn must return shape (vocab,) or (batch, vocab)")

            temp = max(self.temperature, 1e-6)
            log_probs = _log_softmax(logits / temp)
            base_scores = _label_logprobs(log_probs, answer_token_ids)

            adjusted_scores = {}
            for label, base in base_scores.items():
                adjusted_scores[label] = base - self.lambda_penalty * counts[label]

            labels = list(adjusted_scores.keys())
            adj_arr = np.array([adjusted_scores[l] for l in labels], dtype=np.float64)
            probs = np.exp(adj_arr - _logsumexp(adj_arr))
            choice = np.random.choice(len(labels), p=probs)
            picked = labels[choice]

            samples.append(picked)
            counts[picked] += 1

        return self._aggregate(samples)

    def _aggregate(self, samples: List[str]) -> str:
        if not samples:
            return "unknown"
        if self.aggregation == "majority":
            counter = Counter(samples)
            return counter.most_common(1)[0][0]
        if self.aggregation == "weighted":
            counter = Counter(samples)
            return counter.most_common(1)[0][0]
        raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def get_sample_distribution(self, samples: List[str]) -> Dict[str, float]:
        counter = Counter(samples)
        total = len(samples)
        if total == 0:
            return {}
        return {label: count / total for label, count in counter.items()}
