"""
Anti-Majority Self-Consistency (AMSC) for MLX-LM — FULL VERSION

Fixes:
1) Multi-token labels are scored correctly with teacher-forced log P(label | prompt).
2) Includes sanity checks to catch label-tokenization bugs (e.g., sci/tech never predicted).
3) Uses seeded RNG.
4) Supports different aggregations:
   - majority: majority vote over sampled labels
   - base_argmax: choose label with best unpenalized base score (recommended when forcing diversity)

Usage:
  pred, debug = amsc.predict(model, tokenizer, prompt, label_texts, return_debug=True)

Where label_texts maps canonical label -> exact surface form the model should output, e.g.:
  {
    "world": " world",
    "sports": " sports",
    "business": " business",
    "sci/tech": " sci/tech",
  }
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import mlx.core as mx


def _logsumexp_1d(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


def _log_softmax_1d(a: np.ndarray) -> np.ndarray:
    return a - _logsumexp_1d(a)


def _softmax_1d(a: np.ndarray) -> np.ndarray:
    z = a - _logsumexp_1d(a)
    return np.exp(z)

def _unwrap_logits(out: Any) -> mx.array:
    """
    Normalize MLX model output to logits MLX array.
    Common return types:
      - dict with "logits"
      - object with .logits
      - tuple/list (logits, cache)
      - raw logits
    Expected logits shape is (B, T, V).
    """
    if isinstance(out, dict):
        logits = out.get("logits", None)
        if logits is None:
            logits = out
    elif hasattr(out, "logits"):
        logits = out.logits
    elif isinstance(out, (list, tuple)) and len(out) > 0:
        logits = out[0]
    else:
        logits = out

    if not isinstance(logits, mx.array):
        logits = mx.array(logits)
    return logits


def build_answer_token_ids(
    tokenizer: Any,
    label_texts: Dict[str, str],
) -> Dict[str, List[int]]:
    """
    Build label -> token ids mapping from EXACT label surface forms.

    IMPORTANT:
      label_texts values should match what the model emits at the label boundary,
      often with a leading space, e.g. " world" not "world".
    """
    answer_token_ids: Dict[str, List[int]] = {}
    for label, surface in label_texts.items():
        ids = tokenizer.encode(surface, add_special_tokens=False)
        answer_token_ids[label] = list(ids)
    return answer_token_ids


def print_label_token_sanity_check(
    tokenizer: Any,
    answer_token_ids: Dict[str, Sequence[int]],
) -> None:
    """
    Prints tokens and decoded surfaces so you can catch issues like:
      - empty token list for a label
      - decoded surface not matching expected label string
    """
    print("\n=== LABEL TOKENS CHECK ===")
    for label, ids in answer_token_ids.items():
        decoded = tokenizer.decode(list(ids)) if ids else ""
        print(f"{label:>10} | ids={list(ids)} | len={len(ids)} | decoded={repr(decoded)}")
    print("=========================\n")


def teacher_forced_label_logprob(
    model: Any,
    prompt_ids: Sequence[int],
    label_ids: Sequence[int],
) -> float:
    """
    Compute exact log P(label_ids | prompt_ids) for a causal LM via teacher forcing.

    If full_ids = prompt_ids + label_ids (length T),
    logits[0, t, :] predicts token at position t+1 in full_ids.

    To score token at position j, use logits at position j-1.
    """
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids is empty")
    if len(label_ids) == 0:
        return float("-inf")

    full_ids = list(prompt_ids) + list(label_ids)
    x = mx.array([full_ids], dtype=mx.int32)

    out = model(x)
    logits = _unwrap_logits(out)              # (1, T, V) typically
    logits = logits.astype(mx.float32)        # avoid NumPy buffer issues (bf16)
    mx.eval(logits)

    logits_np = np.asarray(logits)[0]         # (T, V)
    prompt_len = len(prompt_ids)

    lp = 0.0
    for i, tid in enumerate(label_ids):
        j = prompt_len + i        # position in full_ids for this label token
        prev_pos = j - 1          # logits index that predicts token at j
        step_logits = logits_np[prev_pos]           # (V,)
        step_logprobs = _log_softmax_1d(step_logits)
        lp += float(step_logprobs[int(tid)])

    return lp


def compute_base_label_scores(
    model: Any,
    tokenizer: Any,
    prompt: str,
    answer_token_ids: Dict[str, Sequence[int]],
    add_special_tokens: bool = True,
) -> Dict[str, float]:
    """
    base_scores[label] = log P(label_tokens | prompt), multi-token correct.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if len(prompt_ids) == 0:
        raise ValueError("Tokenized prompt is empty. Check tokenizer/prompt formatting.")

    scores: Dict[str, float] = {}
    for label, label_ids in answer_token_ids.items():
        scores[label] = teacher_forced_label_logprob(model, prompt_ids, label_ids)
    return scores


@dataclass
class AMSCConfig:
    num_samples: int = 25
    temperature: float = 0.7
    lambda_penalty: float = 0.5
    aggregation: str = "base_argmax"  # "majority" or "base_argmax"
    seed: int = 0
    add_special_tokens: bool = True


class AntiMajoritySelfConsistencyMLX:
    def __init__(self, cfg: Optional[AMSCConfig] = None) -> None:
        self.cfg = cfg or AMSCConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def predict(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        label_texts: Dict[str, str],
        return_debug: bool = False,
        sanity_check: bool = False,
    ):
        """
        Args:
          label_texts: canonical label -> exact surface string to score
                       (include leading space/newline if your prompt requires it)
        Returns:
          pred_label
          optionally debug dict
        """
        answer_token_ids = build_answer_token_ids(tokenizer, label_texts)

        if sanity_check:
            print_label_token_sanity_check(tokenizer, answer_token_ids)

        labels = list(answer_token_ids.keys())
        if not labels:
            raise ValueError("No labels provided in label_texts")

        base_scores = compute_base_label_scores(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            answer_token_ids=answer_token_ids,
            add_special_tokens=self.cfg.add_special_tokens,
        )

        # If a label is impossible (empty tokenization or -inf), this will reveal it
        # and avoids weird sampling.
        for l, s in base_scores.items():
            if not np.isfinite(s):
                # keep it, but it'll effectively never be sampled
                pass

        counts = Counter({l: 0 for l in labels})
        samples: List[str] = []

        temp = max(float(self.cfg.temperature), 1e-6)
        lam = float(self.cfg.lambda_penalty)

        for _ in range(int(self.cfg.num_samples)):
            adjusted = np.array(
                [(base_scores[l] / temp) - lam * counts[l] for l in labels],
                dtype=np.float64,
            )
            probs = _softmax_1d(adjusted)
            idx = int(self.rng.choice(len(labels), p=probs))
            picked = labels[idx]
            samples.append(picked)
            counts[picked] += 1

        pred = self._aggregate(samples, base_scores)

        if not return_debug:
            return pred

        debug = {
            "pred": pred,
            "samples": samples,
            "counts": dict(counts),
            "base_scores": base_scores,
            "answer_token_ids": {k: list(v) for k, v in answer_token_ids.items()},
            "label_texts": label_texts,
            "cfg": vars(self.cfg),
        }
        return pred, debug

    def _aggregate(self, samples: List[str], base_scores: Dict[str, float]) -> str:
        if not samples:
            return "unknown"

        if self.cfg.aggregation == "majority":
            return Counter(samples).most_common(1)[0][0]
        
        if self.cfg.aggregation == "base_argmax":
            return max(base_scores.items(), key=lambda kv: kv[1])[0]

        raise ValueError(f"Unknown aggregation: {self.cfg.aggregation}")
