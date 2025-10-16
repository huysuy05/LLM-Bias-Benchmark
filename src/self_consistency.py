"""Self-Consistency prompting implementation.

Self-consistency improves reasoning by:
1. Sampling multiple diverse reasoning paths (temperature > 0)
2. Aggregating answers via majority vote
3. Marginalizing over reasoning paths to find most consistent answer

Reference: Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2022)
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Callable, Any, Dict


class SelfConsistency:
    """Self-consistency decoding strategy for LLM inference."""

    def __init__(self, num_samples: int = 5, temperature: float = 0.7, aggregation: str = "majority"):
        """
        Initialize self-consistency sampler.

        Args:
            num_samples: Number of reasoning paths to sample (default: 5)
            temperature: Sampling temperature for diversity (default: 0.7)
            aggregation: Aggregation method - "majority" or "weighted" (default: "majority")
        """
        self.num_samples = num_samples
        self.temperature = temperature
        self.aggregation = aggregation

    def sample_and_aggregate(
        self,
        generate_fn: Callable,
        prompt: str,
        valid_labels: List[str],
        normalize_fn: Callable = None,
        **generate_kwargs
    ) -> str:
        """
        Generate multiple samples and aggregate via majority vote.

        Args:
            generate_fn: Function that takes (prompt, **kwargs) and returns generated text
            prompt: Input prompt
            valid_labels: List of valid label strings
            normalize_fn: Optional function to normalize raw generations to valid labels
            **generate_kwargs: Additional kwargs for generate_fn

        Returns:
            Most consistent label from valid_labels
        """
        samples = []

        # Generate multiple diverse reasoning paths
        for _ in range(self.num_samples):
            raw_output = generate_fn(prompt, **generate_kwargs)

            # Normalize to valid label if normalization function provided
            if normalize_fn:
                label = normalize_fn(raw_output)
            else:
                # Simple extraction: take first token and match to valid labels
                label = self._extract_label(raw_output, valid_labels)

            samples.append(label)

        # Aggregate via majority vote
        return self._aggregate(samples)

    def _extract_label(self, text: str, valid_labels: List[str]) -> str:
        """
        Extract label from generated text by matching against valid labels.

        Args:
            text: Generated text
            valid_labels: List of valid label strings

        Returns:
            Matched label or 'unknown'
        """
        if not text:
            return "unknown"

        text_lower = text.lower().strip()
        tokens = text_lower.split()

        # Check if any token matches a valid label
        for token in tokens:
            token_clean = token.strip('.,!?;:"\'')
            for label in valid_labels:
                if token_clean == label.lower():
                    return label

        return "unknown"

    def _aggregate(self, samples: List[str]) -> str:
        """
        Aggregate samples via majority vote.

        Args:
            samples: List of predicted labels

        Returns:
            Most common label
        """
        if not samples:
            return "unknown"

        if self.aggregation == "majority":
            # Simple majority vote
            counter = Counter(samples)
            most_common = counter.most_common(1)[0][0]
            return most_common

        elif self.aggregation == "weighted":
            # Weighted by frequency (same as majority for now, could add confidence scores)
            counter = Counter(samples)
            most_common = counter.most_common(1)[0][0]
            return most_common

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def get_sample_distribution(self, samples: List[str]) -> Dict[str, float]:
        """
        Get distribution of samples for analysis.

        Args:
            samples: List of predicted labels

        Returns:
            Dictionary mapping labels to their frequencies
        """
        counter = Counter(samples)
        total = len(samples)
        return {label: count / total for label, count in counter.items()}