"""Simple sequence generators used as reproducible M3 baselines."""

from __future__ import annotations

import numpy as np


DNA_ALPHABET = np.array(list("ACGT"))


class PositionFrequencyGenerator:
    """Generate DNA by independently sampling each position's base frequency.

    This deliberately simple model is an M3 baseline.  It preserves the
    position-specific base composition of the training set, but cannot learn
    dependencies between positions or target a requested expression strength.
    """

    def __init__(self, smoothing: float = 1.0):
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        self.smoothing = smoothing
        self.probabilities: np.ndarray | None = None

    def fit(self, sequences: np.ndarray) -> "PositionFrequencyGenerator":
        values = np.asarray(sequences).astype(str)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("Expected a non-empty one-dimensional sequence array")
        sequence_length = len(values[0])
        if any(len(sequence) != sequence_length or set(sequence) - set(DNA_ALPHABET) for sequence in values):
            raise ValueError("Sequences must have a shared length and use only A/C/G/T")
        index = {base: position for position, base in enumerate(DNA_ALPHABET)}
        counts = np.full((sequence_length, len(DNA_ALPHABET)), self.smoothing, dtype=np.float64)
        for sequence in values:
            counts[np.arange(sequence_length), [index[base] for base in sequence]] += 1.0
        self.probabilities = counts / counts.sum(axis=1, keepdims=True)
        return self

    def sample(self, count: int, seed: int) -> np.ndarray:
        if self.probabilities is None:
            raise RuntimeError("Generator has not been fitted")
        if count < 1:
            raise ValueError("count must be positive")
        rng = np.random.default_rng(seed)
        generated = np.empty(count, dtype=f"U{len(self.probabilities)}")
        for row in range(count):
            generated[row] = "".join(rng.choice(DNA_ALPHABET, p=distribution) for distribution in self.probabilities)
        return generated
