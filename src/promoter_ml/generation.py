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


class ConditionalPositionFrequencyGenerator:
    """One position-frequency generator per discrete expression-strength label."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.generators: dict[str, PositionFrequencyGenerator] = {}

    def fit(self, sequences: np.ndarray, conditions: np.ndarray) -> "ConditionalPositionFrequencyGenerator":
        values = np.asarray(sequences).astype(str)
        labels = np.asarray(conditions).astype(str)
        if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels):
            raise ValueError("Expected matching one-dimensional sequences and conditions")
        self.generators = {}
        for label in sorted(set(labels.tolist())):
            selected = values[labels == label]
            if len(selected) == 0:
                raise ValueError(f"Condition {label!r} has no training sequences")
            self.generators[label] = PositionFrequencyGenerator(smoothing=self.smoothing).fit(selected)
        return self

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(sorted(self.generators))

    def sample(self, condition: str, count: int, seed: int) -> np.ndarray:
        if condition not in self.generators:
            raise ValueError(f"Unknown condition {condition!r}; expected one of {self.conditions}")
        return self.generators[condition].sample(count, seed)

    def condition_distance(self, first: str, second: str) -> float:
        """Mean L1 distance between two learned position/base distributions."""
        if first not in self.generators or second not in self.generators:
            raise ValueError("Both conditions must be present in the fitted generator")
        first_distribution = self.generators[first].probabilities
        second_distribution = self.generators[second].probabilities
        if first_distribution is None or second_distribution is None:
            raise RuntimeError("Generator has not been fitted")
        return float(np.mean(np.abs(first_distribution - second_distribution).sum(axis=1)))


class ConditionalAutoregressiveGenerator:
    """Position-aware Markov generator conditioned on a discrete strength label.

    It implements the autoregressive factorization by sampling each base from
    the preceding ``order`` generated bases, the sequence position and the
    requested condition.  Sparse contexts fall back to the condition-specific
    position distribution.
    """

    def __init__(self, order: int = 3, smoothing: float = 0.1):
        if order < 1:
            raise ValueError("order must be positive")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.order = order
        self.smoothing = smoothing
        self.context_probabilities: dict[tuple[str, int, str], np.ndarray] = {}
        self.fallback_generators: dict[str, PositionFrequencyGenerator] = {}
        self.sequence_length: int | None = None

    def fit(self, sequences: np.ndarray, conditions: np.ndarray) -> "ConditionalAutoregressiveGenerator":
        values = np.asarray(sequences).astype(str)
        labels = np.asarray(conditions).astype(str)
        if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels) or len(values) == 0:
            raise ValueError("Expected non-empty matching one-dimensional sequences and conditions")
        self.sequence_length = len(values[0])
        if any(len(sequence) != self.sequence_length or set(sequence) - set(DNA_ALPHABET) for sequence in values):
            raise ValueError("Sequences must have a shared length and use only A/C/G/T")
        self.fallback_generators = ConditionalPositionFrequencyGenerator().fit(values, labels).generators
        base_index = {base: index for index, base in enumerate(DNA_ALPHABET)}
        counts: dict[tuple[str, int, str], np.ndarray] = {}
        for sequence, label in zip(values, labels):
            padded = "^" * self.order + sequence
            for position, base in enumerate(sequence):
                context = padded[position : position + self.order]
                key = (str(label), position, context)
                if key not in counts:
                    counts[key] = np.full(len(DNA_ALPHABET), self.smoothing, dtype=np.float64)
                counts[key][base_index[base]] += 1.0
        self.context_probabilities = {key: value / value.sum() for key, value in counts.items()}
        return self

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(sorted(self.fallback_generators))

    def sample(self, condition: str, count: int, seed: int) -> np.ndarray:
        if self.sequence_length is None or condition not in self.fallback_generators:
            raise RuntimeError("Generator has not been fitted for this condition")
        if count < 1:
            raise ValueError("count must be positive")
        fallback = self.fallback_generators[condition].probabilities
        if fallback is None:
            raise RuntimeError("Fallback generator has not been fitted")
        rng = np.random.default_rng(seed)
        generated = np.empty(count, dtype=f"U{self.sequence_length}")
        for row in range(count):
            prefix = "^" * self.order
            sequence = []
            for position in range(self.sequence_length):
                context = prefix[-self.order :]
                probabilities = self.context_probabilities.get((condition, position, context), fallback[position])
                base = str(rng.choice(DNA_ALPHABET, p=probabilities))
                sequence.append(base)
                prefix += base
            generated[row] = "".join(sequence)
        return generated


class ContinuousConditionalAutoregressiveGenerator:
    """Position-aware Markov generator with a continuous numeric condition.

    Training conditions are divided into equal-frequency bins. At sampling
    time, transition distributions from the two nearest bin centres are mixed
    linearly, allowing intermediate numeric conditions while retaining the
    preceding-base context used by the discrete autoregressive baseline.
    """

    def __init__(self, order: int = 3, condition_bins: int = 5, smoothing: float = 0.1):
        if order < 1 or condition_bins < 2 or smoothing <= 0:
            raise ValueError("order must be positive, condition_bins at least 2, and smoothing positive")
        self.order = order
        self.condition_bins = condition_bins
        self.smoothing = smoothing
        self.sequence_length: int | None = None
        self.bin_centres: np.ndarray | None = None
        self.context_probabilities: dict[tuple[int, int, str], np.ndarray] = {}
        self.fallback_probabilities: np.ndarray | None = None

    def fit(self, sequences: np.ndarray, conditions: np.ndarray) -> "ContinuousConditionalAutoregressiveGenerator":
        values = np.asarray(sequences).astype(str)
        numeric_conditions = np.asarray(conditions, dtype=np.float64)
        if values.ndim != 1 or numeric_conditions.ndim != 1 or len(values) != len(numeric_conditions) or len(values) == 0:
            raise ValueError("Expected non-empty matching sequences and numeric conditions")
        if not np.all(np.isfinite(numeric_conditions)):
            raise ValueError("Conditions must be finite")
        self.sequence_length = len(values[0])
        if any(len(sequence) != self.sequence_length or set(sequence) - set(DNA_ALPHABET) for sequence in values):
            raise ValueError("Sequences must have a shared length and use only A/C/G/T")
        edges = np.quantile(numeric_conditions, np.linspace(0.0, 1.0, self.condition_bins + 1))
        bin_ids = np.digitize(numeric_conditions, edges[1:-1], right=True)
        self.bin_centres = np.array([numeric_conditions[bin_ids == bin_id].mean() for bin_id in range(self.condition_bins)])
        base_index = {base: index for index, base in enumerate(DNA_ALPHABET)}
        fallback_counts = np.full((self.condition_bins, self.sequence_length, len(DNA_ALPHABET)), self.smoothing, dtype=np.float64)
        context_counts: dict[tuple[int, int, str], np.ndarray] = {}
        for sequence, bin_id in zip(values, bin_ids):
            padded = "^" * self.order + sequence
            for position, base in enumerate(sequence):
                fallback_counts[bin_id, position, base_index[base]] += 1.0
                context = padded[position : position + self.order]
                key = (int(bin_id), position, context)
                if key not in context_counts:
                    context_counts[key] = np.full(len(DNA_ALPHABET), self.smoothing, dtype=np.float64)
                context_counts[key][base_index[base]] += 1.0
        self.fallback_probabilities = fallback_counts / fallback_counts.sum(axis=2, keepdims=True)
        self.context_probabilities = {key: counts / counts.sum() for key, counts in context_counts.items()}
        return self

    def _nearest_bins(self, condition: float) -> tuple[int, int, float]:
        if self.bin_centres is None:
            raise RuntimeError("Generator has not been fitted")
        if condition <= self.bin_centres[0]:
            return 0, 0, 0.0
        if condition >= self.bin_centres[-1]:
            last = len(self.bin_centres) - 1
            return last, last, 0.0
        upper = int(np.searchsorted(self.bin_centres, condition, side="right"))
        lower = upper - 1
        weight = float((condition - self.bin_centres[lower]) / (self.bin_centres[upper] - self.bin_centres[lower]))
        return lower, upper, weight

    def sample(self, condition: float, count: int, seed: int) -> np.ndarray:
        if self.sequence_length is None or self.fallback_probabilities is None:
            raise RuntimeError("Generator has not been fitted")
        if count < 1 or not np.isfinite(condition):
            raise ValueError("count must be positive and condition finite")
        lower, upper, weight = self._nearest_bins(float(condition))
        rng = np.random.default_rng(seed)
        generated = np.empty(count, dtype=f"U{self.sequence_length}")
        for row in range(count):
            prefix = "^" * self.order
            sequence = []
            for position in range(self.sequence_length):
                context = prefix[-self.order :]
                lower_probability = self.context_probabilities.get((lower, position, context), self.fallback_probabilities[lower, position])
                upper_probability = self.context_probabilities.get((upper, position, context), self.fallback_probabilities[upper, position])
                probabilities = (1.0 - weight) * lower_probability + weight * upper_probability
                base = str(rng.choice(DNA_ALPHABET, p=probabilities))
                sequence.append(base)
                prefix += base
            generated[row] = "".join(sequence)
        return generated
