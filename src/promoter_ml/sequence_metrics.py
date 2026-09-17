"""Data-free sequence quality metrics shared by M3 generator experiments."""

from __future__ import annotations

import numpy as np

DNA_ALPHABET = "ACGT"


def _validate(sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(sequences).astype(str)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Expected a non-empty one-dimensional sequence array")
    length = len(values[0])
    if any(len(sequence) != length or set(sequence) - set(DNA_ALPHABET) for sequence in values):
        raise ValueError("Sequences must have equal length and use only A/C/G/T")
    return values


def _base_frequency_by_position(sequences: np.ndarray) -> np.ndarray:
    values = _validate(sequences)
    frequencies = np.zeros((len(values[0]), 4), dtype=np.float64)
    for position in range(len(values[0])):
        for base_index, base in enumerate(DNA_ALPHABET):
            frequencies[position, base_index] = np.mean([sequence[position] == base for sequence in values])
    return frequencies


def _kmer_distribution(sequences: np.ndarray, k: int = 3) -> np.ndarray:
    values = _validate(sequences)
    vocabulary = ["".join(chars) for chars in __import__("itertools").product(DNA_ALPHABET, repeat=k)]
    index = {token: position for position, token in enumerate(vocabulary)}
    counts = np.full(len(vocabulary), 1e-12, dtype=np.float64)
    for sequence in values:
        for position in range(len(sequence) - k + 1):
            counts[index[sequence[position : position + k]]] += 1.0
    return counts / counts.sum()


def generation_metrics(generated: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compare generated sequences with the training distribution only."""
    generated_values = _validate(generated)
    reference_values = _validate(reference)
    if len(generated_values[0]) != len(reference_values[0]):
        raise ValueError("Generated and reference sequences must share a length")
    gc = lambda sequence: (sequence.count("G") + sequence.count("C")) / len(sequence)
    generated_gc = np.array([gc(sequence) for sequence in generated_values])
    reference_gc = np.array([gc(sequence) for sequence in reference_values])
    generated_position = _base_frequency_by_position(generated_values)
    reference_position = _base_frequency_by_position(reference_values)
    generated_kmers = _kmer_distribution(generated_values)
    reference_kmers = _kmer_distribution(reference_values)
    midpoint = (generated_kmers + reference_kmers) / 2
    js_divergence = 0.5 * np.sum(generated_kmers * np.log(generated_kmers / midpoint)) + 0.5 * np.sum(reference_kmers * np.log(reference_kmers / midpoint))
    reference_set = set(reference_values.tolist())
    unique_count = len(set(generated_values.tolist()))
    return {
        "valid_fraction": 1.0,
        "unique_fraction": float(unique_count / len(generated_values)),
        "novel_fraction": float(np.mean([sequence not in reference_set for sequence in generated_values])),
        "generated_gc_mean": float(generated_gc.mean()),
        "reference_gc_mean": float(reference_gc.mean()),
        "generated_gc_std": float(generated_gc.std()),
        "reference_gc_std": float(reference_gc.std()),
        "mean_position_base_l1": float(np.mean(np.abs(generated_position - reference_position).sum(axis=1))),
        "kmer_3_js_divergence": float(js_divergence),
    }
