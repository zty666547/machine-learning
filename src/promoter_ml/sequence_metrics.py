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


def motif_metrics(generated: np.ndarray, reference: np.ndarray, k: int = 5, top_count: int = 20) -> dict[str, float]:
    """Measure preservation of frequent reference k-mers as candidate motifs."""
    generated_values = _validate(generated)
    reference_values = _validate(reference)
    if len(generated_values[0]) != len(reference_values[0]):
        raise ValueError("Generated and reference sequences must share a length")

    def counts(values: np.ndarray) -> dict[str, int]:
        result: dict[str, int] = {}
        for sequence in values:
            for position in range(len(sequence) - k + 1):
                token = sequence[position : position + k]
                result[token] = result.get(token, 0) + 1
        return result

    reference_counts = counts(reference_values)
    generated_counts = counts(generated_values)
    denominator_reference = len(reference_values) * (len(reference_values[0]) - k + 1)
    denominator_generated = len(generated_values) * (len(generated_values[0]) - k + 1)
    top_reference = [token for token, _ in sorted(reference_counts.items(), key=lambda item: (-item[1], item[0]))[:top_count]]
    absolute_differences = [abs(generated_counts.get(token, 0) / denominator_generated - reference_counts[token] / denominator_reference) for token in top_reference]
    top_generated = {token for token, _ in sorted(generated_counts.items(), key=lambda item: (-item[1], item[0]))[:top_count]}
    return {
        "motif_k": float(k),
        "top_reference_motif_mean_abs_frequency_error": float(np.mean(absolute_differences)),
        "top_motif_overlap_fraction": float(len(set(top_reference).intersection(top_generated)) / top_count),
    }


def diversity_metrics(generated: np.ndarray, reference: np.ndarray, pairwise_limit: int = 250) -> dict[str, float]:
    """Measure within-sample diversity and similarity to the nearest training example."""
    generated_values = _validate(generated)
    reference_values = _validate(reference)
    if len(generated_values[0]) != len(reference_values[0]):
        raise ValueError("Generated and reference sequences must share a length")
    alphabet = {base: index for index, base in enumerate(DNA_ALPHABET)}
    encode = lambda values: np.array([[alphabet[base] for base in sequence] for sequence in values], dtype=np.int8)
    generated_encoded = encode(generated_values)
    reference_encoded = encode(reference_values)
    subset = generated_encoded[: min(pairwise_limit, len(generated_encoded))]
    if len(subset) < 2:
        mean_pairwise_distance = 0.0
    else:
        differences = np.mean(subset[:, None, :] != subset[None, :, :], axis=2)
        mean_pairwise_distance = float(differences[np.triu_indices(len(subset), k=1)].mean())
    nearest_similarities = []
    for start in range(0, len(generated_encoded), 50):
        batch = generated_encoded[start : start + 50]
        similarities = np.mean(batch[:, None, :] == reference_encoded[None, :, :], axis=2)
        nearest_similarities.extend(similarities.max(axis=1).tolist())
    return {
        "mean_pairwise_hamming_distance": mean_pairwise_distance,
        "mean_nearest_training_similarity": float(np.mean(nearest_similarities)),
        "max_nearest_training_similarity": float(np.max(nearest_similarities)),
    }


def sequence_distribution_distance(first: np.ndarray, second: np.ndarray, k: int = 3) -> dict[str, float]:
    """Compare composition between two generated sequence groups."""
    first_values = _validate(first)
    second_values = _validate(second)
    if len(first_values[0]) != len(second_values[0]):
        raise ValueError("Sequence groups must share a length")
    first_position = _base_frequency_by_position(first_values)
    second_position = _base_frequency_by_position(second_values)
    first_kmers = _kmer_distribution(first_values, k=k)
    second_kmers = _kmer_distribution(second_values, k=k)
    midpoint = (first_kmers + second_kmers) / 2
    js_divergence = 0.5 * np.sum(first_kmers * np.log(first_kmers / midpoint)) + 0.5 * np.sum(second_kmers * np.log(second_kmers / midpoint))
    return {
        "mean_position_base_l1": float(np.mean(np.abs(first_position - second_position).sum(axis=1))),
        f"kmer_{k}_js_divergence": float(js_divergence),
    }
