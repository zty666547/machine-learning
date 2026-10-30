"""Position-aware descriptive metrics for fixed-length DNA sequences."""
from __future__ import annotations
import itertools
import numpy as np

ALPHABET = "ACGT"


def encode_sequences(sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(sequences)
    if values.ndim != 1 or not len(values):
        raise ValueError("Expected nonempty one-dimensional sequences")
    mapping = {base: i for i, base in enumerate(ALPHABET)}
    length = len(str(values[0]))
    if not length:
        raise ValueError("Empty sequence")
    encoded = np.empty((len(values), length), dtype=np.uint8)
    for i, value in enumerate(values):
        value = str(value).upper()
        if len(value) != length or set(value) - set(ALPHABET):
            raise ValueError("Sequences must have equal lengths and use A/C/G/T")
        encoded[i] = [mapping[base] for base in value]
    return encoded


def position_kmer_frequency(encoded: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(encoded)
    if x.ndim != 2 or not len(x) or k < 1 or k > x.shape[1] or np.any((x < 0) | (x > 3)):
        raise ValueError("Invalid encoded sequences or k")
    positions = x.shape[1] - k + 1
    codes = np.zeros((len(x), positions), dtype=np.int64)
    for offset in range(k):
        codes = codes * 4 + x[:, offset:offset + positions]
    return np.stack([np.bincount(codes[:, p], minlength=4 ** k) for p in range(positions)]) / len(x)


def position_js(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Frequency matrices must have the same two-dimensional shape")
    midpoint = (reference + candidate) / 2
    def contribution(values):
        out = np.zeros_like(values, dtype=np.float64)
        mask = values > 0
        out[mask] = values[mask] * np.log(values[mask] / midpoint[mask])
        return out.sum(axis=1)
    return (contribution(reference) + contribution(candidate)) / 2


def top_kmers(frequencies: np.ndarray, k: int, count: int = 5) -> list[tuple[int, str]]:
    if frequencies.shape[1] != 4 ** k or count < 1:
        raise ValueError("Bad frequency shape or count")
    scores = frequencies.mean(axis=0)
    indices = np.lexsort((np.arange(len(scores)), -scores))[:count]
    tokens = ["".join(x) for x in itertools.product(ALPHABET, repeat=k)]
    return [(int(i), tokens[i]) for i in indices]


def match_window(encoded: np.ndarray, motif: str, starts: list[int], max_mismatches: int = 1) -> np.ndarray:
    x = np.asarray(encoded)
    motif_code = encode_sequences(np.array([motif]))[0]
    if x.ndim != 2 or not len(x) or not starts or max_mismatches < 0:
        raise ValueError("Invalid sequences, starts or mismatch limit")
    if any(s < 0 or s + len(motif_code) > x.shape[1] for s in starts):
        raise ValueError("Window outside sequence")
    match = np.zeros(len(x), dtype=bool)
    for s in starts:
        match |= np.count_nonzero(x[:, s:s + len(motif_code)] != motif_code, axis=1) <= max_mismatches
    return match


def motif_scan(encoded: np.ndarray, shift: int = 0, radius: int = 1) -> dict[str, float]:
    # Hypothesis: sequence covers -50..-1 relative to +1 TSS, 5' to 3'.
    # Literature consensus starts: -35 TTGACA and -12 TATAAT.
    if radius < 0:
        raise ValueError("radius must not be negative")
    starts_35 = list(range(15 + shift - radius, 15 + shift + radius + 1))
    starts_10 = list(range(38 + shift - radius, 38 + shift + radius + 1))
    a = match_window(encoded, "TTGACA", starts_35)
    b = match_window(encoded, "TATAAT", starts_10)
    return {"minus35_hit_fraction": float(a.mean()), "minus10_hit_fraction": float(b.mean()),
            "both_hit_fraction": float((a & b).mean())}


def shuffle_within_sequence(encoded: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(encoded)
    return np.stack([rng.permutation(row) for row in values])
