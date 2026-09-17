"""Loading, validating and featurizing promoter sequences."""

from __future__ import annotations

import csv
from itertools import combinations, product
from pathlib import Path
import numpy as np

DNA_ALPHABET = "ACGT"


def load_promoter_arrays(
    data_directory: str | Path,
    sequence_file: str = "promoter.npy",
    label_file: str = "gene_expression.npy",
    sequence_length: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the course arrays and enforce the shared data contract."""
    data_dir = Path(data_directory)
    sequences = np.load(data_dir / sequence_file, allow_pickle=False).astype(str)
    strengths = np.load(data_dir / label_file, allow_pickle=False).astype(np.float64)
    sequences = np.char.upper(sequences)

    if sequences.ndim != 1 or strengths.ndim != 1:
        raise ValueError("Sequences and strengths must both be one-dimensional arrays")
    if len(sequences) != len(strengths):
        raise ValueError("Sequence and strength arrays have different lengths")
    if len(sequences) == 0:
        raise ValueError("The dataset is empty")
    if np.any(strengths <= 0) or not np.all(np.isfinite(strengths)):
        raise ValueError("Strength labels must be finite positive values for log10")

    invalid = [seq for seq in sequences if len(seq) != sequence_length or set(seq) - set(DNA_ALPHABET)]
    if invalid:
        raise ValueError(f"Found {len(invalid)} invalid sequences; first example: {invalid[0]!r}")
    return sequences, strengths


def split_indices(
    sample_count: int,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic development split.

    This random split is used only to make the initial pipeline runnable. The
    formal experiments will replace it with a similarity-cluster split.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    train_end = int(sample_count * train_ratio)
    validation_end = train_end + int(sample_count * validation_ratio)
    return indices[:train_end], indices[train_end:validation_end], indices[validation_end:]


class _UnionFind:
    """Small disjoint-set structure used for similarity components."""

    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = int(self.parent[index])
        return index

    def union(self, left: int, right: int) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def similarity_cluster_labels(
    sequences: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Cluster equal-length sequences by aligned Hamming similarity.

    Two sequences are connected when their aligned identity is at least
    ``threshold``. Candidate generation uses five ten-base blocks and all
    eight-position masks within each block. Any pair with at most ten
    mismatches must share at least one such mask, so this remains exact for
    the project's 80%, 85% and 90% thresholds without an all-pairs scan.
    """
    values = np.asarray(sequences).astype(str)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Expected a non-empty one-dimensional sequence array")
    length = len(values[0])
    if length != 50 or any(len(sequence) != length for sequence in values):
        raise ValueError("Similarity clustering currently requires 50 bp sequences")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")

    max_mismatches = int(np.floor((1.0 - threshold) * length + 1e-9))
    if max_mismatches > 10:
        raise ValueError("The exact candidate scheme supports thresholds of 80% or higher")

    union_find = _UnionFind(len(values))
    checked_pairs = 0
    similarity_edges = 0
    masks = tuple(combinations(range(10), 8))

    for block_start in range(0, length, 10):
        block_values = [sequence[block_start : block_start + 10] for sequence in values]
        for mask in masks:
            buckets: dict[str, list[int]] = {}
            for index, block in enumerate(block_values):
                signature = "".join(block[position] for position in mask)
                buckets.setdefault(signature, []).append(index)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                for left, right in combinations(members, 2):
                    checked_pairs += 1
                    mismatches = sum(a != b for a, b in zip(values[left], values[right]))
                    if mismatches <= max_mismatches and union_find.union(left, right):
                        similarity_edges += 1

    roots = np.array([union_find.find(index) for index in range(len(values))])
    root_to_label: dict[int, int] = {}
    labels = np.empty(len(values), dtype=np.int64)
    for index, root in enumerate(roots):
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[index] = root_to_label[root]

    sizes = np.bincount(labels)
    return labels, {
        "threshold": threshold,
        "max_mismatches": max_mismatches,
        "cluster_count": int(len(sizes)),
        "largest_cluster": int(sizes.max()),
        "singleton_clusters": int(np.sum(sizes == 1)),
        "similarity_edges": similarity_edges,
        "candidate_pairs_checked": checked_pairs,
    }


def assign_clusters_to_splits(
    cluster_labels: np.ndarray,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> np.ndarray:
    """Assign entire similarity components to train, validation and test."""
    labels = np.asarray(cluster_labels, dtype=np.int64)
    ratios = np.array([train_ratio, validation_ratio, test_ratio], dtype=np.float64)
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must sum to one")

    members: dict[int, list[int]] = {}
    for index, cluster in enumerate(labels):
        members.setdefault(int(cluster), []).append(index)
    ordered_clusters = sorted(members.items(), key=lambda item: (-len(item[1]), item[0]))
    targets = ratios * len(labels)
    counts = np.zeros(3, dtype=np.int64)
    assignment = np.empty(len(labels), dtype="U10")
    names = ("train", "validation", "test")

    for _, indices in ordered_clusters:
        size = len(indices)
        penalties = []
        for split_index in range(3):
            proposed = counts.copy()
            proposed[split_index] += size
            penalties.append(float(np.sum(((proposed - targets) / targets) ** 2)))
        chosen = min(range(3), key=lambda split_index: (penalties[split_index], counts[split_index]))
        assignment[indices] = names[chosen]
        counts[chosen] += size
    return assignment


def load_fixed_split_indices(
    split_file: str | Path,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate a versioned sample-index split definition."""
    rows: list[dict[str, str]]
    with Path(split_file).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != sample_count:
        raise ValueError(f"Split file has {len(rows)} rows but expected {sample_count}")
    indices = np.array([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if set(indices.tolist()) != set(range(sample_count)):
        raise ValueError("Split file must contain each sample_index exactly once")
    split_values = np.array([row["split"] for row in rows])
    expected = {"train", "validation", "test"}
    if set(split_values).difference(expected):
        raise ValueError("Split file contains an unknown split name")
    return tuple(indices[split_values == name] for name in ("train", "validation", "test"))


class KmerFeaturizer:
    """Convert DNA sequences into normalized k-mer counts and optional GC content."""

    def __init__(self, k_min: int = 1, k_max: int = 3, include_gc: bool = True):
        if k_min < 1 or k_max < k_min:
            raise ValueError("Require 1 <= k_min <= k_max")
        self.k_min = k_min
        self.k_max = k_max
        self.include_gc = include_gc
        self.vocabulary = [
            "".join(chars)
            for k in range(k_min, k_max + 1)
            for chars in product(DNA_ALPHABET, repeat=k)
        ]
        self.index = {token: position for position, token in enumerate(self.vocabulary)}

    @property
    def feature_names(self) -> list[str]:
        names = [f"kmer_{token}" for token in self.vocabulary]
        return names + (["gc_fraction"] if self.include_gc else [])

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        features = np.zeros((len(sequences), len(self.feature_names)), dtype=np.float64)
        for row, sequence in enumerate(sequences):
            for k in range(self.k_min, self.k_max + 1):
                denominator = len(sequence) - k + 1
                for position in range(denominator):
                    token = sequence[position : position + k]
                    features[row, self.index[token]] += 1.0 / denominator
            if self.include_gc:
                features[row, -1] = (sequence.count("G") + sequence.count("C")) / len(sequence)
        return features
