"""Aligned Hamming-neighbour screening for equal-length DNA sequences."""
from __future__ import annotations
from collections import defaultdict


def reverse_complement(sequence: str) -> str:
    sequence = sequence.upper()
    if set(sequence) - set("ACGT"):
        raise ValueError("DNA sequence must contain only A/C/G/T")
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


class AlignedNeighborIndex:
    """Find all references within a substitution threshold on either strand.

    k+1 fixed blocks guarantee at least one exact block if Hamming distance <= k.
    This does not find indels or shifted local alignments.
    """
    def __init__(self, sequences: list[str], max_mismatches: int = 5):
        if not sequences or max_mismatches < 0:
            raise ValueError("Expected nonempty reference and nonnegative threshold")
        self.sequences = [sequence.upper() for sequence in sequences]
        self.length = len(self.sequences[0])
        if max_mismatches >= self.length or any(len(s) != self.length or set(s) - set("ACGT") for s in self.sequences):
            raise ValueError("References must be equal-length A/C/G/T and threshold smaller than length")
        self.threshold = max_mismatches
        self.blocks = [(i * self.length // (max_mismatches + 1), (i + 1) * self.length // (max_mismatches + 1))
                       for i in range(max_mismatches + 1)]
        self.index = defaultdict(set)
        for i, sequence in enumerate(self.sequences):
            for block, (start, end) in enumerate(self.blocks):
                self.index[(block, sequence[start:end])].add(i)

    def find(self, query: str) -> dict[int, int]:
        query = query.upper()
        if len(query) != self.length or set(query) - set("ACGT"):
            raise ValueError("Query must match reference length and use A/C/G/T")
        candidates = set()
        for block, (start, end) in enumerate(self.blocks):
            candidates.update(self.index.get((block, query[start:end]), ()))
        hits = {}
        for index in candidates:
            mismatches = 0
            for a, b in zip(query, self.sequences[index]):
                mismatches += a != b
                if mismatches > self.threshold:
                    break
            if mismatches <= self.threshold:
                hits[index] = mismatches
        return hits
