"""Position-specific, non-adjacent DNA base-pair features."""
import numpy as np


class SpacedPairFeaturizer:
    """One-hot (base at position i, base at i+gap), for fixed gaps."""

    def __init__(self, sequence_length=50, spacings=(10, 17)):
        self.sequence_length = int(sequence_length)
        self.spacings = tuple(spacings)
        if self.sequence_length < 2 or not self.spacings or len(set(self.spacings)) != len(self.spacings):
            raise ValueError('Sequence length and unique spacings are required')
        if any(not isinstance(gap, int) or gap <= 0 or gap >= self.sequence_length for gap in self.spacings):
            raise ValueError('Each spacing must be an integer in [1, sequence_length)')

    @property
    def feature_count(self):
        return sum(self.sequence_length-gap for gap in self.spacings) * 16

    def transform(self, sequences):
        values = np.asarray(sequences)
        if values.ndim != 1:
            raise ValueError('Expected a one-dimensional sequence array')
        values = values.astype(str)
        if any(len(sequence) != self.sequence_length or set(sequence) - set('ACGT') for sequence in values):
            raise ValueError('Sequences must have the specified length and contain only A, C, G, T')
        output = np.zeros((len(values), self.feature_count), dtype=np.float64)
        if not len(values):
            return output
        lookup = np.full(256, -1, dtype=np.int8)
        lookup[np.frombuffer(b'ACGT', dtype=np.uint8)] = np.arange(4)
        bases = lookup[np.frombuffer(''.join(values).encode('ascii'), dtype=np.uint8)].reshape(-1, self.sequence_length)
        offset = 0
        for gap in self.spacings:
            positions = self.sequence_length-gap
            columns = offset + np.arange(positions)[None, :] * 16 + bases[:, :positions] * 4 + bases[:, gap:]
            output[np.arange(len(values))[:, None], columns] = 1.0
            offset += positions * 16
        return output
