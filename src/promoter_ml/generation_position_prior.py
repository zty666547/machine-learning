"""Position-informed prior for a continuous-condition Markov generator."""
import numpy as np

from promoter_ml.generation import ContinuousConditionalAutoregressiveGenerator, DNA_ALPHABET


class PositionPriorAutoregressiveGenerator(ContinuousConditionalAutoregressiveGenerator):
    """Replace a uniform context prior with the same-mass position prior.

    Unseen contexts keep the original position-specific fallback. The prior
    strength is exactly 4 * smoothing, equal to the baseline pseudocount mass.
    """

    def fit(self, sequences, conditions):
        super().fit(sequences, conditions)
        values = np.asarray(sequences).astype(str)
        conditions = np.asarray(conditions, dtype=np.float64)
        edges = np.quantile(conditions, np.linspace(0.0, 1.0, self.condition_bins + 1))
        bin_ids = np.digitize(conditions, edges[1:-1], right=True)
        index = {base: i for i, base in enumerate(DNA_ALPHABET)}
        counts = {}
        for sequence, bin_id in zip(values, bin_ids):
            padded = '^' * self.order + sequence
            for position, base in enumerate(sequence):
                key = (int(bin_id), position, padded[position:position+self.order])
                if key not in counts:
                    counts[key] = np.zeros(4, dtype=np.float64)
                counts[key][index[base]] += 1.0
        mass = 4 * self.smoothing
        self.context_probabilities = {
            key: (count + mass * self.fallback_probabilities[key[0], key[1]]) / (count.sum() + mass)
            for key, count in counts.items()
        }
        return self
