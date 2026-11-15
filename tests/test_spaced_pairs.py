import unittest
import numpy as np
from promoter_ml.spaced_pairs import SpacedPairFeaturizer


class SpacedPairTests(unittest.TestCase):
    def test_position_gap_and_order(self):
        encoder = SpacedPairFeaturizer(sequence_length=4, spacings=(2,))
        actual = encoder.transform(np.array(['ACGT', 'GTAC']))
        expected = np.zeros((2, 32))
        expected[0, [2, 23]] = 1  # A,G at start 0; C,T at start 1
        expected[1, [8, 29]] = 1  # G,A and T,C: ordering matters
        np.testing.assert_array_equal(actual, expected)

    def test_multiple_gaps_and_empty_batch(self):
        encoder = SpacedPairFeaturizer(sequence_length=5, spacings=(1, 3))
        actual = encoder.transform(np.array(['AAAAA']))
        self.assertEqual(actual.shape, (1, 96))
        np.testing.assert_array_equal(np.flatnonzero(actual[0]), [0, 16, 32, 48, 64, 80])
        self.assertEqual(encoder.transform(np.array([], dtype=str)).shape, (0, 96))

    def test_bad_input_rejected(self):
        for spacings in [(0,), (4,), (1, 1)]:
            with self.assertRaises(ValueError):
                SpacedPairFeaturizer(4, spacings)
        encoder = SpacedPairFeaturizer(4, (2,))
        for values in [['ACGN'], ['ACG'], [['ACGT']]]:
            with self.assertRaises(ValueError):
                encoder.transform(np.array(values))


if __name__ == '__main__':
    unittest.main()
