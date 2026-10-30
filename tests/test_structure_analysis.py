import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from promoter_ml.structure_analysis import (
    encode_sequences, position_kmer_frequency, position_js, match_window, motif_scan, shuffle_within_sequence
)


class StructureAnalysisTest(unittest.TestCase):
    def test_position_frequencies_and_js(self):
        a = encode_sequences(np.array(["AAC", "AAT"]))
        b = encode_sequences(np.array(["AAC", "AAC"]))
        fa, fb = position_kmer_frequency(a, 2), position_kmer_frequency(b, 2)
        self.assertEqual(fa.shape, (2, 16))
        np.testing.assert_allclose(fa.sum(axis=1), 1)
        np.testing.assert_allclose(position_js(fa, fa), 0)
        self.assertGreater(position_js(fa, fb)[1], 0)
        self.assertEqual(position_js(fa, fb)[0], 0)

    def test_window_hits_and_shuffle(self):
        seq = list("C" * 50)
        seq[15:21] = "TTGACA"
        seq[38:44] = "TATAAT"
        encoded = encode_sequences(np.array(["".join(seq), "C" * 50]))
        np.testing.assert_array_equal(match_window(encoded, "TTGACA", [15], 0), [True, False])
        self.assertEqual(motif_scan(encoded)["both_hit_fraction"], 0.5)
        self.assertEqual(motif_scan(encoded, shift=-5)["both_hit_fraction"], 0.0)
        shuffled = shuffle_within_sequence(encoded, 11)
        for original, changed in zip(encoded, shuffled):
            np.testing.assert_array_equal(np.bincount(original, minlength=4), np.bincount(changed, minlength=4))


if __name__ == "__main__":
    unittest.main()
