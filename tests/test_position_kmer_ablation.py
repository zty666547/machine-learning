import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from evaluate_position_kmer_ablation import encode_position_kmers
from promoter_ml.models.position_trimer_ridge import PositionTrimerRidge, encode_position_trimers


class PositionKmerEncodingTest(unittest.TestCase):
    def test_position_1_2_and_3_encoding(self):
        seqs = np.array(["ACGTA"])
        base = encode_position_kmers(seqs, 5, 2)
        plus = encode_position_kmers(seqs, 5, 3)
        self.assertEqual(base.shape, (1, 9))
        self.assertEqual(plus.shape, (1, 12))
        np.testing.assert_array_equal(base, plus[:, :9])
        np.testing.assert_array_equal(plus[:, 9:] - 9 * 64, encode_position_trimers(seqs, 5))
        self.assertEqual(base[0, :5].tolist(), [0, 65, 130, 195, 256])
        self.assertEqual(base[0, 5:9].tolist(), [320 + 1, 384 + 6, 448 + 11, 512 + 12])

    def test_rejects_invalid_sequence(self):
        with self.assertRaises(ValueError):
            encode_position_kmers(np.array(["ACNTA"]), 5, 3)

    def test_mixed_slots_match_dense_ridge(self):
        seqs = np.array(["AAAAA", "ACGTA", "TGCAT", "GGGTT", "ACTGA", "AACGT"])
        codes = encode_position_kmers(seqs, 5, 3)
        y = np.array([1.3, 2.4, 0.8, 3.0, 1.9, 2.1])
        dense = np.zeros((len(y), codes.shape[1] * 64))
        dense[np.arange(len(y))[:, None], codes] = 1.0
        centered = dense - dense.mean(axis=0)
        weights = np.linalg.solve(centered.T @ centered + 3 * np.eye(dense.shape[1]),
                                  centered.T @ (y - y.mean()))
        expected = y.mean() + centered @ weights
        model = PositionTrimerRidge(alpha=3).fit(codes, y)
        np.testing.assert_allclose(model.predict(codes), expected, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
