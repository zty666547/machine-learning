import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from promoter_ml.models.position_trimer_ridge import PositionTrimerRidge, encode_position_trimers


class PositionTrimerRidgeTest(unittest.TestCase):
    def test_encoding_and_validation(self):
        codes = encode_position_trimers(np.array(["ACGTA", "TGCAT"]), 5)
        self.assertEqual(codes[0].tolist(), [6, 64 + 27, 128 + 44])
        with self.assertRaises(ValueError):
            encode_position_trimers(np.array(["ACNTT"]), 5)

    def test_fit_matches_dense_ridge_and_roundtrip(self):
        seqs = np.array(["AAAAA", "ACGTA", "TGCAT", "GGGTT", "ACTGA", "AACGT"])
        codes = encode_position_trimers(seqs, 5)
        y = np.array([1.3, 2.4, 0.8, 3.0, 1.9, 2.1])
        dense = np.zeros((len(y), 3 * 64))
        dense[np.arange(len(y))[:, None], codes] = 1.0
        centered = dense - dense.mean(axis=0)
        weights = np.linalg.solve(centered.T @ centered + 3 * np.eye(dense.shape[1]),
                                  centered.T @ (y - y.mean()))
        expected = y.mean() + centered @ weights
        model = PositionTrimerRidge(alpha=3).fit(codes, y)
        np.testing.assert_allclose(model.predict(codes), expected, rtol=1e-7, atol=1e-7)
        self.assertLess(model.relative_residual, 1e-8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            np.testing.assert_allclose(PositionTrimerRidge.load(path).predict(codes), expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
