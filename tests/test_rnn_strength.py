from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.models.rnn_strength import RNNStrengthPredictor


class RNNStrengthTest(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        rng = np.random.default_rng(71)
        x = np.eye(4)[rng.integers(0, 4, size=(4, 12))]
        y = rng.normal(size=4)
        model = RNNStrengthPredictor(hidden_size=4, location_bins=3, head_size=5, weight_decay=.001)
        model._initialize(np.random.default_rng(42))
        _, gradients = model._loss_and_gradients(x, y)
        for name, parameter in model.parameters.items():
            for flat_index in np.linspace(0, parameter.size - 1, min(parameter.size, 4), dtype=int):
                ix = np.unravel_index(flat_index, parameter.shape)
                original = parameter[ix]
                parameter[ix] = original + 1e-6
                upper, _ = model._loss_and_gradients(x, y)
                parameter[ix] = original - 1e-6
                lower, _ = model._loss_and_gradients(x, y)
                parameter[ix] = original
                self.assertAlmostEqual(gradients[name][ix], (upper - lower) / 2e-6, places=5, msg=name)

    def test_fit_save_load_and_data_validation(self):
        rng = np.random.default_rng(11)
        x = np.eye(4)[rng.integers(0, 4, size=(80, 12))]
        y = (x[:, 2, 0] * 2 + x[:, 9, 1] - 1).astype(float)
        model = RNNStrengthPredictor(hidden_size=8, location_bins=3, head_size=12, epochs=45,
                                     batch_size=16, learning_rate=.01, patience=12, seed=3)
        model.fit(x[:60], y[:60], validation=(x[60:], y[60:]))
        self.assertLess(np.mean(np.abs(model.predict(x[60:]) - y[60:])), np.mean(np.abs(y[60:] - y[:60].mean())))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rnn.npz"
            model.save(path)
            np.testing.assert_allclose(model.predict(x), RNNStrengthPredictor.load(path).predict(x), atol=0, rtol=0)
            candidates = Path(tmp) / "candidates.txt"
            candidates.write_text("A" * 50 + "\n" + "C" * 50 + "\n", encoding="utf-8")
            output = Path(tmp) / "scores.csv"
            subprocess.run([sys.executable, str(ROOT / "scripts/score_rnn_strength.py"), "--model", str(path),
                            "--candidate-file", str(candidates), "--output", str(output)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(len(output.read_text().splitlines()), 3)
        with self.assertRaises(ValueError):
            model.fit(x[:10], y[:9], validation=(x[60:], y[60:]))


if __name__ == "__main__": unittest.main()
