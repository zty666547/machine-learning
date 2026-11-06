from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.models import GRUStrengthPredictor, RNNStrengthPredictor


class GruAndHuberTest(unittest.TestCase):
    def test_huber_loss_and_gradient(self):
        model = RNNStrengthPredictor(loss_type="huber", huber_delta=1)
        loss, gradient = model._data_loss_and_gradient(np.array([-2., -.5, .25, 4.]), np.zeros(4))
        self.assertAlmostEqual(loss, (1.5 + .125 + .03125 + 3.5) / 4)
        np.testing.assert_allclose(gradient, np.array([-1, -.5, .25, 1]) / 4)

    def test_gru_bptt_finite_difference(self):
        rng = np.random.default_rng(71)
        x = np.eye(4)[rng.integers(0, 4, size=(4, 9))]
        y = rng.normal(size=4)
        model = GRUStrengthPredictor(hidden_size=4, location_bins=3, head_size=5, weight_decay=.001)
        model._initialize(np.random.default_rng(42))
        _, gradients = model._loss_and_gradients(x, y)
        for name, parameter in model.parameters.items():
            for flat_index in np.linspace(0, parameter.size-1, min(parameter.size, 4), dtype=int):
                ix = np.unravel_index(flat_index, parameter.shape)
                original = parameter[ix]
                parameter[ix] = original + 1e-6
                upper, _ = model._loss_and_gradients(x, y)
                parameter[ix] = original - 1e-6
                lower, _ = model._loss_and_gradients(x, y)
                parameter[ix] = original
                self.assertAlmostEqual(gradients[name][ix], (upper - lower) / 2e-6, places=5, msg=name)

    def test_rnn_huber_full_gradient(self):
        rng = np.random.default_rng(72)
        x = np.eye(4)[rng.integers(0, 4, size=(4, 9))]
        y = np.array([-4.0, -0.2, 0.2, 4.0])
        model = RNNStrengthPredictor(hidden_size=4, location_bins=3, head_size=5,
                                     weight_decay=.001, loss_type="huber", huber_delta=1)
        model._initialize(np.random.default_rng(42))
        _, gradients = model._loss_and_gradients(x, y)
        for name, parameter in model.parameters.items():
            ix = np.unravel_index(parameter.size // 2, parameter.shape)
            original = parameter[ix]
            parameter[ix] = original + 1e-6
            upper, _ = model._loss_and_gradients(x, y)
            parameter[ix] = original - 1e-6
            lower, _ = model._loss_and_gradients(x, y)
            parameter[ix] = original
            self.assertAlmostEqual(gradients[name][ix], (upper - lower) / 2e-6, places=5, msg=name)

    def test_gru_fit_and_save_roundtrip(self):
        rng = np.random.default_rng(11)
        x = np.eye(4)[rng.integers(0, 4, size=(80, 12))]
        y = (x[:, 2, 0] * 2 + x[:, 9, 1] - 1).astype(float)
        model = GRUStrengthPredictor(hidden_size=8, location_bins=3, head_size=12, epochs=40,
                                     batch_size=16, learning_rate=.01, patience=12, seed=3)
        model.fit(x[:60], y[:60], validation=(x[60:], y[60:]))
        self.assertLess(np.mean(np.abs(model.predict(x[60:]) - y[60:])), np.mean(np.abs(y[60:] - y[:60].mean())))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gru.npz"
            model.save(path)
            np.testing.assert_allclose(model.predict(x), GRUStrengthPredictor.load(path).predict(x), atol=0, rtol=0)
            candidates = Path(tmp) / "candidate.txt"
            candidates.write_text("A" * 50 + "\n" + "C" * 50 + "\n", encoding="utf-8")
            output = Path(tmp) / "scores.csv"
            subprocess.run([sys.executable, str(ROOT / "scripts/score_rnn_strength.py"), "--model", str(path),
                            "--model-type", "gru", "--candidate-file", str(candidates), "--output", str(output)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(len(output.read_text().splitlines()), 3)


if __name__ == "__main__": unittest.main()
