from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.models import RNNStrengthPredictor, RandomFourierRidge


class RankingRffTest(unittest.TestCase):
    def test_pairwise_gradient_and_no_valid_pairs(self):
        model = RNNStrengthPredictor(loss_type="mse_pairwise", rank_weight=.2, rank_min_gap=.25)
        pred = np.array([-.8, .1, .3, .9, -.2, .6])
        target = np.array([-1., .5, .05, .1, -.5, 1.5])
        loss, grad = model._data_loss_and_gradient(pred, target)
        for i in range(len(pred)):
            a, b = pred.copy(), pred.copy()
            a[i] += 1e-6
            b[i] -= 1e-6
            numeric = (model._data_loss_and_gradient(a, target)[0] - model._data_loss_and_gradient(b, target)[0]) / 2e-6
            self.assertAlmostEqual(grad[i], numeric, places=6)
        same = np.zeros(6)
        _, plain_grad = model._data_loss_and_gradient(pred, same)
        np.testing.assert_allclose(plain_grad, pred / len(pred))
        self.assertGreater(loss, 0)

    def test_pairwise_backprop(self):
        rng = np.random.default_rng(33)
        x = np.eye(4)[rng.integers(0, 4, size=(6, 9))]
        target = np.array([-2., -1., -.5, .5, 1., 2.])
        model = RNNStrengthPredictor(hidden_size=4, location_bins=3, head_size=5,
                                     weight_decay=.001, loss_type="mse_pairwise")
        model._initialize(np.random.default_rng(42))
        _, gradients = model._loss_and_gradients(x, target)
        for name, parameter in model.parameters.items():
            index = np.unravel_index(parameter.size // 2, parameter.shape)
            v = parameter[index]
            parameter[index] = v + 1e-6
            a, _ = model._loss_and_gradients(x, target)
            parameter[index] = v - 1e-6
            b, _ = model._loss_and_gradients(x, target)
            parameter[index] = v
            self.assertAlmostEqual(gradients[name][index], (a - b) / 2e-6, places=5, msg=name)

    def test_rff_learns_nonlinear_interaction_and_roundtrip(self):
        rng = np.random.default_rng(11)
        x = rng.uniform(-1, 1, size=(180, 2))
        y = x[:, 0] * x[:, 1]
        model = RandomFourierRidge(features=128, alpha=1, seed=5).fit(x[:140], y[:140])
        prediction = model.predict(x[140:])
        self.assertGreater(np.corrcoef(prediction, y[140:])[0, 1], .9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rff.npz"
            model.save(path)
            np.testing.assert_allclose(prediction, RandomFourierRidge.load(path).predict(x[140:]), atol=0, rtol=0)


if __name__ == "__main__": unittest.main()
