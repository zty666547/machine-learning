"""Position-aware GRU strength regressor implemented with NumPy."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .rnn_strength import RNNStrengthPredictor


class GRUStrengthPredictor(RNNStrengthPredictor):
    """Single-layer unidirectional GRU with the same five-bin prediction head."""

    def _initialize(self, rng):
        h, p, q = self.hidden_size, self.hidden_size * self.location_bins, self.head_size
        self.parameters = {}
        for name in ("z", "r", "n"):
            self.parameters[f"w_{name}"] = rng.normal(0, np.sqrt(1 / 4), (4, h))
            self.parameters[f"u_{name}"] = rng.normal(0, np.sqrt(1 / h), (h, h))
            self.parameters[f"b_{name}"] = np.zeros(h)
        self.parameters.update({
            "head_weight": rng.normal(0, np.sqrt(2 / p), (p, q)),
            "head_bias": np.zeros(q),
            "output_weight": rng.normal(0, np.sqrt(1 / q), (q, 1)),
            "output_bias": np.zeros(1),
        })

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -40, 40)))

    def _forward(self, x):
        if self.parameters is None: raise RuntimeError("Predictor has not been fitted")
        if x.ndim != 3 or x.shape[2] != 4 or x.shape[1] < self.location_bins:
            raise ValueError("Expected [sample, position, 4] and enough positions")
        p = self.parameters
        states = np.zeros((len(x), x.shape[1] + 1, self.hidden_size))
        z = np.zeros_like(states[:, 1:])
        r = np.zeros_like(z)
        candidate = np.zeros_like(z)
        for t in range(x.shape[1]):
            previous = states[:, t]
            z[:, t] = self._sigmoid(x[:, t] @ p["w_z"] + previous @ p["u_z"] + p["b_z"])
            r[:, t] = self._sigmoid(x[:, t] @ p["w_r"] + previous @ p["u_r"] + p["b_r"])
            candidate[:, t] = np.tanh(x[:, t] @ p["w_n"] + (r[:, t] * previous) @ p["u_n"] + p["b_n"])
            states[:, t+1] = (1 - z[:, t]) * previous + z[:, t] * candidate[:, t]
        groups = np.array_split(np.arange(x.shape[1]), self.location_bins)
        pooled = np.concatenate([states[:, g + 1, :].mean(axis=1) for g in groups], axis=1)
        hidden_linear = pooled @ p["head_weight"] + p["head_bias"]
        hidden = np.maximum(hidden_linear, 0)
        output = (hidden @ p["output_weight"] + p["output_bias"])[:, 0]
        return output, (states, z, r, candidate, groups, pooled, hidden_linear, hidden)

    def _loss_and_gradients(self, x, y):
        prediction, (states, z, r, candidate, groups, pooled, hidden_linear, hidden) = self._forward(x)
        p = self.parameters
        data_loss, error = self._data_loss_and_gradient(prediction, y)
        g = {"output_weight": hidden.T @ error[:, None] + self.weight_decay * p["output_weight"],
             "output_bias": np.array([error.sum()])}
        d_hidden = (error[:, None] @ p["output_weight"].T) * (hidden_linear > 0)
        g["head_weight"] = pooled.T @ d_hidden + self.weight_decay * p["head_weight"]
        g["head_bias"] = d_hidden.sum(axis=0)
        d_pooled = (d_hidden @ p["head_weight"].T).reshape(len(x), self.location_bins, self.hidden_size)
        d_states = np.zeros_like(states)
        for i, positions in enumerate(groups):
            d_states[:, positions + 1, :] = d_pooled[:, i:i+1, :] / len(positions)
        for gate in ("z", "r", "n"):
            for prefix in ("w", "u", "b"):
                name = f"{prefix}_{gate}"
                g[name] = np.zeros_like(p[name])
        carry = np.zeros((len(x), self.hidden_size))
        for t in range(x.shape[1] - 1, -1, -1):
            previous = states[:, t]
            dh = d_states[:, t+1] + carry
            dz = dh * (candidate[:, t] - previous)
            dn = dh * z[:, t]
            carry = dh * (1 - z[:, t])
            an = dn * (1 - candidate[:, t] ** 2)
            g["w_n"] += x[:, t].T @ an
            g["u_n"] += (r[:, t] * previous).T @ an
            g["b_n"] += an.sum(axis=0)
            d_reset_product = an @ p["u_n"].T
            dr = d_reset_product * previous
            carry += d_reset_product * r[:, t]
            ar = dr * r[:, t] * (1 - r[:, t])
            g["w_r"] += x[:, t].T @ ar
            g["u_r"] += previous.T @ ar
            g["b_r"] += ar.sum(axis=0)
            carry += ar @ p["u_r"].T
            az = dz * z[:, t] * (1 - z[:, t])
            g["w_z"] += x[:, t].T @ az
            g["u_z"] += previous.T @ az
            g["b_z"] += az.sum(axis=0)
            carry += az @ p["u_z"].T
        weights = [f"{prefix}_{gate}" for gate in ("z", "r", "n") for prefix in ("w", "u")]
        for name in weights:
            g[name] += self.weight_decay * p[name]
        penalty = sum(np.sum(p[n] ** 2) for n in (*weights, "head_weight", "output_weight"))
        return float(data_loss + .5 * self.weight_decay * penalty), g

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as f:
            model = cls(hidden_size=int(f["hidden_size"]), location_bins=int(f["location_bins"]), head_size=int(f["head_size"]))
            model.target_mean, model.target_scale = float(f["target_mean"]), float(f["target_scale"])
            names = [f"{prefix}_{gate}" for gate in ("z", "r", "n") for prefix in ("w", "u", "b")]
            names += ["head_weight", "head_bias", "output_weight", "output_bias"]
            model.parameters = {name: f[name].copy() for name in names}
        return model
