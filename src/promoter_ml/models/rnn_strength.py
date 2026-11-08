"""Position-aware, vanilla tanh RNN regressor implemented with NumPy."""
from __future__ import annotations
from pathlib import Path
import numpy as np


class RNNStrengthPredictor:
    """Scan one-hot bases left to right and pool recurrent states into bins."""

    def __init__(self, hidden_size=32, location_bins=5, head_size=64, learning_rate=0.001,
                 weight_decay=0.0001, batch_size=256, epochs=150, patience=20,
                 gradient_clip=1.0, seed=0, loss_type="mse", huber_delta=1.0,
                 rank_weight=0.2, rank_min_gap=0.25, tail_weight=0.75):
        if min(hidden_size, location_bins, head_size, batch_size, epochs, patience) < 1 or gradient_clip <= 0:
            raise ValueError("Positive dimensions, training settings and gradient clip required")
        if loss_type not in ("mse", "huber", "mse_pairwise", "tail_weighted_mse") or huber_delta <= 0 or rank_weight < 0 or rank_min_gap < 0 or tail_weight < 0:
            raise ValueError("Invalid loss settings")
        self.hidden_size, self.location_bins, self.head_size = hidden_size, location_bins, head_size
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.batch_size, self.epochs, self.patience = batch_size, epochs, patience
        self.gradient_clip, self.seed = gradient_clip, seed
        self.loss_type, self.huber_delta = loss_type, huber_delta
        self.rank_weight, self.rank_min_gap = rank_weight, rank_min_gap
        self.tail_weight = tail_weight
        self.parameters = None
        self.target_mean = self.target_scale = None
        self.history = []

    def _initialize(self, rng):
        h, p, q = self.hidden_size, self.hidden_size * self.location_bins, self.head_size
        self.parameters = {
            "input_weight": rng.normal(0, np.sqrt(1 / 4), (4, h)),
            "recurrent_weight": rng.normal(0, np.sqrt(1 / h), (h, h)),
            "recurrent_bias": np.zeros(h),
            "head_weight": rng.normal(0, np.sqrt(2 / p), (p, q)),
            "head_bias": np.zeros(q),
            "output_weight": rng.normal(0, np.sqrt(1 / q), (q, 1)),
            "output_bias": np.zeros(1),
        }

    def _forward(self, x):
        if self.parameters is None: raise RuntimeError("Predictor has not been fitted")
        if x.ndim != 3 or x.shape[2] != 4 or x.shape[1] < self.location_bins:
            raise ValueError("Expected [sample, position, 4] and enough positions")
        p = self.parameters
        states = np.zeros((len(x), x.shape[1] + 1, self.hidden_size))
        for t in range(x.shape[1]):
            states[:, t + 1] = np.tanh(x[:, t] @ p["input_weight"] + states[:, t] @ p["recurrent_weight"] + p["recurrent_bias"])
        groups = np.array_split(np.arange(x.shape[1]), self.location_bins)
        pooled = np.concatenate([states[:, g + 1, :].mean(axis=1) for g in groups], axis=1)
        hidden_linear = pooled @ p["head_weight"] + p["head_bias"]
        hidden = np.maximum(hidden_linear, 0)
        output = (hidden @ p["output_weight"] + p["output_bias"])[:, 0]
        return output, (states, groups, pooled, hidden_linear, hidden)

    def _loss_and_gradients(self, x, y):
        prediction, (states, groups, pooled, hidden_linear, hidden) = self._forward(x)
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
            d_states[:, positions + 1, :] = d_pooled[:, i:i + 1, :] / len(positions)
        g["input_weight"] = np.zeros_like(p["input_weight"])
        g["recurrent_weight"] = np.zeros_like(p["recurrent_weight"])
        g["recurrent_bias"] = np.zeros_like(p["recurrent_bias"])
        carry = np.zeros((len(x), self.hidden_size))
        for t in range(x.shape[1] - 1, -1, -1):
            delta = (d_states[:, t + 1, :] + carry) * (1 - states[:, t + 1, :] ** 2)
            g["input_weight"] += x[:, t, :].T @ delta
            g["recurrent_weight"] += states[:, t, :].T @ delta
            g["recurrent_bias"] += delta.sum(axis=0)
            carry = delta @ p["recurrent_weight"].T
        for name in ("input_weight", "recurrent_weight"):
            g[name] += self.weight_decay * p[name]
        penalty = sum(np.sum(p[n] ** 2) for n in ("input_weight", "recurrent_weight", "head_weight", "output_weight"))
        loss = data_loss + .5 * self.weight_decay * penalty
        return float(loss), g

    def _data_loss_and_gradient(self, prediction, target):
        residual = prediction - target
        if self.loss_type == "tail_weighted_mse":
            weights = 1.0 + self.tail_weight * np.abs(target)
            normalizer = float(np.mean(weights))
            return float(.5 * np.mean(weights * residual ** 2) / normalizer), weights * residual / (len(residual) * normalizer)
        if self.loss_type in ("mse", "mse_pairwise"):
            loss, gradient = float(.5 * np.mean(residual ** 2)), residual / len(residual)
            if self.loss_type == "mse_pairwise":
                pairs = len(residual) // 2
                left = np.arange(pairs)
                right = len(residual) - 1 - left
                target_gap = target[left] - target[right]
                valid = np.abs(target_gap) >= self.rank_min_gap
                left, right, target_gap = left[valid], right[valid], target_gap[valid]
                if len(left):
                    direction = np.sign(target_gap)
                    margin = direction * (prediction[left] - prediction[right])
                    loss += self.rank_weight * float(np.mean(np.logaddexp(0, -margin)))
                    pair_gradient = -self.rank_weight * direction * np.exp(-np.logaddexp(0, margin)) / len(left)
                    np.add.at(gradient, left, pair_gradient)
                    np.add.at(gradient, right, -pair_gradient)
            return loss, gradient
        delta = self.huber_delta
        magnitude = np.abs(residual)
        loss = np.where(magnitude <= delta, .5 * residual ** 2, delta * (magnitude - .5 * delta))
        return float(np.mean(loss)), np.clip(residual, -delta, delta) / len(residual)

    def fit(self, features, targets, validation):
        x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
        vx, vy = np.asarray(validation[0], dtype=np.float64), np.asarray(validation[1], dtype=np.float64)
        if x.ndim != 3 or x.shape[2] != 4 or len(x) != len(y) or len(x) == 0 or y.ndim != 1:
            raise ValueError("Expected nonempty training sequences and targets")
        if vx.ndim != 3 or vx.shape[1:] != x.shape[1:] or vy.ndim != 1 or len(vx) != len(vy) or len(vx) == 0:
            raise ValueError("Expected matching nonempty validation sequences and targets")
        if not all(np.isfinite(a).all() for a in (x, y, vx, vy)):
            raise ValueError("Training and validation values must be finite")
        self.target_mean, scale = float(y.mean()), float(y.std())
        self.target_scale = scale if scale > 1e-12 else 1.0
        normalized = (y - self.target_mean) / self.target_scale
        rng = np.random.default_rng(self.seed)
        self._initialize(rng)
        first = {name: np.zeros_like(v) for name, v in self.parameters.items()}
        second = {name: np.zeros_like(v) for name, v in self.parameters.items()}
        best, best_loss, remaining, step = None, float("inf"), self.patience, 0
        self.history = []
        for epoch in range(1, self.epochs + 1):
            clip_batches = 0
            for batch in np.array_split(rng.permutation(len(x)), max(1, len(x) // self.batch_size)):
                _, gradients = self._loss_and_gradients(x[batch], normalized[batch])
                norm = np.sqrt(sum(np.sum(v * v) for v in gradients.values()))
                if not np.isfinite(norm): raise FloatingPointError("Non-finite RNN gradient")
                if norm > self.gradient_clip:
                    gradients = {name: v * (self.gradient_clip / norm) for name, v in gradients.items()}
                    clip_batches += 1
                step += 1
                for name, gradient in gradients.items():
                    first[name] = .9 * first[name] + .1 * gradient
                    second[name] = .999 * second[name] + .001 * gradient**2
                    self.parameters[name] -= self.learning_rate * (first[name] / (1 - .9**step)) / (np.sqrt(second[name] / (1 - .999**step)) + 1e-8)
            val_mae = float(np.mean(np.abs(self.predict(vx) - vy)))
            self.history.append({"epoch": epoch, "validation_mae": val_mae, "clipped_batches": clip_batches})
            if val_mae < best_loss - 1e-6:
                best_loss, remaining = val_mae, self.patience
                best = {name: value.copy() for name, value in self.parameters.items()}
            else:
                remaining -= 1
                if remaining == 0: break
        if best is None: raise RuntimeError("No finite validation checkpoint")
        self.parameters = best
        return self

    def predict(self, features):
        if self.target_mean is None or self.target_scale is None: raise RuntimeError("Predictor has not been fitted")
        return self._forward(np.asarray(features, dtype=np.float64))[0] * self.target_scale + self.target_mean

    def save(self, path):
        if self.parameters is None or self.target_mean is None: raise RuntimeError("Predictor has not been fitted")
        np.savez_compressed(Path(path), target_mean=self.target_mean, target_scale=self.target_scale,
                            hidden_size=self.hidden_size, location_bins=self.location_bins, head_size=self.head_size,
                            **self.parameters)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as f:
            model = cls(hidden_size=int(f["hidden_size"]), location_bins=int(f["location_bins"]), head_size=int(f["head_size"]))
            model.target_mean, model.target_scale = float(f["target_mean"]), float(f["target_scale"])
            model.parameters = {name: f[name].copy() for name in ("input_weight", "recurrent_weight", "recurrent_bias",
                                "head_weight", "head_bias", "output_weight", "output_bias")}
        return model
