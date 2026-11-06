"""Nonlinear random Fourier sequence-feature ridge regression."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .ridge_strength import RidgeStrengthPredictor


class RandomFourierRidge:
    """Gaussian random Fourier features plus the original linear feature map."""
    def __init__(self, features=256, alpha=1000.0, seed=0):
        if features < 1 or alpha <= 0: raise ValueError("features and alpha must be positive")
        self.features, self.alpha, self.seed = features, alpha, seed
        self.gamma = self.projection = self.phase = self.ridge = None

    def transform(self, x):
        if self.projection is None: raise RuntimeError("Model must be fitted")
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.projection.shape[0]:
            raise ValueError("Invalid feature shape")
        nonlinear = np.sqrt(2 / self.features) * np.cos(values @ self.projection + self.phase)
        return np.concatenate([values, nonlinear], axis=1)

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
            raise ValueError("Expected at least two paired examples")
        if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError("Expected finite inputs")
        rng = np.random.default_rng(self.seed)
        left = rng.integers(0, len(x), size=2048)
        right = rng.integers(0, len(x) - 1, size=2048)
        right += right >= left
        mean_distance = float(np.mean(np.sum((x[left] - x[right]) ** 2, axis=1)))
        if mean_distance <= 1e-12: raise ValueError("Training features do not vary")
        self.gamma = 1 / mean_distance
        self.projection = rng.normal(0, np.sqrt(2 * self.gamma), (x.shape[1], self.features))
        self.phase = rng.uniform(0, 2 * np.pi, self.features)
        self.ridge = RidgeStrengthPredictor(alpha=self.alpha).fit(self.transform(x), y)
        return self

    def predict(self, x):
        if self.ridge is None: raise RuntimeError("Model must be fitted")
        return self.ridge.predict(self.transform(x))

    def save(self, path):
        if self.ridge is None: raise RuntimeError("Model must be fitted")
        np.savez_compressed(Path(path), features=self.features, alpha=self.alpha, seed=self.seed, gamma=self.gamma,
                            projection=self.projection, phase=self.phase, feature_mean=self.ridge.feature_mean,
                            feature_scale=self.ridge.feature_scale, weights=self.ridge.weights, intercept=self.ridge.intercept)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as f:
            model = cls(features=int(f["features"]), alpha=float(f["alpha"]), seed=int(f["seed"]))
            model.gamma, model.projection, model.phase = float(f["gamma"]), f["projection"].copy(), f["phase"].copy()
            model.ridge = RidgeStrengthPredictor(alpha=model.alpha)
            model.ridge.feature_mean, model.ridge.feature_scale = f["feature_mean"].copy(), f["feature_scale"].copy()
            model.ridge.weights, model.ridge.intercept = f["weights"].copy(), float(f["intercept"])
        return model
