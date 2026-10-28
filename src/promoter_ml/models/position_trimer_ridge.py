"""Position-specific 3-mer ridge regression with a sparse implicit design matrix."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def encode_position_trimers(sequences: np.ndarray, sequence_length: int = 50) -> np.ndarray:
    """Encode one active 3-mer feature per start position, without a dense matrix."""
    if sequence_length < 3:
        raise ValueError("sequence_length must be at least 3")
    values = np.asarray(sequences)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Expected a nonempty one-dimensional sequence array")
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.empty((len(values), sequence_length), dtype=np.int64)
    for i, value in enumerate(values):
        seq = str(value).upper()
        if len(seq) != sequence_length or any(base not in mapping for base in seq):
            raise ValueError("Sequence must have the specified length and only A/C/G/T")
        encoded[i] = [mapping[base] for base in seq]
    codes = encoded[:, :-2] * 16 + encoded[:, 1:-1] * 4 + encoded[:, 2:]
    return codes + 64 * np.arange(sequence_length - 2)[None, :]


class PositionTrimerRidge:
    """Ridge with unstandardized one-hot features and an unpenalized intercept."""
    def __init__(self, alpha: float = 10.0, tolerance: float = 1e-8, max_iter: int = 1000):
        if not np.isfinite(alpha) or alpha <= 0 or tolerance <= 0 or max_iter < 1:
            raise ValueError("alpha, tolerance, and max_iter must be positive")
        self.alpha, self.tolerance, self.max_iter = float(alpha), tolerance, max_iter
        self.weights = None
        self.intercept = None
        self.sequence_length = None
        self.iterations = 0
        self.relative_residual = None

    @staticmethod
    def _xt(codes: np.ndarray, values: np.ndarray, width: int) -> np.ndarray:
        return np.bincount(codes.ravel(), weights=np.repeat(values, codes.shape[1]), minlength=width)

    def fit(self, codes: np.ndarray, targets: np.ndarray) -> "PositionTrimerRidge":
        codes = np.asarray(codes, dtype=np.int64)
        y = np.asarray(targets, dtype=np.float64)
        if codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] == 0 or y.shape != (len(codes),):
            raise ValueError("Expected nonempty position codes and matching 1D targets")
        width = codes.shape[1] * 64
        expected = 64 * np.arange(codes.shape[1])
        if np.any(codes < expected) or np.any(codes >= expected + 64) or not np.all(np.isfinite(y)):
            raise ValueError("Invalid position codes or non-finite targets")
        n = len(y)
        mean = self._xt(codes, np.ones(n), width) / n
        y_mean = float(y.mean())
        rhs = self._xt(codes, y - y_mean, width)

        def matvec(weights):
            predictions = weights[codes].sum(axis=1)
            return self._xt(codes, predictions, width) - n * mean * (mean @ weights) + self.alpha * weights

        weights = np.zeros(width)
        residual = rhs.copy()
        direction = residual.copy()
        rs = float(residual @ residual)
        initial_norm = np.sqrt(rs)
        if initial_norm > 0:
            for iteration in range(1, self.max_iter + 1):
                product = matvec(direction)
                step = rs / float(direction @ product)
                weights += step * direction
                residual -= step * product
                new_rs = float(residual @ residual)
                self.iterations = iteration
                if np.sqrt(new_rs) <= self.tolerance * initial_norm:
                    rs = new_rs
                    break
                direction = residual + (new_rs / rs) * direction
                rs = new_rs
            else:
                raise RuntimeError(f"Conjugate gradient did not converge in {self.max_iter} iterations")
        self.relative_residual = float(np.sqrt(rs) / initial_norm) if initial_norm else 0.0
        self.weights = weights
        self.intercept = y_mean - float(mean @ weights)
        self.sequence_length = codes.shape[1] + 2
        return self

    def predict(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.int64)
        if self.weights is None or self.intercept is None:
            raise RuntimeError("Predictor is not fitted")
        if codes.ndim != 2 or codes.shape[1] != self.sequence_length - 2:
            raise ValueError("Incorrect position code shape")
        expected = 64 * np.arange(codes.shape[1])
        if np.any(codes < expected) or np.any(codes >= expected + 64):
            raise ValueError("Invalid position codes")
        return self.intercept + self.weights[codes].sum(axis=1)

    def save(self, path: str | Path) -> None:
        if self.weights is None:
            raise RuntimeError("Predictor is not fitted")
        np.savez_compressed(path, weights=self.weights, intercept=self.intercept,
                            sequence_length=self.sequence_length, alpha=self.alpha)

    @classmethod
    def load(cls, path: str | Path) -> "PositionTrimerRidge":
        with np.load(path, allow_pickle=False) as state:
            model = cls(alpha=float(state["alpha"]))
            model.weights = state["weights"].copy()
            model.intercept = float(state["intercept"])
            model.sequence_length = int(state["sequence_length"])
        return model
