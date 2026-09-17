"""Dependency-light nonlinear predictor for position-encoded promoter sequences."""

from __future__ import annotations

from pathlib import Path
import numpy as np


class PositionMLPStrengthPredictor:
    """Two-layer ReLU MLP trained with mini-batch Adam using NumPy only."""

    def __init__(
        self,
        hidden_one: int = 128,
        hidden_two: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 150,
        patience: int = 20,
        seed: int = 0,
    ):
        self.hidden_one = hidden_one
        self.hidden_two = hidden_two
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.target_mean: float | None = None
        self.target_scale: float | None = None
        self.parameters: dict[str, np.ndarray] | None = None
        self.history: list[dict[str, float]] = []

    def _standardize_features(self, values: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("Predictor has not been fitted")
        return (values - self.feature_mean) / self.feature_scale

    def _forward(self, values: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if self.parameters is None:
            raise RuntimeError("Predictor has not been fitted")
        first_linear = values @ self.parameters["w1"] + self.parameters["b1"]
        first_hidden = np.maximum(first_linear, 0.0)
        second_linear = first_hidden @ self.parameters["w2"] + self.parameters["b2"]
        second_hidden = np.maximum(second_linear, 0.0)
        output = second_hidden @ self.parameters["w3"] + self.parameters["b3"]
        return output[:, 0], (first_linear, first_hidden, second_linear, second_hidden)

    def fit(self, features: np.ndarray, targets: np.ndarray, validation: tuple[np.ndarray, np.ndarray] | None = None) -> "PositionMLPStrengthPredictor":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("Expected 2D features and matching 1D targets")
        self.feature_mean = x.mean(axis=0)
        self.feature_scale = x.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-12] = 1.0
        standardized = self._standardize_features(x)
        self.target_mean = float(y.mean())
        self.target_scale = float(y.std()) if float(y.std()) > 1e-12 else 1.0
        normalized_targets = (y - self.target_mean) / self.target_scale
        rng = np.random.default_rng(self.seed)
        self.parameters = {
            "w1": rng.normal(0, np.sqrt(2 / x.shape[1]), size=(x.shape[1], self.hidden_one)),
            "b1": np.zeros(self.hidden_one),
            "w2": rng.normal(0, np.sqrt(2 / self.hidden_one), size=(self.hidden_one, self.hidden_two)),
            "b2": np.zeros(self.hidden_two),
            "w3": rng.normal(0, np.sqrt(1 / self.hidden_two), size=(self.hidden_two, 1)),
            "b3": np.zeros(1),
        }
        first_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        second_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        best_parameters = {name: value.copy() for name, value in self.parameters.items()}
        best_validation = float("inf")
        remaining_patience = self.patience
        step = 0

        validation_x = validation_y = None
        if validation is not None:
            validation_x = self._standardize_features(np.asarray(validation[0], dtype=np.float64))
            validation_y = np.asarray(validation[1], dtype=np.float64)

        for epoch in range(1, self.epochs + 1):
            for batch_indices in np.array_split(rng.permutation(len(standardized)), max(1, len(standardized) // self.batch_size)):
                batch_x = standardized[batch_indices]
                batch_y = normalized_targets[batch_indices]
                predicted, cache = self._forward(batch_x)
                first_linear, first_hidden, second_linear, second_hidden = cache
                error = (predicted - batch_y) / len(batch_x)
                gradients = {
                    "w3": second_hidden.T @ error[:, None] + self.weight_decay * self.parameters["w3"],
                    "b3": np.array([error.sum()]),
                }
                second_gradient = (error[:, None] @ self.parameters["w3"].T) * (second_linear > 0)
                gradients["w2"] = first_hidden.T @ second_gradient + self.weight_decay * self.parameters["w2"]
                gradients["b2"] = second_gradient.sum(axis=0)
                first_gradient = (second_gradient @ self.parameters["w2"].T) * (first_linear > 0)
                gradients["w1"] = batch_x.T @ first_gradient + self.weight_decay * self.parameters["w1"]
                gradients["b1"] = first_gradient.sum(axis=0)
                step += 1
                for name, gradient in gradients.items():
                    first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
                    second_moment[name] = 0.999 * second_moment[name] + 0.001 * gradient**2
                    adjusted_first = first_moment[name] / (1 - 0.9**step)
                    adjusted_second = second_moment[name] / (1 - 0.999**step)
                    self.parameters[name] -= self.learning_rate * adjusted_first / (np.sqrt(adjusted_second) + 1e-8)
            train_prediction, _ = self._forward(standardized)
            record = {"epoch": float(epoch), "train_mse_normalized": float(np.mean((train_prediction - normalized_targets) ** 2))}
            if validation_x is not None and validation_y is not None:
                validation_prediction, _ = self._forward(validation_x)
                validation_prediction = validation_prediction * self.target_scale + self.target_mean
                validation_mae = float(np.mean(np.abs(validation_prediction - validation_y)))
                record["validation_mae"] = validation_mae
                if validation_mae < best_validation - 1e-6:
                    best_validation = validation_mae
                    best_parameters = {name: value.copy() for name, value in self.parameters.items()}
                    remaining_patience = self.patience
                else:
                    remaining_patience -= 1
                    if remaining_patience == 0:
                        self.history.append(record)
                        break
            self.history.append(record)
        self.parameters = best_parameters
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.target_mean is None or self.target_scale is None:
            raise RuntimeError("Predictor has not been fitted")
        prediction, _ = self._forward(self._standardize_features(np.asarray(features, dtype=np.float64)))
        return prediction * self.target_scale + self.target_mean

    def save(self, path: str | Path) -> None:
        if self.parameters is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("Cannot save an unfitted predictor")
        np.savez_compressed(
            Path(path),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            target_mean=self.target_mean,
            target_scale=self.target_scale,
            hidden_one=self.hidden_one,
            hidden_two=self.hidden_two,
            **self.parameters,
        )
