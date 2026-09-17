"""Multi-scale position-aware CNN for promoter-strength prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MultiScaleMotifCNNStrengthPredictor:
    """Learn short and long motif candidates while retaining coarse position."""

    def __init__(
        self,
        kernel_sizes: tuple[int, ...] = (5, 7, 9),
        filters_per_scale: int = 24,
        location_bins: int = 5,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 150,
        patience: int = 20,
        seed: int = 0,
    ):
        if not kernel_sizes or any(size < 1 for size in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive window lengths")
        self.kernel_sizes = tuple(kernel_sizes)
        self.filters_per_scale = filters_per_scale
        self.location_bins = location_bins
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.target_mean: float | None = None
        self.target_scale: float | None = None
        self.parameters: dict[str, np.ndarray] | None = None
        self.history: list[dict[str, float]] = []

    @staticmethod
    def _windows(values: np.ndarray, kernel_size: int) -> np.ndarray:
        positions = values.shape[1] - kernel_size + 1
        if positions < 1:
            raise ValueError("Sequence is too short for the chosen convolution window")
        return np.stack([values[:, offset : offset + positions, :] for offset in range(kernel_size)], axis=2)

    def _forward(self, values: np.ndarray) -> tuple[np.ndarray, tuple[list[tuple[np.ndarray, ...]], np.ndarray, np.ndarray, np.ndarray]]:
        if self.parameters is None:
            raise RuntimeError("Predictor has not been fitted")
        branch_cache: list[tuple[np.ndarray, ...]] = []
        pooled_parts = []
        for kernel_size in self.kernel_sizes:
            windows = self._windows(values, kernel_size)
            convolution = np.einsum("npwc,fwc->npf", windows, self.parameters[f"filter_{kernel_size}"]) + self.parameters[f"bias_{kernel_size}"]
            activated = np.maximum(convolution, 0.0)
            groups = tuple(np.array_split(np.arange(activated.shape[1]), self.location_bins))
            pooled = np.concatenate([activated[:, group, :].mean(axis=1) for group in groups], axis=1)
            branch_cache.append((windows, convolution, groups))
            pooled_parts.append(pooled)
        combined = np.concatenate(pooled_parts, axis=1)
        hidden_linear = combined @ self.parameters["head_weight"] + self.parameters["head_bias"]
        hidden = np.maximum(hidden_linear, 0.0)
        output = hidden @ self.parameters["output_weight"] + self.parameters["output_bias"]
        return output[:, 0], (branch_cache, combined, hidden_linear, hidden)

    def fit(self, features: np.ndarray, targets: np.ndarray, validation: tuple[np.ndarray, np.ndarray] | None = None) -> "MultiScaleMotifCNNStrengthPredictor":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 3 or x.shape[2] != 4 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("Expected [sample, position, 4] features and matching targets")
        if any(x.shape[1] - size + 1 < self.location_bins for size in self.kernel_sizes):
            raise ValueError("Each convolution branch needs at least one position per location bin")
        self.target_mean = float(y.mean())
        self.target_scale = float(y.std()) if float(y.std()) > 1e-12 else 1.0
        normalized_targets = (y - self.target_mean) / self.target_scale
        rng = np.random.default_rng(self.seed)
        combined_size = len(self.kernel_sizes) * self.location_bins * self.filters_per_scale
        self.parameters = {
            "head_weight": rng.normal(0, np.sqrt(2 / combined_size), size=(combined_size, self.hidden_size)),
            "head_bias": np.zeros(self.hidden_size),
            "output_weight": rng.normal(0, np.sqrt(1 / self.hidden_size), size=(self.hidden_size, 1)),
            "output_bias": np.zeros(1),
        }
        for kernel_size in self.kernel_sizes:
            self.parameters[f"filter_{kernel_size}"] = rng.normal(0, np.sqrt(2 / (kernel_size * 4)), size=(self.filters_per_scale, kernel_size, 4))
            self.parameters[f"bias_{kernel_size}"] = np.zeros(self.filters_per_scale)
        first = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        second = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        best_parameters = {name: value.copy() for name, value in self.parameters.items()}
        best_validation = float("inf")
        remaining_patience = self.patience
        step = 0
        validation_x = validation_y = None
        if validation is not None:
            validation_x = np.asarray(validation[0], dtype=np.float64)
            validation_y = np.asarray(validation[1], dtype=np.float64)

        for epoch in range(1, self.epochs + 1):
            for batch_indices in np.array_split(rng.permutation(len(x)), max(1, len(x) // self.batch_size)):
                batch_x = x[batch_indices]
                batch_y = normalized_targets[batch_indices]
                predicted, cache = self._forward(batch_x)
                branch_cache, combined, hidden_linear, hidden = cache
                error = (predicted - batch_y) / len(batch_x)
                gradients = {
                    "output_weight": hidden.T @ error[:, None] + self.weight_decay * self.parameters["output_weight"],
                    "output_bias": np.array([error.sum()]),
                }
                hidden_gradient = (error[:, None] @ self.parameters["output_weight"].T) * (hidden_linear > 0)
                gradients["head_weight"] = combined.T @ hidden_gradient + self.weight_decay * self.parameters["head_weight"]
                gradients["head_bias"] = hidden_gradient.sum(axis=0)
                combined_gradient = hidden_gradient @ self.parameters["head_weight"].T
                offset = 0
                for kernel_size, (windows, convolution, groups) in zip(self.kernel_sizes, branch_cache):
                    part_size = self.location_bins * self.filters_per_scale
                    pooled_gradient = combined_gradient[:, offset : offset + part_size].reshape(len(batch_x), self.location_bins, self.filters_per_scale)
                    offset += part_size
                    activated_gradient = np.zeros_like(convolution)
                    for group_index, group in enumerate(groups):
                        activated_gradient[:, group, :] = pooled_gradient[:, group_index : group_index + 1, :] / len(group)
                    convolution_gradient = activated_gradient * (convolution > 0)
                    gradients[f"filter_{kernel_size}"] = np.einsum("npwc,npf->fwc", windows, convolution_gradient) + self.weight_decay * self.parameters[f"filter_{kernel_size}"]
                    gradients[f"bias_{kernel_size}"] = convolution_gradient.sum(axis=(0, 1))
                step += 1
                for name, gradient in gradients.items():
                    first[name] = 0.9 * first[name] + 0.1 * gradient
                    second[name] = 0.999 * second[name] + 0.001 * gradient**2
                    corrected_first = first[name] / (1 - 0.9**step)
                    corrected_second = second[name] / (1 - 0.999**step)
                    self.parameters[name] -= self.learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            train_prediction, _ = self._forward(x)
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
        prediction, _ = self._forward(np.asarray(features, dtype=np.float64))
        return prediction * self.target_scale + self.target_mean

    def save(self, path: str | Path) -> None:
        if self.parameters is None:
            raise RuntimeError("Cannot save an unfitted predictor")
        np.savez_compressed(
            Path(path), target_mean=self.target_mean, target_scale=self.target_scale,
            kernel_sizes=np.asarray(self.kernel_sizes), filters_per_scale=self.filters_per_scale,
            location_bins=self.location_bins, hidden_size=self.hidden_size, **self.parameters,
        )
