"""A dependency-light conditional VAE for fixed-length DNA sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ConditionalSequenceVAE:
    """NumPy conditional VAE with a continuous normalized strength condition."""

    def __init__(
        self,
        sequence_length: int = 50,
        hidden_size: int = 96,
        latent_size: int = 16,
        learning_rate: float = 1e-3,
        beta: float = 0.20,
        batch_size: int = 256,
        epochs: int = 120,
        patience: int = 15,
        seed: int = 0,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.condition_mean: float | None = None
        self.condition_scale: float | None = None
        self.parameters: dict[str, np.ndarray] | None = None
        self.history: list[dict[str, float]] = []

    @property
    def input_size(self) -> int:
        return self.sequence_length * 4

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=2, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=2, keepdims=True)

    def _normalize_condition(self, conditions: np.ndarray) -> np.ndarray:
        if self.condition_mean is None or self.condition_scale is None:
            raise RuntimeError("VAE has not been fitted")
        return ((np.asarray(conditions, dtype=np.float64) - self.condition_mean) / self.condition_scale)[:, None]

    def _initialize(self, rng: np.random.Generator) -> None:
        encoder_input = self.input_size + 1
        decoder_input = self.latent_size + 1
        self.parameters = {
            "encoder_weight": rng.normal(0, np.sqrt(2 / encoder_input), size=(encoder_input, self.hidden_size)),
            "encoder_bias": np.zeros(self.hidden_size),
            "mu_weight": rng.normal(0, np.sqrt(1 / self.hidden_size), size=(self.hidden_size, self.latent_size)),
            "mu_bias": np.zeros(self.latent_size),
            "logvar_weight": rng.normal(0, np.sqrt(1 / self.hidden_size), size=(self.hidden_size, self.latent_size)),
            "logvar_bias": np.full(self.latent_size, -2.0),
            "decoder_weight": rng.normal(0, np.sqrt(2 / decoder_input), size=(decoder_input, self.hidden_size)),
            "decoder_bias": np.zeros(self.hidden_size),
            "output_weight": rng.normal(0, np.sqrt(1 / self.hidden_size), size=(self.hidden_size, self.input_size)),
            "output_bias": np.zeros(self.input_size),
        }

    def _forward(self, features: np.ndarray, normalized_conditions: np.ndarray, epsilon: np.ndarray | None) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        if self.parameters is None:
            raise RuntimeError("VAE has not been fitted")
        flattened = features.reshape(len(features), -1)
        encoder_input = np.concatenate([flattened, normalized_conditions], axis=1)
        encoder_linear = encoder_input @ self.parameters["encoder_weight"] + self.parameters["encoder_bias"]
        encoder_hidden = np.maximum(encoder_linear, 0.0)
        mu = encoder_hidden @ self.parameters["mu_weight"] + self.parameters["mu_bias"]
        logvar_raw = encoder_hidden @ self.parameters["logvar_weight"] + self.parameters["logvar_bias"]
        logvar = np.clip(logvar_raw, -8.0, 4.0)
        if epsilon is None:
            latent = mu
        else:
            latent = mu + np.exp(0.5 * logvar) * epsilon
        decoder_input = np.concatenate([latent, normalized_conditions], axis=1)
        decoder_linear = decoder_input @ self.parameters["decoder_weight"] + self.parameters["decoder_bias"]
        decoder_hidden = np.maximum(decoder_linear, 0.0)
        logits = (decoder_hidden @ self.parameters["output_weight"] + self.parameters["output_bias"]).reshape(len(features), self.sequence_length, 4)
        probabilities = self._softmax(logits)
        return probabilities, (flattened, encoder_input, encoder_linear, encoder_hidden, mu, logvar_raw, logvar, latent, decoder_input, decoder_linear, decoder_hidden)

    def fit(self, features: np.ndarray, conditions: np.ndarray, validation: tuple[np.ndarray, np.ndarray] | None = None) -> "ConditionalSequenceVAE":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(conditions, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != (self.sequence_length, 4) or y.ndim != 1 or len(x) != len(y):
            raise ValueError("Expected [sample, sequence_length, 4] features and matching conditions")
        self.condition_mean = float(y.mean())
        self.condition_scale = float(y.std()) if float(y.std()) > 1e-12 else 1.0
        normalized_conditions = self._normalize_condition(y)
        rng = np.random.default_rng(self.seed)
        self._initialize(rng)
        if self.parameters is None:
            raise RuntimeError("Parameter initialization failed")
        first = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        second = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        best_parameters = {name: value.copy() for name, value in self.parameters.items()}
        best_validation = float("inf")
        remaining_patience = self.patience
        step = 0
        validation_x = validation_y = None
        if validation is not None:
            validation_x = np.asarray(validation[0], dtype=np.float64)
            validation_y = self._normalize_condition(np.asarray(validation[1], dtype=np.float64))

        for epoch in range(1, self.epochs + 1):
            for indices in np.array_split(rng.permutation(len(x)), max(1, len(x) // self.batch_size)):
                batch_x = x[indices]
                batch_condition = normalized_conditions[indices]
                epsilon = rng.normal(size=(len(indices), self.latent_size))
                probabilities, cache = self._forward(batch_x, batch_condition, epsilon)
                flattened, encoder_input, encoder_linear, encoder_hidden, mu, logvar_raw, logvar, latent, decoder_input, decoder_linear, decoder_hidden = cache
                target = batch_x
                reconstruction = -np.sum(target * np.log(probabilities + 1e-12)) / len(indices)
                kl_divergence = 0.5 * np.sum(np.exp(logvar) + mu**2 - 1.0 - logvar) / len(indices)
                logits_gradient = (probabilities - target) / len(indices)
                logits_flat_gradient = logits_gradient.reshape(len(indices), -1)
                gradients = {
                    "output_weight": decoder_hidden.T @ logits_flat_gradient,
                    "output_bias": logits_flat_gradient.sum(axis=0),
                }
                decoder_hidden_gradient = (logits_flat_gradient @ self.parameters["output_weight"].T) * (decoder_linear > 0)
                gradients["decoder_weight"] = decoder_input.T @ decoder_hidden_gradient
                gradients["decoder_bias"] = decoder_hidden_gradient.sum(axis=0)
                latent_gradient = (decoder_hidden_gradient @ self.parameters["decoder_weight"].T)[:, : self.latent_size]
                mu_gradient = latent_gradient + self.beta * mu / len(indices)
                logvar_gradient = latent_gradient * epsilon * 0.5 * np.exp(0.5 * logvar) + self.beta * 0.5 * (np.exp(logvar) - 1.0) / len(indices)
                logvar_gradient *= (logvar_raw > -8.0) & (logvar_raw < 4.0)
                gradients["mu_weight"] = encoder_hidden.T @ mu_gradient
                gradients["mu_bias"] = mu_gradient.sum(axis=0)
                gradients["logvar_weight"] = encoder_hidden.T @ logvar_gradient
                gradients["logvar_bias"] = logvar_gradient.sum(axis=0)
                encoder_hidden_gradient = (mu_gradient @ self.parameters["mu_weight"].T + logvar_gradient @ self.parameters["logvar_weight"].T) * (encoder_linear > 0)
                gradients["encoder_weight"] = encoder_input.T @ encoder_hidden_gradient
                gradients["encoder_bias"] = encoder_hidden_gradient.sum(axis=0)
                step += 1
                for name, gradient in gradients.items():
                    first[name] = 0.9 * first[name] + 0.1 * gradient
                    second[name] = 0.999 * second[name] + 0.001 * gradient**2
                    corrected_first = first[name] / (1 - 0.9**step)
                    corrected_second = second[name] / (1 - 0.999**step)
                    self.parameters[name] -= self.learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            train_probabilities, train_cache = self._forward(x, normalized_conditions, None)
            train_mu, train_logvar = train_cache[4], train_cache[6]
            train_reconstruction = float(-np.sum(x * np.log(train_probabilities + 1e-12)) / len(x))
            train_kl = float(0.5 * np.sum(np.exp(train_logvar) + train_mu**2 - 1.0 - train_logvar) / len(x))
            record = {"epoch": float(epoch), "train_reconstruction": train_reconstruction, "train_kl": train_kl}
            if validation_x is not None and validation_y is not None:
                validation_probabilities, validation_cache = self._forward(validation_x, validation_y, None)
                validation_mu, validation_logvar = validation_cache[4], validation_cache[6]
                validation_reconstruction = float(-np.sum(validation_x * np.log(validation_probabilities + 1e-12)) / len(validation_x))
                validation_kl = float(0.5 * np.sum(np.exp(validation_logvar) + validation_mu**2 - 1.0 - validation_logvar) / len(validation_x))
                validation_loss = validation_reconstruction + self.beta * validation_kl
                record.update({"validation_reconstruction": validation_reconstruction, "validation_kl": validation_kl, "validation_loss": validation_loss})
                if validation_loss < best_validation - 1e-6:
                    best_validation = validation_loss
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

    def sample(self, conditions: np.ndarray, seed: int) -> np.ndarray:
        if self.parameters is None:
            raise RuntimeError("VAE has not been fitted")
        condition_values = np.asarray(conditions, dtype=np.float64)
        normalized_conditions = self._normalize_condition(condition_values)
        rng = np.random.default_rng(seed)
        latent = rng.normal(size=(len(condition_values), self.latent_size))
        decoder_input = np.concatenate([latent, normalized_conditions], axis=1)
        decoder_hidden = np.maximum(decoder_input @ self.parameters["decoder_weight"] + self.parameters["decoder_bias"], 0.0)
        logits = (decoder_hidden @ self.parameters["output_weight"] + self.parameters["output_bias"]).reshape(len(condition_values), self.sequence_length, 4)
        probabilities = self._softmax(logits)
        generated = np.empty(len(condition_values), dtype=f"U{self.sequence_length}")
        alphabet = np.array(list("ACGT"))
        for row, distribution in enumerate(probabilities):
            generated[row] = "".join(rng.choice(alphabet, p=position) for position in distribution)
        return generated

    def save(self, path: str | Path) -> None:
        if self.parameters is None or self.condition_mean is None or self.condition_scale is None:
            raise RuntimeError("Cannot save an unfitted VAE")
        np.savez_compressed(Path(path), sequence_length=self.sequence_length, hidden_size=self.hidden_size, latent_size=self.latent_size, condition_mean=self.condition_mean, condition_scale=self.condition_scale, **self.parameters)
