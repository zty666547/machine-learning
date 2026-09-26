#!/usr/bin/env python3
"""Run a dependency-light end-to-end check using synthetic DNA data."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.data import KmerFeaturizer, PositionKmerFeaturizer
from promoter_ml.generation import ContinuousConditionalAutoregressiveGenerator, ConditionalAutoregressiveGenerator, ConditionalPositionFrequencyGenerator, PositionFrequencyGenerator
from promoter_ml.logging_utils import configure_logging
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import RidgeStrengthPredictor
from promoter_ml.sequence_metrics import generation_metrics


def main() -> None:
    rng = np.random.default_rng(7)
    alphabet = np.array(list("ACGT"))
    sequences = np.array(["".join(rng.choice(alphabet, size=50)) for _ in range(200)])
    targets = np.array([(seq.count("G") + seq.count("C")) / 50 for seq in sequences])
    featurizer = KmerFeaturizer(k_min=1, k_max=2, include_gc=True)
    features = featurizer.transform(sequences)
    model = RidgeStrengthPredictor(alpha=1.0).fit(features[:150], targets[:150])
    predictions = model.predict(features[150:])
    metrics = regression_metrics(targets[150:], predictions)

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.npz"
        model.save(model_path)
        restored = RidgeStrengthPredictor.load(model_path)
        if not np.allclose(predictions, restored.predict(features[150:])):
            raise AssertionError("Saved and loaded predictions do not match")
        logger = configure_logging("smoke_test", Path(directory) / "logs")
        logger.info("Smoke test metrics: %s", metrics)

    if not np.all(np.isfinite(predictions)) or metrics["pearson"] < 0.95:
        raise AssertionError(f"Unexpected smoke test result: {metrics}")
    position_features = PositionKmerFeaturizer(sequence_length=50).transform(sequences[:2])
    if position_features.shape != (2, 1069) or not np.allclose(position_features[:, :200].sum(axis=1), 50):
        raise AssertionError("Position-aware feature encoding failed")
    generated = PositionFrequencyGenerator().fit(sequences[:150]).sample(20, seed=7)
    generation_result = generation_metrics(generated, sequences[:150])
    if generation_result["valid_fraction"] != 1.0 or len(generated) != 20:
        raise AssertionError("Generation baseline failed")
    conditions = np.array(["weak"] * 75 + ["strong"] * 75)
    conditional_generated = ConditionalPositionFrequencyGenerator().fit(sequences[:150], conditions).sample("strong", 10, seed=8)
    if len(conditional_generated) != 10 or any(len(sequence) != 50 for sequence in conditional_generated):
        raise AssertionError("Conditional generation baseline failed")
    autoregressive_generated = ConditionalAutoregressiveGenerator(order=2).fit(sequences[:150], conditions).sample("weak", 10, seed=9)
    if len(autoregressive_generated) != 10 or any(len(sequence) != 50 for sequence in autoregressive_generated):
        raise AssertionError("Autoregressive generation baseline failed")
    continuous_generated = ContinuousConditionalAutoregressiveGenerator(order=2, condition_bins=3).fit(sequences[:150], targets[:150]).sample(0.4, 10, seed=10)
    if len(continuous_generated) != 10 or any(len(sequence) != 50 for sequence in continuous_generated):
        raise AssertionError("Continuous autoregressive generation failed")
    print("Smoke test passed")


if __name__ == "__main__":
    main()
