#!/usr/bin/env python3
"""Diagnose errors of the frozen motif CNN without changing its architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import MotifCNNStrengthPredictor
from promoter_ml.reproducibility import set_random_seed
from train_motif_cnn_predictor import one_hot_sequences


GROUPS = ("weak", "medium", "strong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m2_motif_cnn_selected.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m2_frozen_motif_cnn_diagnostic.json")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the frozen model once on the held-out test set.")
    return parser.parse_args()


def assign_groups(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    return np.asarray(GROUPS, dtype="U6")[np.digitize(values, boundaries, right=True)]


def split_diagnostic(targets: np.ndarray, predictions: np.ndarray, boundaries: np.ndarray) -> dict:
    residuals = predictions - targets
    groups = assign_groups(targets, boundaries)
    by_group = {}
    for group in GROUPS:
        mask = groups == group
        by_group[group] = {
            "sample_count": int(mask.sum()),
            **regression_metrics(targets[mask], predictions[mask]),
            "mean_true_log_strength": float(targets[mask].mean()),
            "mean_predicted_log_strength": float(predictions[mask].mean()),
            "mean_signed_error": float(residuals[mask].mean()),
        }
    return {
        **regression_metrics(targets, predictions),
        "target_std": float(targets.std()),
        "prediction_std": float(predictions.std()),
        "prediction_to_target_std_ratio": float(predictions.std() / targets.std()),
        "residual_vs_target_pearson": float(np.corrcoef(residuals, targets)[0, 1]),
        "by_true_strength_group": by_group,
    }


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    data_config = config["data"]
    model_config = config["motif_cnn"]
    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, test_idx = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    targets = np.log10(strengths)
    features = one_hot_sequences(sequences)
    predictor = MotifCNNStrengthPredictor(seed=seed, **model_config)
    predictor.fit(features[train_idx], targets[train_idx], validation=(features[validation_idx], targets[validation_idx]))
    boundaries = np.quantile(targets[train_idx], [1 / 3, 2 / 3])
    result = {
        "status": "frozen_architecture_post_hoc_diagnostic",
        "architecture": model_config,
        "training_strength_boundaries": boundaries.tolist(),
        "validation": split_diagnostic(targets[validation_idx], predictor.predict(features[validation_idx]), boundaries),
        "test": split_diagnostic(targets[test_idx], predictor.predict(features[test_idx]), boundaries) if args.evaluate_test else None,
        "interpretation_boundary": "This diagnostic describes a fixed M2 architecture. It must not be used to select a new architecture after viewing test metrics.",
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
