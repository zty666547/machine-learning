#!/usr/bin/env python3
"""Train a nonlinear position-aware MLP strength predictor on the fixed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import PositionKmerFeaturizer, load_fixed_split_indices, load_promoter_arrays
from promoter_ml.logging_utils import configure_logging
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import PositionMLPStrengthPredictor
from promoter_ml.reproducibility import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/position_mlp_strength_predictor")
    parser.add_argument("--epochs", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    logger = configure_logging("train_position_mlp_predictor", REPOSITORY_ROOT / config["logging"]["directory"])
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    split_path = REPOSITORY_ROOT / data_config["split_file"]
    train_idx, validation_idx, test_idx = load_fixed_split_indices(split_path, len(sequences))
    targets = np.log10(strengths)
    featurizer = PositionKmerFeaturizer(sequence_length=50, include_global_features=False, include_dinucleotide_features=False)
    features = featurizer.transform(sequences)
    predictor = PositionMLPStrengthPredictor(epochs=args.epochs, seed=seed)
    predictor.fit(features[train_idx], targets[train_idx], validation=(features[validation_idx], targets[validation_idx]))
    validation_prediction = predictor.predict(features[validation_idx])
    test_prediction = predictor.predict(features[test_idx])
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(output_dir / "model.npz")
    metrics = {
        "status": "nonlinear_position_aware_predictor_candidate",
        "feature_count": featurizer.feature_count,
        "validation": regression_metrics(targets[validation_idx], validation_prediction),
        "test": regression_metrics(targets[test_idx], test_prediction),
        "epochs_completed": len(predictor.history),
        "best_validation_mae": min(row.get("validation_mae", float("inf")) for row in predictor.history),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Position MLP metrics: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
