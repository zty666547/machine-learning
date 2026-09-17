#!/usr/bin/env python3
"""Train a multi-scale motif CNN, using the test split only when requested."""

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
from promoter_ml.logging_utils import configure_logging
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import MultiScaleMotifCNNStrengthPredictor
from promoter_ml.reproducibility import set_random_seed
from train_motif_cnn_predictor import one_hot_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/multiscale_motif_cnn")
    parser.add_argument("--kernel-sizes", default="5,7,9")
    parser.add_argument("--filters-per-scale", type=int, default=24)
    parser.add_argument("--location-bins", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    logger = configure_logging("train_multiscale_motif_cnn", REPOSITORY_ROOT / config["logging"]["directory"])
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, test_idx = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    features = one_hot_sequences(sequences)
    targets = np.log10(strengths)
    kernel_sizes = tuple(int(value) for value in args.kernel_sizes.split(","))
    model = MultiScaleMotifCNNStrengthPredictor(kernel_sizes=kernel_sizes, filters_per_scale=args.filters_per_scale, location_bins=args.location_bins, hidden_size=args.hidden_size, epochs=args.epochs, seed=seed)
    model.fit(features[train_idx], targets[train_idx], validation=(features[validation_idx], targets[validation_idx]))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.npz")
    metrics = {
        "status": "multiscale_position_aware_cnn_candidate",
        "architecture": {"kernel_sizes": kernel_sizes, "filters_per_scale": args.filters_per_scale, "location_bins": args.location_bins, "hidden_size": args.hidden_size},
        "validation": regression_metrics(targets[validation_idx], model.predict(features[validation_idx])),
        "test": regression_metrics(targets[test_idx], model.predict(features[test_idx])) if args.evaluate_test else None,
        "epochs_completed": len(model.history),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Multi-scale CNN metrics: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
