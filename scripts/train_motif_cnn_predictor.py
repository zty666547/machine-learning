#!/usr/bin/env python3
"""Train a compact position-aware motif CNN on the fixed E. coli split."""

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
from promoter_ml.models import MotifCNNStrengthPredictor
from promoter_ml.reproducibility import set_random_seed


def one_hot_sequences(sequences: np.ndarray) -> np.ndarray:
    mapping = {base: index for index, base in enumerate("ACGT")}
    encoded = np.zeros((len(sequences), len(sequences[0]), 4), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        encoded[row, np.arange(len(sequence)), [mapping[base] for base in sequence]] = 1.0
    return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/motif_cnn_strength_predictor")
    parser.add_argument("--epochs", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    logger = configure_logging("train_motif_cnn_predictor", REPOSITORY_ROOT / config["logging"]["directory"])
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, test_idx = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    features = one_hot_sequences(sequences)
    targets = np.log10(strengths)
    predictor = MotifCNNStrengthPredictor(epochs=args.epochs, seed=seed)
    predictor.fit(features[train_idx], targets[train_idx], validation=(features[validation_idx], targets[validation_idx]))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(output_dir / "model.npz")
    metrics = {
        "status": "position_aware_motif_cnn_candidate",
        "architecture": "7-bp convolution, 32 filters, five location bins, 64-unit ReLU head",
        "validation": regression_metrics(targets[validation_idx], predictor.predict(features[validation_idx])),
        "test": regression_metrics(targets[test_idx], predictor.predict(features[test_idx])),
        "epochs_completed": len(predictor.history),
        "best_validation_mae": min(row.get("validation_mae", float("inf")) for row in predictor.history),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Motif CNN metrics: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
