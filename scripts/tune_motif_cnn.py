#!/usr/bin/env python3
"""Select a motif-CNN architecture using only the fixed validation split."""

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
from train_motif_cnn_predictor import one_hot_sequences


SEARCH_SPACE = (
    {"name": "cnn_k5_f32", "kernel_size": 5, "filters": 32, "location_bins": 5, "hidden_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4},
    {"name": "cnn_k7_f32", "kernel_size": 7, "filters": 32, "location_bins": 5, "hidden_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4},
    {"name": "cnn_k9_f32", "kernel_size": 9, "filters": 32, "location_bins": 5, "hidden_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4},
    {"name": "cnn_k7_f64", "kernel_size": 7, "filters": 64, "location_bins": 5, "hidden_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m2_motif_cnn_validation_search.json")
    parser.add_argument("--epochs", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    logger = configure_logging("tune_motif_cnn", REPOSITORY_ROOT / config["logging"]["directory"])
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    features = one_hot_sequences(sequences)
    targets = np.log10(strengths)
    results = []
    for setting in SEARCH_SPACE:
        model = MotifCNNStrengthPredictor(**{key: value for key, value in setting.items() if key != "name"}, epochs=args.epochs, seed=seed)
        model.fit(features[train_idx], targets[train_idx], validation=(features[validation_idx], targets[validation_idx]))
        record = {
            **setting,
            "epochs_completed": len(model.history),
            "validation": regression_metrics(targets[validation_idx], model.predict(features[validation_idx])),
        }
        results.append(record)
        logger.info("Validation-only CNN candidate: %s", record)
    results.sort(key=lambda row: (row["validation"]["mae"], -row["validation"]["pearson"]))
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"selection_rule": "lowest validation MAE, then highest validation Pearson", "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": results[0], "candidates": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
