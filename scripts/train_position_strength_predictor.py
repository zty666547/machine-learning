#!/usr/bin/env python3
"""Train the M2 position-aware promoter-strength predictor on the fixed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import (
    PositionKmerFeaturizer,
    load_fixed_split_indices,
    load_promoter_arrays,
)
from promoter_ml.logging_utils import configure_logging
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import RidgeStrengthPredictor
from promoter_ml.reproducibility import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-dir", default="outputs/position_strength_predictor")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    return parser.parse_args()


def resolve_split_file(setting: str | None) -> Path:
    if not setting:
        raise ValueError("A fixed split file is required for the position-aware predictor")
    path = Path(setting)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def grouped_metrics(targets: np.ndarray, predictions: np.ndarray, boundaries: np.ndarray) -> dict[str, dict[str, float]]:
    names = ("weak", "medium", "strong")
    masks = (
        targets < boundaries[0],
        (targets >= boundaries[0]) & (targets < boundaries[1]),
        targets >= boundaries[1],
    )
    return {
        name: regression_metrics(targets[mask], predictions[mask])
        for name, mask in zip(names, masks)
        if int(mask.sum()) >= 2
    }


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    data_config = config["data"]
    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    logger = configure_logging(
        "train_position_strength_predictor",
        REPOSITORY_ROOT / config["logging"]["directory"],
        filename=config["logging"]["filename"],
        level=config["logging"]["level"],
    )
    sequences, strengths = load_promoter_arrays(
        args.data_dir,
        sequence_file=data_config["sequence_file"],
        label_file=data_config["label_file"],
        sequence_length=int(data_config["sequence_length"]),
    )
    train_idx, validation_idx, test_idx = load_fixed_split_indices(
        resolve_split_file(args.split_file or data_config.get("split_file")), len(sequences)
    )
    targets = np.log10(strengths)
    featurizer = PositionKmerFeaturizer(sequence_length=int(data_config["sequence_length"]))
    logger.info("Building %d position-aware features", featurizer.feature_count)
    features = featurizer.transform(sequences)

    validation_results = []
    for alpha in args.alphas:
        model = RidgeStrengthPredictor(alpha=alpha).fit(features[train_idx], targets[train_idx])
        metrics = regression_metrics(targets[validation_idx], model.predict(features[validation_idx]))
        validation_results.append({"alpha": alpha, **metrics})
        logger.info("Validation metrics at alpha=%s: %s", alpha, metrics)
    selected = min(validation_results, key=lambda result: (result["mae"], result["rmse"]))
    model = RidgeStrengthPredictor(alpha=float(selected["alpha"])).fit(features[train_idx], targets[train_idx])
    validation_predictions = model.predict(features[validation_idx])
    test_predictions = model.predict(features[test_idx])
    boundaries = np.quantile(targets[train_idx], [1 / 3, 2 / 3])
    test_metrics = regression_metrics(targets[test_idx], test_predictions)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.npz")
    result = {
        "status": "position_aware_strength_predictor_candidate",
        "seed": seed,
        "split_file": str(resolve_split_file(args.split_file or data_config.get("split_file"))),
        "feature_design": {
            "position_one_hot": 200,
            "position_dinucleotide": 784,
            "global_kmer_and_gc": 85,
            "total": featurizer.feature_count,
        },
        "validation_alpha_search": validation_results,
        "selected_alpha": selected["alpha"],
        "validation": regression_metrics(targets[validation_idx], validation_predictions),
        "test": test_metrics,
        "test_by_strength_group": grouped_metrics(targets[test_idx], test_predictions, boundaries),
        "training_log10_tertiles": [float(value) for value in boundaries],
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Selected alpha=%s; test metrics=%s", selected["alpha"], test_metrics)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
