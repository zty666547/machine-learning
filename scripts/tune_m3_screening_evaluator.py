#!/usr/bin/env python3
"""Select a validation-only blended sequence-strength screening evaluator for M3."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import KmerFeaturizer, PositionKmerFeaturizer, load_fixed_split_indices, load_promoter_arrays
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import MotifCNNStrengthPredictor, RidgeStrengthPredictor
from promoter_ml.reproducibility import set_random_seed
from train_motif_cnn_predictor import one_hot_sequences


MODEL_NAMES = ("motif_cnn", "global_kmer_ridge", "position_ridge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m2_motif_cnn_selected.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m3_screening_evaluator_tuning.json")
    parser.add_argument("--weight-step", type=float, default=0.1)
    return parser.parse_args()


def candidate_weights(step: float) -> list[tuple[float, float, float]]:
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("weight-step must divide 1")
    return [(cnn / units, global_ridge / units, (units - cnn - global_ridge) / units) for cnn in range(units + 1) for global_ridge in range(units - cnn + 1)]


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    data_config = config["data"]
    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    targets = np.log10(strengths)

    global_features = KmerFeaturizer(k_min=1, k_max=3, include_gc=True).transform(sequences)
    position_features = PositionKmerFeaturizer(sequence_length=50).transform(sequences)
    global_ridge = RidgeStrengthPredictor(alpha=10.0).fit(global_features[train_idx], targets[train_idx])
    position_ridge = RidgeStrengthPredictor(alpha=10.0).fit(position_features[train_idx], targets[train_idx])
    cnn = MotifCNNStrengthPredictor(seed=seed, **config["motif_cnn"])
    cnn_features = one_hot_sequences(sequences)
    cnn.fit(cnn_features[train_idx], targets[train_idx], validation=(cnn_features[validation_idx], targets[validation_idx]))
    predictions = {
        "motif_cnn": cnn.predict(cnn_features[validation_idx]),
        "global_kmer_ridge": global_ridge.predict(global_features[validation_idx]),
        "position_ridge": position_ridge.predict(position_features[validation_idx]),
    }
    base_metrics = {name: regression_metrics(targets[validation_idx], values) for name, values in predictions.items()}
    weight_trials = []
    for weights in candidate_weights(args.weight_step):
        blended = sum(weight * predictions[name] for weight, name in zip(weights, MODEL_NAMES))
        weight_trials.append({"weights": dict(zip(MODEL_NAMES, weights)), **regression_metrics(targets[validation_idx], blended)})
    weight_trials.sort(key=lambda row: (-row["pearson"], row["mae"], row["rmse"]))
    best = weight_trials[0]
    result = {
        "selection_data": "validation set only; the test set is not loaded or evaluated",
        "base_validation_metrics": base_metrics,
        "blend_weight_grid_step": args.weight_step,
        "best_blend": best,
        "top_blends": weight_trials[:10],
        "interpretation_boundary": "The selected blend is an M3 screening score for relative candidate prioritization. It is not an independently validated measurement of real expression strength.",
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
