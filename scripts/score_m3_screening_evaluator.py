#!/usr/bin/env python3
"""Score generated 50 bp promoter candidates with the frozen M3 screening evaluator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import DNA_ALPHABET, KmerFeaturizer, PositionKmerFeaturizer, load_fixed_split_indices, load_promoter_arrays
from promoter_ml.models import MotifCNNStrengthPredictor, RidgeStrengthPredictor
from promoter_ml.reproducibility import set_random_seed
from train_motif_cnn_predictor import one_hot_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m2_motif_cnn_selected.toml")
    parser.add_argument("--evaluator-config", default="configs/m3_screening_evaluator.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--candidate-file", required=True, help="One 50 bp A/C/G/T sequence per line.")
    parser.add_argument("--output", default="outputs/m3_screening_scores.csv")
    return parser.parse_args()


def load_candidates(path: str) -> np.ndarray:
    values = np.array([line.strip().upper() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()])
    if len(values) == 0 or any(len(value) != 50 or set(value) - set(DNA_ALPHABET) for value in values):
        raise ValueError("Candidate file must contain one valid 50 bp A/C/G/T sequence per line")
    return values


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    with (REPOSITORY_ROOT / args.evaluator_config).open("rb") as handle:
        evaluator_config = tomllib.load(handle)["screening_evaluator"]
    weights = np.array([
        float(evaluator_config["motif_cnn_weight"]),
        float(evaluator_config["global_kmer_ridge_weight"]),
        float(evaluator_config["position_ridge_weight"]),
    ])
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Screening evaluator weights must sum to 1")
    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    targets = np.log10(strengths)
    candidates = load_candidates(args.candidate_file)
    all_sequences = np.concatenate([sequences, candidates])
    global_features = KmerFeaturizer(k_min=1, k_max=3, include_gc=True).transform(all_sequences)
    position_features = PositionKmerFeaturizer(sequence_length=50).transform(all_sequences)
    cnn_features = one_hot_sequences(all_sequences)
    global_ridge = RidgeStrengthPredictor(alpha=10.0).fit(global_features[train_idx], targets[train_idx])
    position_ridge = RidgeStrengthPredictor(alpha=10.0).fit(position_features[train_idx], targets[train_idx])
    cnn = MotifCNNStrengthPredictor(seed=seed, **config["motif_cnn"])
    cnn.fit(cnn_features[train_idx], targets[train_idx], validation=(cnn_features[validation_idx], targets[validation_idx]))
    candidate_slice = slice(len(sequences), len(all_sequences))
    components = np.column_stack([
        cnn.predict(cnn_features[candidate_slice]),
        global_ridge.predict(global_features[candidate_slice]),
        position_ridge.predict(position_features[candidate_slice]),
    ])
    score = components @ weights
    boundaries = np.quantile(targets[train_idx], [1 / 3, 2 / 3])
    labels = np.array(["weak", "medium", "strong"])[np.digitize(score, boundaries, right=False)]
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "screening_log_strength", "screening_group", "motif_cnn_score", "global_kmer_ridge_score", "position_ridge_score"])
        writer.writeheader()
        for sequence, combined, group, component in zip(candidates, score, labels, components):
            writer.writerow({"sequence": sequence, "screening_log_strength": float(combined), "screening_group": group, "motif_cnn_score": float(component[0]), "global_kmer_ridge_score": float(component[1]), "position_ridge_score": float(component[2])})
    metadata = {
        "candidate_count": len(candidates),
        "weights": {"motif_cnn": weights[0], "global_kmer_ridge": weights[1], "position_ridge": weights[2]},
        "training_tertile_boundaries": boundaries.tolist(),
        "interpretation_boundary": evaluator_config["interpretation"],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
