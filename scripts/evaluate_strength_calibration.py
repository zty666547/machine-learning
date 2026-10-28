#!/usr/bin/env python3
"""Compare cross-fitted calibration with the existing M3 screening blend."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.config import load_config
from promoter_ml.data import KmerFeaturizer, PositionKmerFeaturizer, load_fixed_split_indices, load_promoter_arrays
from promoter_ml.models import MotifCNNStrengthPredictor, RidgeStrengthPredictor
from analyze_frozen_motif_cnn import split_diagnostic
from train_motif_cnn_predictor import one_hot_sequences


def cluster_ids(path: Path, count: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != count:
        raise ValueError("Split file does not match data")
    groups = np.empty(count, dtype=np.int64)
    seen = set()
    for row in rows:
        idx = int(row["sample_index"])
        if idx < 0 or idx >= count or idx in seen:
            raise ValueError("Invalid or duplicate split index")
        seen.add(idx)
        groups[idx] = int(row["cluster_id"])
    return groups


def group_folds(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Assign complete clusters to approximately balanced folds."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < folds:
        raise ValueError("Too few clusters")
    order = np.random.default_rng(seed).permutation(len(unique))
    order = order[np.argsort(-counts[order], kind="stable")]
    totals = np.zeros(folds, dtype=np.int64)
    assignment = np.empty(len(unique), dtype=np.int64)
    for idx in order:
        chosen = int(np.argmin(totals))
        assignment[idx] = chosen
        totals[chosen] += counts[idx]
    return assignment[inverse]


def make_features(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (one_hot_sequences(sequences),
            KmerFeaturizer(k_min=1, k_max=3, include_gc=True).transform(sequences),
            PositionKmerFeaturizer(sequence_length=50).transform(sequences))


def fit_predict(features, targets, train, cnn_validation, predict, cnn_config, seed):
    cnn_features, global_features, position_features = features
    cnn = MotifCNNStrengthPredictor(seed=seed, **cnn_config)
    cnn.fit(cnn_features[train], targets[train],
            validation=(cnn_features[cnn_validation], targets[cnn_validation]))
    ridge_train = np.concatenate([train, cnn_validation])
    global_model = RidgeStrengthPredictor(alpha=10.0).fit(global_features[ridge_train], targets[ridge_train])
    position_model = RidgeStrengthPredictor(alpha=10.0).fit(position_features[ridge_train], targets[ridge_train])
    return (0.5 * cnn.predict(cnn_features[predict])
            + 0.4 * global_model.predict(global_features[predict])
            + 0.1 * position_model.predict(position_features[predict]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m4_strength_calibration_2026-10-28.json")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/m2_motif_cnn_selected.toml")
    data = cfg["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data["sequence_file"], data["label_file"], 50)
    train_idx, validation_idx, _ = load_fixed_split_indices(ROOT / data["split_file"], len(sequences))
    all_groups = cluster_ids(ROOT / data["split_file"], len(sequences))
    if set(all_groups[train_idx]) & set(all_groups[validation_idx]):
        raise ValueError("A similarity cluster crosses train and validation")
    selected = np.concatenate([train_idx, validation_idx])
    n_train = len(train_idx)
    features = make_features(sequences[selected])
    targets = np.log10(strengths[selected])
    groups = all_groups[train_idx]
    boundaries = np.quantile(targets[:n_train], [1 / 3, 2 / 3])
    fold = group_folds(groups, args.folds, args.seed)
    oof = np.empty(n_train, dtype=np.float64)
    fold_details = []
    for k in range(args.folds):
        outer_train = np.flatnonzero(fold != k)
        outer_holdout = np.flatnonzero(fold == k)
        inner_fold = group_folds(groups[outer_train], 5, args.seed + 10 + k)
        inner_train = outer_train[inner_fold != 0]
        inner_val = outer_train[inner_fold == 0]
        if set(groups[outer_holdout]) & set(groups[outer_train]) or set(groups[inner_train]) & set(groups[inner_val]):
            raise ValueError("A similarity cluster crosses an internal fold")
        print(f"Fold {k + 1}/{args.folds}: train={len(inner_train)}, early-stop={len(inner_val)}, holdout={len(outer_holdout)}", flush=True)
        oof[outer_holdout] = fit_predict(features, targets, inner_train, inner_val, outer_holdout,
                                         cfg["motif_cnn"], int(cfg["project"]["seed"]))
        fold_details.append({"train_rows": len(inner_train), "early_stop_rows": len(inner_val),
                             "holdout_rows": len(outer_holdout)})
    if not np.all(np.isfinite(oof)) or np.std(oof) < 1e-12:
        raise ValueError("OOF predictions are not finite or have zero variance")
    slope = float(np.cov(oof, targets[:n_train], ddof=0)[0, 1] / np.var(oof))
    intercept = float(targets[:n_train].mean() - slope * oof.mean())
    train = np.arange(n_train)
    val = np.arange(n_train, len(selected))
    cnn = MotifCNNStrengthPredictor(seed=int(cfg["project"]["seed"]), **cfg["motif_cnn"])
    cnn.fit(features[0][train], targets[train], validation=(features[0][val], targets[val]))
    global_model = RidgeStrengthPredictor(alpha=10.0).fit(features[1][train], targets[train])
    position_model = RidgeStrengthPredictor(alpha=10.0).fit(features[2][train], targets[train])
    baseline = (0.5 * cnn.predict(features[0][val]) + 0.4 * global_model.predict(features[1][val])
                + 0.1 * position_model.predict(features[2][val]))
    calibrated = intercept + slope * baseline
    result = {
        "protocol": "Similarity-cluster OOF folds within training; CNN early stopping within each fold. "
                    "Affine calibration fitted on OOF predictions only. Fixed validation is development data, "
                    "also used for M3 selection and final CNN early stopping. No test metrics computed.",
        "fold_seed": args.seed, "folds": fold_details, "calibration": {"intercept": intercept, "slope": slope},
        "training_oof_uncalibrated": split_diagnostic(targets[train], oof, boundaries),
        "validation_baseline": split_diagnostic(targets[val], baseline, boundaries),
        "validation_calibrated": split_diagnostic(targets[val], calibrated, boundaries),
        "training_tertile_boundaries": boundaries.tolist(),
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
