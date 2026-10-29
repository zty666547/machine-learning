#!/usr/bin/env python3
"""Compare motif CNN position-bin counts with group-separated training CV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays
from promoter_ml.models import MotifCNNStrengthPredictor
from evaluate_strength_calibration import cluster_ids, group_folds
from analyze_frozen_motif_cnn import split_diagnostic
from train_motif_cnn_predictor import one_hot_sequences

BINS = (5, 10, 20)


def fold_layout(groups: np.ndarray, fold_seed: int) -> tuple[list[tuple[np.ndarray, ...]], tuple[np.ndarray, ...]]:
    """Keep all positions in the same cluster on one side of every split."""
    outer = group_folds(groups, 3, fold_seed)
    parts = []
    for k in range(3):
        outer_train, holdout = np.flatnonzero(outer != k), np.flatnonzero(outer == k)
        inner = group_folds(groups[outer_train], 5, fold_seed + 10 + k)
        fit, early = outer_train[inner != 0], outer_train[inner == 0]
        if set(groups[fit]) & set(groups[early]) or set(groups[outer_train]) & set(groups[holdout]):
            raise ValueError("A similarity cluster crosses an internal split")
        parts.append((fit, early, holdout))
    final_split = group_folds(groups, 5, fold_seed + 99)
    final_fit, final_early = np.flatnonzero(final_split != 0), np.flatnonzero(final_split == 0)
    if set(groups[final_fit]) & set(groups[final_early]):
        raise ValueError("A similarity cluster crosses the final training split")
    return parts, (final_fit, final_early)


def make_predictor(config: dict, seed: int, bins: int) -> MotifCNNStrengthPredictor:
    settings = dict(config)
    settings["location_bins"] = bins
    return MotifCNNStrengthPredictor(seed=seed, **settings)


def fit_model(features, targets, fit, early, config, seed, bins):
    model = make_predictor(config, seed, bins)
    model.fit(features[fit], targets[fit], validation=(features[early], targets[early]))
    parameter_count = sum(value.size for value in model.parameters.values())
    return model, {"fit_count": len(fit), "early_stop_count": len(early),
                   "epochs_run": len(model.history), "parameter_count": parameter_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, required=True, help="Model initialization and minibatch seed")
    parser.add_argument("--fold-seed", type=int, default=20261029)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/m2_motif_cnn_selected.toml")
    data = cfg["data"]
    seqs, strengths = load_promoter_arrays(args.data_dir, data["sequence_file"], data["label_file"], 50)
    train, validation, _ = load_fixed_split_indices(ROOT / data["split_file"], len(seqs))
    clusters = cluster_ids(ROOT / data["split_file"], len(seqs))
    if set(clusters[train]) & set(clusters[validation]):
        raise ValueError("A cluster crosses train and validation")
    selected = np.concatenate([train, validation])
    n_train = len(train)
    features = one_hot_sequences(seqs[selected])
    targets = np.log10(strengths[selected])
    groups = clusters[train]
    boundaries = np.quantile(targets[:n_train], [1 / 3, 2 / 3])
    layouts, final_layout = fold_layout(groups, args.fold_seed)
    train_indices = np.arange(n_train)
    val_indices = np.arange(n_train, len(selected))
    variants = {}
    # All choices are evaluated with the same folds before validation comparison.
    for bins in BINS:
        oof = np.empty(n_train)
        fold_records = []
        for i, (fit, early, holdout) in enumerate(layouts):
            print(f"seed={args.seed} bins={bins} fold={i+1}/3", flush=True)
            model, info = fit_model(features, targets, fit, early, cfg["motif_cnn"], args.seed, bins)
            oof[holdout] = model.predict(features[holdout])
            fold_records.append({**info, "holdout_count": len(holdout)})
        final, final_info = fit_model(features, targets, *final_layout, cfg["motif_cnn"], args.seed, bins)
        variants[str(bins)] = {
            "training_oof": split_diagnostic(targets[train_indices], oof, boundaries),
            "folds": fold_records,
            "validation": split_diagnostic(targets[val_indices], final.predict(features[val_indices]), boundaries),
            "final_fit": final_info,
        }
    result = {
        "protocol": "Training-only cluster-separated outer 3-fold OOF, with inner 5-way cluster split for CNN early stopping. "
                    "Final CNN uses only train-internal early stopping; fixed development validation is previously reused. "
                    "Model choices are compared across seeds by training OOF MAE. Test rows are loaded only for data format "
                    "validation and excluded from model features, fitting, selection and metrics.",
        "model_seed": args.seed, "fold_seed": args.fold_seed,
        "train_count": n_train, "validation_count": len(val_indices),
        "model_config": cfg["motif_cnn"], "bins": list(BINS),
        "training_boundaries": boundaries.tolist(), "variants": variants,
    }
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {"oof_mae": v["training_oof"]["mae"], "validation_mae": v["validation"]["mae"]}
                      for k, v in variants.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
