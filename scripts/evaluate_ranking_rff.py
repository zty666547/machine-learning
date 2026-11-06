#!/usr/bin/env python3
"""Compare pairwise-trained RNN and nonlinear sequence-feature ridge."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.config import load_config
from promoter_ml.data import PositionKmerFeaturizer, load_fixed_split_indices, load_promoter_arrays
from promoter_ml.models import RandomFourierRidge, RidgeStrengthPredictor, RNNStrengthPredictor
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from analyze_frozen_motif_cnn import split_diagnostic
from train_motif_cnn_predictor import one_hot_sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()
    path = ROOT / "configs/m4_ranking_rff_2026-11-06.toml"
    with path.open("rb") as stream: protocol = tomllib.load(stream)
    if args.seed not in protocol["experiment"]["seeds"] or protocol["experiment"]["outer_folds"] != 3:
        raise ValueError("Seed or folds differ from frozen protocol")
    if protocol["rff_ridge"]["gamma"] != "inverse_mean_squared_distance_of_2048_random_fit_only_pairs":
        raise ValueError("Unexpected Fourier bandwidth rule")
    cfg = load_config(ROOT / "configs/m2_motif_cnn_selected.toml")["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, cfg["sequence_file"], cfg["label_file"], 50)
    train_idx, val_idx, _ = load_fixed_split_indices(ROOT / cfg["split_file"], len(sequences))
    clusters = cluster_ids(ROOT / cfg["split_file"], len(sequences))
    if set(clusters[train_idx]) & set(clusters[val_idx]): raise ValueError("Cluster crosses train/validation")
    data_sha = {"sequence": hashlib.sha256((Path(args.data_dir) / cfg["sequence_file"]).read_bytes()).hexdigest(),
                "strength": hashlib.sha256((Path(args.data_dir) / cfg["label_file"]).read_bytes()).hexdigest(),
                "split": hashlib.sha256((ROOT / cfg["split_file"]).read_bytes()).hexdigest()}
    prior_path = ROOT / f"reports/m4_gru_loss_seed_{args.seed}.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (prior["data_sha256"] != data_sha or prior["fold_seed"] != protocol["experiment"]["fold_seed"]
            or prior["seed"] != args.seed or prior["numpy_version"] != np.__version__):
        raise ValueError("Prior model data, folds, seed or NumPy differs")
    prior_config = ROOT / "configs/m4_gru_loss_2026-11-06.toml"
    if hashlib.sha256(prior_config.read_bytes()).hexdigest() != prior["protocol_sha256"]:
        raise ValueError("Prior multi-strategy configuration differs")
    selected = np.concatenate([train_idx, val_idx])
    n = len(train_idx)
    y = np.log10(strengths[selected])
    layouts, final_layout = fold_layout(clusters[train_idx], protocol["experiment"]["fold_seed"])
    boundaries = np.quantile(y[:n], [1/3, 2/3])
    if not np.array_equal(boundaries, prior["training_boundaries"]):
        raise ValueError("Prior strength boundaries differ")
    result = {
        "protocol": "Pairwise RNN: same fit, early-stop and outer folds as previous RNN; target gap threshold is on standardized log targets. "
                    "Linear and Fourier ridge: same outer holdout, but use all outer-training rows because no early stopping; "
                    "same features and fixed alpha for the paired linear/Fourier comparison. Fourier bandwidth and feature scaling fit only on each training fold. "
                    "Development validation is previously reused. Test rows are loaded only for format checks, not featurized or evaluated.",
        "seed": args.seed, "fold_seed": protocol["experiment"]["fold_seed"], "train_count": n, "validation_count": len(val_idx),
        "data_sha256": data_sha, "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prior_result_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "training_boundaries": boundaries.tolist(),
        "references": {"cnn": prior["prior_models"]["cnn"], "rnn_mse": prior["prior_models"]["rnn"],
                       "rnn_huber": prior["models"]["rnn_huber"], "gru_mse": prior["models"]["gru"]},
        "models": {},
    }
    onehot = one_hot_sequences(sequences[selected])
    oof = np.empty(n)
    folds = []
    for i, (fit, early, holdout) in enumerate(layouts):
        print(f"seed={args.seed} model=ranking_rnn fold={i+1}/3", flush=True)
        model = RNNStrengthPredictor(seed=args.seed, **protocol["ranking_rnn"])
        model.fit(onehot[fit], y[fit], validation=(onehot[early], y[early]))
        oof[holdout] = model.predict(onehot[holdout])
        folds.append({"fit_count": len(fit), "early_stop_count": len(early), "holdout_count": len(holdout),
                      "epochs_run": len(model.history), "parameter_count": sum(v.size for v in model.parameters.values()),
                      "clipped_batches": sum(v["clipped_batches"] for v in model.history)})
    fit, early = final_layout
    model = RNNStrengthPredictor(seed=args.seed, **protocol["ranking_rnn"])
    print(f"seed={args.seed} model=ranking_rnn final", flush=True)
    model.fit(onehot[fit], y[fit], validation=(onehot[early], y[early]))
    result["models"]["ranking_rnn"] = {"training_oof": split_diagnostic(y[:n], oof, boundaries), "folds": folds,
                                        "development_validation": split_diagnostic(y[n:], model.predict(onehot[n:]), boundaries),
                                        "final_fit": {"fit_count": len(fit), "early_stop_count": len(early), "epochs_run": len(model.history),
                                                      "parameter_count": sum(v.size for v in model.parameters.values())}}
    if args.checkpoint_dir:
        directory = Path(args.checkpoint_dir)
        if not directory.is_absolute(): directory = ROOT / directory
        directory.mkdir(parents=True, exist_ok=True)
        model.save(directory / f"ranking_rnn_seed_{args.seed}.npz")
    del onehot
    print(f"seed={args.seed} computing position k-mer features", flush=True)
    features = PositionKmerFeaturizer(sequence_length=50).transform(sequences[selected])
    for name in ("linear_ridge", "fourier_ridge"):
        oof = np.empty(n)
        folds = []
        for i, (fit, early, holdout) in enumerate(layouts):
            outer_train = np.sort(np.concatenate([fit, early]))
            print(f"seed={args.seed} model={name} fold={i+1}/3", flush=True)
            if name == "linear_ridge":
                model = RidgeStrengthPredictor(alpha=protocol["rff_ridge"]["alpha"])
            else:
                model = RandomFourierRidge(features=protocol["rff_ridge"]["features"],
                                           alpha=protocol["rff_ridge"]["alpha"], seed=args.seed + 100 * i)
            model.fit(features[outer_train], y[outer_train])
            oof[holdout] = model.predict(features[holdout])
            folds.append({"fit_count": len(outer_train), "holdout_count": len(holdout),
                          "gamma": model.gamma if name == "fourier_ridge" else None})
        if name == "linear_ridge": model = RidgeStrengthPredictor(alpha=protocol["rff_ridge"]["alpha"])
        else: model = RandomFourierRidge(features=protocol["rff_ridge"]["features"],
                                         alpha=protocol["rff_ridge"]["alpha"], seed=args.seed + 999)
        print(f"seed={args.seed} model={name} final", flush=True)
        model.fit(features[:n], y[:n])
        result["models"][name] = {"training_oof": split_diagnostic(y[:n], oof, boundaries), "folds": folds,
                                  "development_validation": split_diagnostic(y[n:], model.predict(features[n:]), boundaries),
                                  "final_fit": {"fit_count": n, "gamma": model.gamma if name == "fourier_ridge" else None}}
        if name == "fourier_ridge" and args.checkpoint_dir:
            model.save(directory / f"fourier_ridge_seed_{args.seed}.npz")
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {"oof_mae": v["training_oof"]["mae"], "oof_spearman": v["training_oof"]["spearman"],
                             "validation_mae": v["development_validation"]["mae"]}
                      for name, v in result["models"].items()}, indent=2))


if __name__ == "__main__": main()
