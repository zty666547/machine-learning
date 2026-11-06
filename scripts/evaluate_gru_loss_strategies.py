#!/usr/bin/env python3
"""Compare gated recurrence and robust loss with fixed previous RNN/CNN folds."""
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
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays
from promoter_ml.models import GRUStrengthPredictor, RNNStrengthPredictor
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
    path = ROOT / "configs/m4_gru_loss_2026-11-06.toml"
    with path.open("rb") as stream: protocol = tomllib.load(stream)
    experiment = protocol["experiment"]
    if args.seed not in experiment["model_seeds"] or experiment["outer_folds"] != 3:
        raise ValueError("Seed or folds differ from the frozen protocol")
    cfg = load_config(ROOT / "configs/m2_motif_cnn_selected.toml")["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, cfg["sequence_file"], cfg["label_file"], 50)
    train_idx, val_idx, _ = load_fixed_split_indices(ROOT / cfg["split_file"], len(sequences))
    clusters = cluster_ids(ROOT / cfg["split_file"], len(sequences))
    if set(clusters[train_idx]) & set(clusters[val_idx]): raise ValueError("Cluster crosses train/validation")
    data_sha = {"sequence": hashlib.sha256((Path(args.data_dir) / cfg["sequence_file"]).read_bytes()).hexdigest(),
                "strength": hashlib.sha256((Path(args.data_dir) / cfg["label_file"]).read_bytes()).hexdigest(),
                "split": hashlib.sha256((ROOT / cfg["split_file"]).read_bytes()).hexdigest()}
    prior_path = ROOT / f"reports/m4_rnn_vs_cnn_seed_{args.seed}.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (prior["data_sha256"] != data_sha or prior["fold_seed"] != experiment["fold_seed"]
            or prior["seed"] != args.seed or prior["numpy_version"] != np.__version__
            or not prior["previous_cnn_mae_reproduced"]):
        raise ValueError("Earlier baseline provenance differs from this protocol")
    prior_config = ROOT / "configs/m4_rnn_predictor_2026-11-06.toml"
    if hashlib.sha256(prior_config.read_bytes()).hexdigest() != prior["protocol_sha256"]:
        raise ValueError("Previous RNN protocol differs from recorded baseline")
    selected = np.concatenate([train_idx, val_idx])
    n = len(train_idx)
    x, y = one_hot_sequences(sequences[selected]), np.log10(strengths[selected])
    layouts, final_layout = fold_layout(clusters[train_idx], experiment["fold_seed"])
    boundaries = np.quantile(y[:n], [1/3, 2/3])
    if not np.allclose(boundaries, prior["training_boundaries"], atol=0, rtol=0):
        raise ValueError("Prior and current strength boundaries differ")
    result = {"protocol": "Same three cluster-separated outer folds, same train-only early-stop splits and model seeds "
                          "as prior CNN/RNN run. Huber delta=1 on training-subset standardized log targets; "
                          "GRU uses MSE. Development validation was previously reused; test rows are loaded only for "
                          "format checking, not featurized or evaluated. No automatic model replacement.",
              "seed": args.seed, "fold_seed": experiment["fold_seed"], "train_count": n, "validation_count": len(val_idx),
              "data_sha256": data_sha, "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "prior_result_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
              "prior_protocol_sha256": prior["protocol_sha256"],
              "numpy_version": np.__version__, "training_boundaries": boundaries.tolist(),
              "prior_models": prior["models"], "models": {}}
    for name in ("rnn_huber", "gru"):
        def make():
            cls = RNNStrengthPredictor if name == "rnn_huber" else GRUStrengthPredictor
            return cls(seed=args.seed, **protocol[name])
        oof = np.empty(n)
        folds = []
        for i, (fit, early, holdout) in enumerate(layouts):
            print(f"seed={args.seed} model={name} fold={i+1}/3", flush=True)
            model = make()
            model.fit(x[fit], y[fit], validation=(x[early], y[early]))
            oof[holdout] = model.predict(x[holdout])
            folds.append({"fit_count": len(fit), "early_stop_count": len(early), "holdout_count": len(holdout),
                          "epochs_run": len(model.history), "parameter_count": sum(v.size for v in model.parameters.values()),
                          "clipped_batches": sum(v["clipped_batches"] for v in model.history)})
        fit, early = final_layout
        final = make()
        print(f"seed={args.seed} model={name} final/development", flush=True)
        final.fit(x[fit], y[fit], validation=(x[early], y[early]))
        result["models"][name] = {
            "training_oof": split_diagnostic(y[:n], oof, boundaries),
            "folds": folds,
            "development_validation": split_diagnostic(y[n:], final.predict(x[n:]), boundaries),
            "final_fit": {"fit_count": len(fit), "early_stop_count": len(early), "epochs_run": len(final.history),
                          "parameter_count": sum(v.size for v in final.parameters.values()),
                          "clipped_batches": sum(v["clipped_batches"] for v in final.history)},
        }
        if args.checkpoint_dir:
            directory = Path(args.checkpoint_dir)
            if not directory.is_absolute(): directory = ROOT / directory
            directory.mkdir(parents=True, exist_ok=True)
            final.save(directory / f"{name}_seed_{args.seed}.npz")
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {"oof_mae": v["training_oof"]["mae"], "validation_mae": v["development_validation"]["mae"]}
                      for name, v in result["models"].items()}, indent=2))


if __name__ == "__main__": main()
