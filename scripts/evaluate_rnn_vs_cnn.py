#!/usr/bin/env python3
"""Compare a vanilla RNN and motif CNN under identical cluster-aware folds."""
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
from promoter_ml.models import MotifCNNStrengthPredictor, RNNStrengthPredictor
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from analyze_frozen_motif_cnn import split_diagnostic
from train_motif_cnn_predictor import one_hot_sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", default=None, help="Optional experimental RNN checkpoint (.npz).")
    args = parser.parse_args()
    protocol_path = ROOT / "configs/m4_rnn_predictor_2026-11-06.toml"
    with protocol_path.open("rb") as f: protocol = tomllib.load(f)
    experiment = protocol["experiment"]
    if args.seed not in experiment["model_seeds"] or experiment["outer_folds"] != 3:
        raise ValueError("Seed or number of folds differs from frozen protocol")
    cfg = load_config(ROOT / "configs/m2_motif_cnn_selected.toml")
    data = cfg["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data["sequence_file"], data["label_file"], 50)
    train_idx, val_idx, _ = load_fixed_split_indices(ROOT / data["split_file"], len(sequences))
    clusters = cluster_ids(ROOT / data["split_file"], len(sequences))
    if set(clusters[train_idx]) & set(clusters[val_idx]): raise ValueError("Cluster crosses train/validation")
    selected = np.concatenate([train_idx, val_idx])
    n = len(train_idx)
    x, y = one_hot_sequences(sequences[selected]), np.log10(strengths[selected])
    layouts, final_layout = fold_layout(clusters[train_idx], experiment["fold_seed"])
    boundaries = np.quantile(y[:n], [1/3, 2/3])
    result = {"protocol": "Same fixed 3 outer cluster folds and train-only inner early-stopping folds for both models. "
                          "Each final model fits the same train-internal split. Development validation is previously reused; "
                          "raw data loader validates all rows but test records are not featurized or scored. "
                          "No model replacement based on this exploratory comparison.",
              "seed": args.seed, "fold_seed": experiment["fold_seed"], "train_count": n, "validation_count": len(val_idx),
              "data_sha256": {
                  "sequence": hashlib.sha256((Path(args.data_dir) / data["sequence_file"]).read_bytes()).hexdigest(),
                  "strength": hashlib.sha256((Path(args.data_dir) / data["label_file"]).read_bytes()).hexdigest(),
                  "split": hashlib.sha256((ROOT / data["split_file"]).read_bytes()).hexdigest()},
              "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "numpy_version": np.__version__, "training_boundaries": boundaries.tolist(), "models": {}}
    for name in ("cnn", "rnn"):
        def make():
            if name == "cnn": return MotifCNNStrengthPredictor(seed=args.seed, **cfg["motif_cnn"])
            return RNNStrengthPredictor(seed=args.seed, **protocol["rnn"])
        oof = np.empty(n)
        folds = []
        for k, (fit, early, holdout) in enumerate(layouts):
            print(f"seed={args.seed} model={name} fold={k+1}/3", flush=True)
            model = make()
            model.fit(x[fit], y[fit], validation=(x[early], y[early]))
            oof[holdout] = model.predict(x[holdout])
            folds.append({"fit_count": len(fit), "early_stop_count": len(early), "holdout_count": len(holdout),
                          "epochs_run": len(model.history),
                          "parameter_count": sum(v.size for v in model.parameters.values()),
                          "clipped_batches": sum(v.get("clipped_batches", 0) for v in model.history)})
        fit, early = final_layout
        final = make()
        print(f"seed={args.seed} model={name} final/development", flush=True)
        final.fit(x[fit], y[fit], validation=(x[early], y[early]))
        result["models"][name] = {
            "training_oof": split_diagnostic(y[:n], oof, boundaries),
            "folds": folds,
            "development_validation": split_diagnostic(y[n:], final.predict(x[n:]), boundaries),
            "final_fit": {"fit_count": len(fit), "early_stop_count": len(early),
                          "epochs_run": len(final.history),
                          "parameter_count": sum(v.size for v in final.parameters.values()),
                          "clipped_batches": sum(v.get("clipped_batches", 0) for v in final.history)},
        }
        if name == "rnn" and args.model_output:
            path = Path(args.model_output)
            if not path.is_absolute(): path = ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            final.save(path)
    # Compare rerun CNN with the previously recorded same-seed/same-fold experiment.
    previous_path = ROOT / f"reports/m4_cnn_position_seed_{args.seed}.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous["fold_seed"] != experiment["fold_seed"] or previous["model_seed"] != args.seed:
        raise ValueError("Prior CNN comparison used different seeds")
    for context, previous_context in (("training_oof", "training_oof"), ("development_validation", "validation")):
        value = result["models"]["cnn"][context]["mae"]
        saved = previous["variants"]["5"][previous_context]["mae"]
        if abs(value - saved) > 1e-8: raise ValueError(f"Previous CNN rerun differs: {context} {value} != {saved}")
    result["previous_cnn_mae_reproduced"] = True
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {"oof_mae": v["training_oof"]["mae"], "validation_mae": v["development_validation"]["mae"]}
                      for name, v in result["models"].items()}, indent=2))


if __name__ == "__main__": main()
