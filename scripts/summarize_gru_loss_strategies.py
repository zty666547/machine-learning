#!/usr/bin/env python3
"""Summarize fixed-fold, fixed-seed gated and robust-loss strategies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mae", "rmse", "pearson", "spearman", "prediction_to_target_std_ratio")
GROUPS = ("weak", "medium", "strong")


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.inputs]
    if sorted(r["seed"] for r in runs) != [20260912, 20260913, 20260914]:
        raise ValueError("Expected all frozen seeds exactly once")
    for run in runs:
        for key in ("fold_seed", "train_count", "validation_count", "data_sha256", "protocol_sha256",
                    "prior_protocol_sha256", "training_boundaries", "numpy_version"):
            if run[key] != runs[0][key]: raise ValueError(f"Mismatch in {key}")
    sources = {"cnn": ("prior_models", "cnn"), "rnn_mse": ("prior_models", "rnn"),
               "rnn_huber": ("models", "rnn_huber"), "gru_mse": ("models", "gru")}
    result = {"protocol": "Three model seeds, one shared set of cluster folds. Seeds are not independent data splits. "
                          "Training out-of-fold is the main development comparison; development validation has been reused. "
                          "GRU architecture comparison is not an isolated causal ablation. Huber comparison holds the RNN architecture and settings fixed.",
              "seeds": [r["seed"] for r in runs], "fold_seed": runs[0]["fold_seed"],
              "data_sha256": runs[0]["data_sha256"], "protocol_sha256": runs[0]["protocol_sha256"],
              "models": {}}
    for name, (part, key) in sources.items():
        info = {"parameter_count": runs[0][part][key]["final_fit"]["parameter_count"]}
        for context in ("training_oof", "development_validation"):
            source = [r[part][key][context] for r in runs]
            info[context] = {metric: mean_std([s[metric] for s in source]) for metric in METRICS}
            info[context]["groups"] = {group: {metric: mean_std([s["by_true_strength_group"][group][metric] for s in source])
                                      for metric in ("mae", "mean_signed_error")} for group in GROUPS}
        info["final_fit_epochs"] = [r[part][key]["final_fit"]["epochs_run"] for r in runs]
        result["models"][name] = info
    result["paired_differences"] = {}
    for name, (newpart, newkey, oldpart, oldkey) in {
        "huber_minus_mse_rnn": ("models", "rnn_huber", "prior_models", "rnn"),
        "gru_minus_mse_rnn": ("models", "gru", "prior_models", "rnn"),
        "gru_minus_cnn": ("models", "gru", "prior_models", "cnn"),
    }.items():
        result["paired_differences"][name] = {
            context: {metric: mean_std([r[newpart][newkey][context][metric] - r[oldpart][oldkey][context][metric] for r in runs])
                      for metric in METRICS}
            for context in ("training_oof", "development_validation")}
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
