#!/usr/bin/env python3
"""Summarize fixed-seed, same-fold CNN/RNN comparisons."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mae", "rmse", "pearson", "spearman", "prediction_to_target_std_ratio")
GROUPS = ("weak", "medium", "strong")


def mean_std(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.inputs]
    if sorted(r["seed"] for r in runs) != [20260912, 20260913, 20260914]:
        raise ValueError("Expected the three fixed seeds exactly once")
    for run in runs:
        for key in ("fold_seed", "train_count", "validation_count", "data_sha256", "protocol_sha256", "training_boundaries", "numpy_version"):
            if run[key] != runs[0][key]: raise ValueError(f"Mismatch in {key}")
        if not run["previous_cnn_mae_reproduced"]: raise ValueError("Prior CNN baseline not reproduced")
    result = {"protocol": "Three model seeds, one shared set of cluster folds; standard deviation is over seeds, not independent data splits. "
                          "Training out-of-fold and development are exploratory; previously viewed test data is not used. "
                          "Different architectures and training details mean this is not an isolated architecture-only causal comparison.",
              "seeds": [r["seed"] for r in runs], "fold_seed": runs[0]["fold_seed"],
              "data_sha256": runs[0]["data_sha256"], "protocol_sha256": runs[0]["protocol_sha256"],
              "models": {}}
    for model in ("cnn", "rnn"):
        info = {"parameter_count": runs[0]["models"][model]["final_fit"]["parameter_count"]}
        for context in ("training_oof", "development_validation"):
            source = [r["models"][model][context] for r in runs]
            info[context] = {metric: mean_std([s[metric] for s in source]) for metric in METRICS}
            info[context]["groups"] = {
                group: {metric: mean_std([s["by_true_strength_group"][group][metric] for s in source])
                        for metric in ("mae", "mean_signed_error")}
                for group in GROUPS}
        info["final_fit_epochs"] = [r["models"][model]["final_fit"]["epochs_run"] for r in runs]
        info["final_fit_clipped_batches"] = [r["models"][model]["final_fit"]["clipped_batches"] for r in runs]
        result["models"][model] = info
    result["paired_rnn_minus_cnn"] = {
        context: {metric: mean_std([r["models"]["rnn"][context][metric] - r["models"]["cnn"][context][metric] for r in runs])
                  for metric in METRICS}
        for context in ("training_oof", "development_validation")}
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
