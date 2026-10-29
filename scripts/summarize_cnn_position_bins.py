#!/usr/bin/env python3
"""Select CNN bin count by mean training OOF MAE across distinct seeds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    if len(runs) < 2 or len({r["model_seed"] for r in runs}) != len(runs):
        raise ValueError("Require at least two distinct model seeds")
    if len({r["fold_seed"] for r in runs}) != 1 or any(r["bins"] != runs[0]["bins"] for r in runs):
        raise ValueError("Runs must use identical splits and bin configurations")
    if any(r["model_config"] != runs[0]["model_config"] or
           r["train_count"] != runs[0]["train_count"] or
           r["validation_count"] != runs[0]["validation_count"] for r in runs):
        raise ValueError("Runs must use the same data sizes and model configuration")
    summary = {}
    for bins in runs[0]["bins"]:
        values = [r["variants"][str(bins)] for r in runs]
        item = {"parameter_count": values[0]["final_fit"]["parameter_count"]}
        for partition in ("training_oof", "validation"):
            item[partition] = {}
            for metric in ("mae", "rmse", "pearson", "spearman", "prediction_to_target_std_ratio"):
                v = np.array([value[partition][metric] for value in values])
                item[partition][metric] = {"mean": float(v.mean()), "std_population": float(v.std())}
            item[partition]["group_mae"] = {}
            for group in ("weak", "medium", "strong"):
                v = np.array([value[partition]["by_true_strength_group"][group]["mae"] for value in values])
                item[partition]["group_mae"][group] = {"mean": float(v.mean()), "std_population": float(v.std())}
        summary[str(bins)] = item
    selected = min(runs[0]["bins"], key=lambda b: (summary[str(b)]["training_oof"]["mae"]["mean"], b))
    result = {"model_seeds": [r["model_seed"] for r in runs], "fold_seed": runs[0]["fold_seed"],
              "selection_rule": "Lowest mean training OOF MAE across seeds; validation not used for selection.",
              "selected_bins": selected, "variants": summary}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
