#!/usr/bin/env python3
"""Select position-trimer ridge using training-only group CV; report development validation."""
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
from promoter_ml.models.position_trimer_ridge import PositionTrimerRidge, encode_position_trimers
from evaluate_strength_calibration import cluster_ids, group_folds
from analyze_frozen_motif_cnn import split_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m4_position_trimer_2026-10-28.json")
    parser.add_argument("--model-output", default="outputs/m4_position_trimer/model.npz")
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    config = load_config(ROOT / "configs/m2_motif_cnn_selected.toml")
    data = config["data"]
    seqs, strength = load_promoter_arrays(args.data_dir, data["sequence_file"], data["label_file"], 50)
    train, validation, _ = load_fixed_split_indices(ROOT / data["split_file"], len(seqs))
    groups = cluster_ids(ROOT / data["split_file"], len(seqs))
    if set(groups[train]) & set(groups[validation]):
        raise ValueError("A cluster crosses train and validation")
    tr_codes = encode_position_trimers(seqs[train])
    val_codes = encode_position_trimers(seqs[validation])
    tr_y, val_y = np.log10(strength[train]), np.log10(strength[validation])
    boundaries = np.quantile(tr_y, [1 / 3, 2 / 3])
    fold = group_folds(groups[train], 3, args.seed)
    candidates = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
    constant_oof = np.empty(len(train))
    for k in range(3):
        constant_oof[fold == k] = tr_y[fold != k].mean()
    experiments = []
    for alpha in candidates:
        predictions = np.empty(len(train))
        convergence = []
        for k in range(3):
            fit, hold = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            if set(groups[train][fit]) & set(groups[train][hold]):
                raise ValueError("A cluster crosses an internal fold")
            model = PositionTrimerRidge(alpha=alpha).fit(tr_codes[fit], tr_y[fit])
            predictions[hold] = model.predict(tr_codes[hold])
            convergence.append({"iterations": model.iterations, "relative_residual": model.relative_residual})
        diagnostic = split_diagnostic(tr_y, predictions, boundaries)
        experiments.append({"alpha": alpha, "oof": diagnostic, "convergence": convergence})
        print(f"alpha={alpha}: OOF MAE={diagnostic['mae']:.6f}", flush=True)
    best = min(experiments, key=lambda row: (row["oof"]["mae"], row["alpha"]))
    model = PositionTrimerRidge(alpha=best["alpha"]).fit(tr_codes, tr_y)
    validation_metrics = split_diagnostic(val_y, model.predict(val_codes), boundaries)
    baseline_file = ROOT / "reports/m4_strength_calibration_seed_20261028.json"
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))["validation_baseline"]
    result = {
        "protocol": "Choose alpha by similarity-cluster 3-fold CV on training MAE only; "
                    "evaluate chosen model on the previously reused development validation set. "
                    "No test metrics calculated. M3 blend baseline uses the same fixed validation "
                    "but its CNN used validation early stopping and its blend weights used validation selection.",
        "fold_seed": args.seed, "train_count": len(train), "validation_count": len(validation),
        "feature": "48 start positions x 64 3-mers, one-hot, not standardized; unpenalized intercept",
        "candidate_alphas": list(candidates), "selected_alpha": best["alpha"],
        "constant_mean_oof": {
            "mae": float(np.mean(np.abs(constant_oof - tr_y))),
            "rmse": float(np.sqrt(np.mean((constant_oof - tr_y) ** 2))),
        },
        "constant_mean_validation": {
            "mae": float(np.mean(np.abs(tr_y.mean() - val_y))),
            "rmse": float(np.sqrt(np.mean((tr_y.mean() - val_y) ** 2))),
        },
        "oof_candidates": experiments, "validation_position_trimer": validation_metrics,
        "validation_m3_blend": baseline,
        "final_fit": {"iterations": model.iterations, "relative_residual": model.relative_residual},
        "boundaries": boundaries.tolist(),
    }
    output, model_path = Path(args.output), Path(args.model_output)
    if not output.is_absolute(): output = ROOT / output
    if not model_path.is_absolute(): model_path = ROOT / model_path
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    model.save(model_path)
    print(json.dumps({"selected_alpha": best["alpha"], "validation": validation_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
