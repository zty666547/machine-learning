#!/usr/bin/env python3
"""Matched comparison of position 1+2-mers with and without position 3-mers."""
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
from promoter_ml.models.position_trimer_ridge import PositionTrimerRidge
from evaluate_strength_calibration import cluster_ids, group_folds
from analyze_frozen_motif_cnn import split_diagnostic


def encode_position_kmers(sequences: np.ndarray, sequence_length: int, max_k: int) -> np.ndarray:
    """Each position has a 64-slot one-hot block; unused categories remain zero."""
    if max_k not in (2, 3) or sequence_length < max_k:
        raise ValueError("max_k must be 2 or 3 and fit within the sequence")
    values = np.asarray(sequences)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Expected a nonempty one-dimensional sequence array")
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.empty((len(values), sequence_length), dtype=np.int64)
    for i, value in enumerate(values):
        seq = str(value).upper()
        if len(seq) != sequence_length or any(base not in mapping for base in seq):
            raise ValueError("Sequence must have the specified length and contain A/C/G/T only")
        encoded[i] = [mapping[base] for base in seq]
    slots = []
    for k in range(1, max_k + 1):
        codes = np.zeros((len(values), sequence_length - k + 1), dtype=np.int64)
        for offset in range(k):
            codes = codes * 4 + encoded[:, offset:offset + sequence_length - k + 1]
        slots.append(codes)
    raw = np.concatenate(slots, axis=1)
    return raw + 64 * np.arange(raw.shape[1])[None, :]


def run_variant(codes_train, codes_val, train_y, val_y, groups, fold, boundaries, alphas):
    candidates = []
    for alpha in alphas:
        oof = np.empty(len(train_y), dtype=np.float64)
        convergence = []
        for k in range(3):
            fit, hold = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            if set(groups[fit]) & set(groups[hold]):
                raise ValueError("A cluster crosses an internal fold")
            # Existing sparse categorical-slot solver implements ridge on the
            # padded 64-category blocks; zero features have zero coefficients.
            model = PositionTrimerRidge(alpha=alpha).fit(codes_train[fit], train_y[fit])
            oof[hold] = model.predict(codes_train[hold])
            convergence.append({"iterations": model.iterations, "relative_residual": model.relative_residual})
        result = split_diagnostic(train_y, oof, boundaries)
        candidates.append({"alpha": alpha, "oof": result, "convergence": convergence})
        print(f"slots={codes_train.shape[1]} alpha={alpha}: OOF MAE={result['mae']:.6f}", flush=True)
    best = min(candidates, key=lambda row: (row["oof"]["mae"], row["alpha"]))
    final = PositionTrimerRidge(alpha=best["alpha"]).fit(codes_train, train_y)
    return {"selected_alpha": best["alpha"], "candidates": candidates,
            "validation": split_diagnostic(val_y, final.predict(codes_val), boundaries),
            "final_fit": {"iterations": final.iterations, "relative_residual": final.relative_residual}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--output", default="reports/m4_position_kmer_ablation_2026-10-29.json")
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/m2_motif_cnn_selected.toml")
    data = cfg["data"]
    seqs, strengths = load_promoter_arrays(args.data_dir, data["sequence_file"], data["label_file"], 50)
    train, validation, _ = load_fixed_split_indices(ROOT / data["split_file"], len(seqs))
    cluster = cluster_ids(ROOT / data["split_file"], len(seqs))
    groups = cluster[train]
    if set(groups) & set(cluster[validation]):
        raise ValueError("A cluster crosses train and validation")
    train_y = np.log10(strengths[train])
    val_y = np.log10(strengths[validation])
    boundaries = np.quantile(train_y, [1 / 3, 2 / 3])
    fold = group_folds(groups, 3, args.seed)
    alphas = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
    variants = {}
    for max_k in (2, 3):
        train_codes = encode_position_kmers(seqs[train], 50, max_k)
        val_codes = encode_position_kmers(seqs[validation], 50, max_k)
        variants[f"position_1_to_{max_k}"] = run_variant(
            train_codes, val_codes, train_y, val_y, groups, fold, boundaries, alphas)
    baseline = json.loads((ROOT / "reports/m4_strength_calibration_seed_20261028.json").read_text(encoding="utf-8"))["validation_baseline"]
    result = {
        "protocol": "Same train clusters, folds, unstandardized position one-hot encoding, ridge solver and alpha grid. "
                    "Each variant selects alpha by training-only 3-fold OOF MAE. Previously reused validation is exploratory. "
                    "No test metrics calculated.",
        "fold_seed": args.seed, "train_count": len(train), "validation_count": len(validation),
        "candidate_alphas": alphas, "variants": variants, "m3_blend_validation": baseline,
        "training_boundaries": boundaries.tolist(),
    }
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {"alpha": v["selected_alpha"], "validation": v["validation"]} for k,v in variants.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
