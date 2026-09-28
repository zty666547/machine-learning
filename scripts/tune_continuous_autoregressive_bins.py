#!/usr/bin/env python3
"""Choose continuous autoregressive condition-bin count using validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays
from promoter_ml.generation import ContinuousConditionalAutoregressiveGenerator
from promoter_ml.sequence_metrics import generation_metrics, motif_metrics
from generate_conditional_position_baseline import CONDITION_NAMES, assign_strength_groups


TARGET_GRID = np.array([1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9])


def gc_mean(sequences: np.ndarray) -> float:
    return float(np.mean([(sequence.count("G") + sequence.count("C")) / len(sequence) for sequence in sequences]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m3_continuous_autoregressive_bin_tuning.json")
    parser.add_argument("--condition-bins", default="3,5,7")
    parser.add_argument("--seeds", default="20260912,20260913,20260914")
    parser.add_argument("--samples-per-condition", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    train_sequences, validation_sequences = sequences[train_idx], sequences[validation_idx]
    train_targets, validation_targets = np.log10(strengths[train_idx]), np.log10(strengths[validation_idx])
    train_labels, boundaries = assign_strength_groups(train_targets)
    validation_labels = CONDITION_NAMES[np.digitize(validation_targets, boundaries, right=False)]
    group_targets = {group: float(train_targets[train_labels == group].mean()) for group in CONDITION_NAMES}
    bin_values = [int(value) for value in args.condition_bins.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    trials = []
    for bin_count in bin_values:
        generator = ContinuousConditionalAutoregressiveGenerator(order=3, condition_bins=bin_count).fit(train_sequences, train_targets)
        for seed in seeds:
            per_group = []
            for offset, group in enumerate(CONDITION_NAMES):
                generated = generator.sample(group_targets[str(group)], args.samples_per_condition, seed + offset)
                reference = validation_sequences[validation_labels == group]
                metrics = {**generation_metrics(generated, reference), **motif_metrics(generated, reference)}
                per_group.append(metrics)
            response_gc = []
            for offset, target in enumerate(TARGET_GRID):
                generated = generator.sample(float(target), args.samples_per_condition, seed + 20 + offset)
                response_gc.append(gc_mean(generated))
            trials.append({
                "condition_bins": bin_count,
                "seed": seed,
                "mean_validation_kmer_3_js": float(np.mean([row["kmer_3_js_divergence"] for row in per_group])),
                "mean_validation_top_5mer_error": float(np.mean([row["top_reference_motif_mean_abs_frequency_error"] for row in per_group])),
                "mean_validation_top_5mer_overlap": float(np.mean([row["top_motif_overlap_fraction"] for row in per_group])),
                "target_gc_pearson": float(np.corrcoef(TARGET_GRID, response_gc)[0, 1]),
                "gc_response_slope": float(np.polyfit(TARGET_GRID, response_gc, 1)[0]),
            })
    summary = []
    for bin_count in bin_values:
        matching = [trial for trial in trials if trial["condition_bins"] == bin_count]
        summary.append({
            "condition_bins": bin_count,
            "mean_validation_kmer_3_js": float(np.mean([trial["mean_validation_kmer_3_js"] for trial in matching])),
            "mean_validation_top_5mer_error": float(np.mean([trial["mean_validation_top_5mer_error"] for trial in matching])),
            "mean_validation_top_5mer_overlap": float(np.mean([trial["mean_validation_top_5mer_overlap"] for trial in matching])),
            "mean_target_gc_pearson": float(np.mean([trial["target_gc_pearson"] for trial in matching])),
            "std_target_gc_pearson": float(np.std([trial["target_gc_pearson"] for trial in matching])),
        })
    summary.sort(key=lambda row: (row["mean_validation_kmer_3_js"], row["mean_validation_top_5mer_error"], -abs(row["mean_target_gc_pearson"])))
    result = {
        "selection_data": "validation set only; generator fitting uses training set only",
        "order": 3,
        "training_tertile_boundaries": boundaries,
        "group_target_values": group_targets,
        "samples_per_condition": args.samples_per_condition,
        "trials": trials,
        "summary": summary,
        "selection_rule": "lowest mean validation 3-mer JS, then lowest validation 5-mer error, then strongest continuous GC response",
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "best": summary[0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
