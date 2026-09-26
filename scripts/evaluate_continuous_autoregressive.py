#!/usr/bin/env python3
"""Evaluate a continuous-condition autoregressive promoter generator on M3 metrics."""

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
from promoter_ml.sequence_metrics import diversity_metrics, generation_metrics, motif_metrics
from generate_conditional_position_baseline import assign_strength_groups


TARGET_GRID = np.array([1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9])


def gc_mean(sequences: np.ndarray) -> float:
    return float(np.mean([(sequence.count("G") + sequence.count("C")) / len(sequence) for sequence in sequences]))


def evaluate(generated: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {**generation_metrics(generated, reference), **motif_metrics(generated, reference), **diversity_metrics(generated, reference)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m3_continuous_autoregressive.json")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--condition-bins", type=int, default=5)
    parser.add_argument("--samples-per-condition", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    seed = int(config["project"]["seed"])
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, _, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    training_sequences = sequences[train_idx]
    training_targets = np.log10(strengths[train_idx])
    labels, boundaries = assign_strength_groups(training_targets)
    generator = ContinuousConditionalAutoregressiveGenerator(order=args.order, condition_bins=args.condition_bins).fit(training_sequences, training_targets)
    group_targets = {group: float(training_targets[labels == group].mean()) for group in ("weak", "medium", "strong")}
    group_results = {}
    for offset, group in enumerate(("weak", "medium", "strong")):
        generated = generator.sample(group_targets[group], args.samples_per_condition, seed + offset)
        group_results[group] = evaluate(generated, training_sequences[labels == group])
    response = []
    for offset, target in enumerate(TARGET_GRID):
        generated = generator.sample(float(target), args.samples_per_condition, seed + 20 + offset)
        response.append({"target_log_strength": float(target), "generated_gc_mean": gc_mean(generated)})
    gc_values = np.array([row["generated_gc_mean"] for row in response])
    result = {
        "generator": "continuous_condition_autoregressive_markov_interpolation",
        "configuration": {"order": args.order, "condition_bins": args.condition_bins, "bin_centres": generator.bin_centres.tolist()},
        "condition_definition": "continuous log10(strength), equal-frequency training bins with interpolation",
        "training_tertile_boundaries": boundaries,
        "group_target_values": group_targets,
        "samples_per_condition": args.samples_per_condition,
        "per_tertile_group": group_results,
        "continuous_response": {"targets": response, "target_vs_generated_gc_pearson": float(np.corrcoef(TARGET_GRID, gc_values)[0, 1]), "gc_response_slope_per_log_strength": float(np.polyfit(TARGET_GRID, gc_values, 1)[0])},
        "interpretation_boundary": "Interpolation creates continuous sampling conditions, but GC response is distributional evidence and not experimental expression validation.",
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
