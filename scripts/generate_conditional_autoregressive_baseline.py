#!/usr/bin/env python3
"""Train and evaluate a conditional 3rd-order autoregressive promoter generator."""

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
from promoter_ml.generation import ConditionalAutoregressiveGenerator
from promoter_ml.logging_utils import configure_logging
from promoter_ml.sequence_metrics import generation_metrics


CONDITION_NAMES = np.array(["weak", "medium", "strong"])


def assign_strength_groups(log_strengths: np.ndarray) -> tuple[np.ndarray, list[float]]:
    boundaries = np.quantile(log_strengths, [1 / 3, 2 / 3])
    return CONDITION_NAMES[np.digitize(log_strengths, boundaries, right=False)], [float(value) for value in boundaries]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/conditional_autoregressive_baseline")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--samples-per-condition", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    seed = int(config["project"]["seed"])
    logger = configure_logging("generate_conditional_autoregressive_baseline", REPOSITORY_ROOT / config["logging"]["directory"])
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, _, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    training_sequences = sequences[train_idx]
    training_targets = np.log10(strengths[train_idx])
    labels, boundaries = assign_strength_groups(training_targets)
    generator = ConditionalAutoregressiveGenerator(order=args.order).fit(training_sequences, labels)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_condition = {}
    for offset, condition in enumerate(CONDITION_NAMES):
        generated = generator.sample(str(condition), args.samples_per_condition, seed + offset)
        reference = training_sequences[labels == condition]
        (output_dir / f"generated_{condition}.txt").write_text("\n".join(generated.tolist()) + "\n", encoding="utf-8")
        per_condition[str(condition)] = {
            "training_samples": int(len(reference)),
            "metrics_against_same_condition_training_data": generation_metrics(generated, reference),
        }
    metrics = {
        "generator": "conditional_position_aware_autoregressive_markov",
        "condition_definition": "training log10(strength) tertiles",
        "training_log_strength_boundaries": boundaries,
        "order": args.order,
        "samples_per_condition": args.samples_per_condition,
        "per_condition": per_condition,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Conditional autoregressive generation metrics: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
