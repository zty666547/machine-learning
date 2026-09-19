#!/usr/bin/env python3
"""Train and compare conditional autoregressive and VAE generators uniformly."""

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
from promoter_ml.models import ConditionalSequenceVAE
from promoter_ml.sequence_metrics import diversity_metrics, generation_metrics, motif_metrics
from generate_conditional_position_baseline import assign_strength_groups
from train_motif_cnn_predictor import one_hot_sequences


CONDITIONS = ("weak", "medium", "strong")


def evaluate(generated: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {**generation_metrics(generated, reference), **motif_metrics(generated, reference), **diversity_metrics(generated, reference)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m3_conditional_generator_comparison.json")
    parser.add_argument("--samples-per-condition", type=int, default=500)
    parser.add_argument("--vae-epochs", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    seed = int(config["project"]["seed"])
    logger = configure_logging("compare_conditional_generators", REPOSITORY_ROOT / config["logging"]["directory"])
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    training_sequences = sequences[train_idx]
    training_targets = np.log10(strengths[train_idx])
    labels, boundaries = assign_strength_groups(training_targets)
    group_targets = {condition: float(training_targets[labels == condition].mean()) for condition in CONDITIONS}
    autoregressive = ConditionalAutoregressiveGenerator(order=3).fit(training_sequences, labels)
    vae = ConditionalSequenceVAE(beta=0.1, epochs=args.vae_epochs, seed=seed)
    features = one_hot_sequences(sequences)
    vae.fit(features[train_idx], np.log10(strengths[train_idx]), validation=(features[validation_idx], np.log10(strengths[validation_idx])))
    result = {"condition_definition": "training log10(strength) tertiles", "boundaries": boundaries, "samples_per_condition": args.samples_per_condition, "models": {"conditional_autoregressive": {}, "continuous_condition_vae": {}}}
    for offset, condition in enumerate(CONDITIONS):
        reference = training_sequences[labels == condition]
        autoregressive_generated = autoregressive.sample(condition, args.samples_per_condition, seed + offset)
        vae_generated = vae.sample(np.full(args.samples_per_condition, group_targets[condition]), seed + offset)
        result["models"]["conditional_autoregressive"][condition] = evaluate(autoregressive_generated, reference)
        result["models"]["continuous_condition_vae"][condition] = evaluate(vae_generated, reference)
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Conditional generator comparison: %s", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
