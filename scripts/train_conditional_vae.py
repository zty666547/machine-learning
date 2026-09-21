#!/usr/bin/env python3
"""Train a continuous-condition VAE and evaluate samples against train groups."""

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
from promoter_ml.logging_utils import configure_logging
from promoter_ml.models import ConditionalSequenceVAE
from promoter_ml.sequence_metrics import generation_metrics
from generate_conditional_position_baseline import assign_strength_groups
from train_motif_cnn_predictor import one_hot_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/conditional_vae")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--samples-per-condition", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    seed = int(config["project"]["seed"])
    logger = configure_logging("train_conditional_vae", REPOSITORY_ROOT / config["logging"]["directory"])
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    features = one_hot_sequences(sequences)
    targets = np.log10(strengths)
    model = ConditionalSequenceVAE(hidden_size=args.hidden_size, latent_size=args.latent_size, beta=args.beta, epochs=args.epochs, seed=seed)
    model.fit(features[train_idx], targets[train_idx], validation=(features[validation_idx], targets[validation_idx]))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.npz")
    labels, boundaries = assign_strength_groups(targets[train_idx])
    group_targets = {str(group): float(targets[train_idx][labels == group].mean()) for group in ("weak", "medium", "strong")}
    per_condition = {}
    for offset, condition in enumerate(("weak", "medium", "strong")):
        generated = model.sample(np.full(args.samples_per_condition, group_targets[condition]), seed + offset)
        reference = sequences[train_idx][labels == condition]
        (output_dir / f"generated_{condition}.txt").write_text("\n".join(generated.tolist()) + "\n", encoding="utf-8")
        per_condition[condition] = {"target_log_strength": group_targets[condition], "metrics_against_same_condition_training_data": generation_metrics(generated, reference)}
    metrics = {"generator": "continuous_condition_vae", "hidden_size": model.hidden_size, "latent_size": model.latent_size, "beta": model.beta, "training_log_strength_boundaries": boundaries, "per_condition": per_condition, "epochs_completed": len(model.history), "best_validation_loss": min(row.get("validation_loss", float("inf")) for row in model.history)}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Conditional VAE metrics: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
