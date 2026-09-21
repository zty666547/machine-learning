#!/usr/bin/env python3
"""Check whether continuous VAE conditions produce a smooth distribution response."""

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
from promoter_ml.sequence_metrics import generation_metrics, sequence_distribution_distance
from train_motif_cnn_predictor import one_hot_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m3_vae_continuous_response.json")
    parser.add_argument("--targets", default="1.7,1.9,2.1,2.3,2.5,2.7,2.9")
    parser.add_argument("--samples-per-target", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, help="Override the shared project seed for response replication.")
    parser.add_argument("--beta", type=float, default=0.20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=16)
    return parser.parse_args()


def gc_mean(sequences: np.ndarray) -> float:
    return float(np.mean([(sequence.count("G") + sequence.count("C")) / len(sequence) for sequence in sequences]))


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    logger = configure_logging("analyze_vae_condition_response", REPOSITORY_ROOT / config["logging"]["directory"])
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    features = one_hot_sequences(sequences)
    targets = np.log10(strengths)
    model = ConditionalSequenceVAE(hidden_size=args.hidden_size, latent_size=args.latent_size, beta=args.beta, epochs=args.epochs, seed=seed)
    model.fit(features[train_idx], targets[train_idx], validation=(features[validation_idx], targets[validation_idx]))
    requested = np.array([float(value) for value in args.targets.split(",")])
    generated_by_target = []
    records = []
    for offset, target in enumerate(requested):
        generated = model.sample(np.full(args.samples_per_target, target), seed + offset)
        generated_by_target.append(generated)
        records.append({"target_log_strength": float(target), "generated_gc_mean": gc_mean(generated), "within_target": generation_metrics(generated, sequences[train_idx])})
    for index in range(1, len(records)):
        records[index]["change_from_previous_target"] = sequence_distribution_distance(generated_by_target[index - 1], generated_by_target[index])
    response = np.array([record["generated_gc_mean"] for record in records])
    result = {
        "model": {"hidden_size": model.hidden_size, "beta": args.beta, "latent_size": model.latent_size, "epochs_completed": len(model.history)},
        "samples_per_target": args.samples_per_target,
        "gc_response_slope_per_log_strength": float(np.polyfit(requested, response, deg=1)[0]),
        "target_vs_generated_gc_pearson": float(np.corrcoef(requested, response)[0, 1]),
        "targets": records,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("VAE continuous condition response: %s", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
