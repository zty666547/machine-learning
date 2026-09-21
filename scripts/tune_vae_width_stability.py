#!/usr/bin/env python3
"""Compare VAE hidden widths across seeds with the selected latent setting."""

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
from train_motif_cnn_predictor import one_hot_sequences


TARGETS = np.array([1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9])


def gc_mean(sequences: np.ndarray) -> float:
    return float(np.mean([(sequence.count("G") + sequence.count("C")) / len(sequence) for sequence in sequences]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m3_vae_width_stability.json")
    parser.add_argument("--hidden-sizes", default="96,128")
    parser.add_argument("--seeds", default="20260912,20260913,20260914")
    parser.add_argument("--beta", type=float, default=0.20)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples-per-target", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    logger = configure_logging("tune_vae_width_stability", REPOSITORY_ROOT / config["logging"]["directory"])
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    features = one_hot_sequences(sequences)
    targets = np.log10(strengths)
    hidden_sizes = [int(value) for value in args.hidden_sizes.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    trials = []
    for hidden_size in hidden_sizes:
        for seed in seeds:
            model = ConditionalSequenceVAE(hidden_size=hidden_size, latent_size=args.latent_size, beta=args.beta, epochs=args.epochs, seed=seed)
            model.fit(features[train_idx], targets[train_idx], validation=(features[validation_idx], targets[validation_idx]))
            gc_values = []
            kmer_values = []
            for offset, target in enumerate(TARGETS):
                generated = model.sample(np.full(args.samples_per_target, target), seed + offset)
                gc_values.append(gc_mean(generated))
                kmer_values.append(generation_metrics(generated, sequences[train_idx])["kmer_3_js_divergence"])
            record = {
                "hidden_size": hidden_size,
                "seed": seed,
                "validation_loss": min(row.get("validation_loss", float("inf")) for row in model.history),
                "target_gc_pearson": float(np.corrcoef(TARGETS, gc_values)[0, 1]),
                "gc_response_slope": float(np.polyfit(TARGETS, gc_values, deg=1)[0]),
                "mean_kmer_3_js": float(np.mean(kmer_values)),
                "epochs_completed": len(model.history),
            }
            trials.append(record)
            logger.info("VAE width stability trial: %s", record)
    summary = []
    for hidden_size in hidden_sizes:
        matching = [record for record in trials if record["hidden_size"] == hidden_size]
        correlations = [record["target_gc_pearson"] for record in matching]
        summary.append({
            "hidden_size": hidden_size,
            "mean_validation_loss": float(np.mean([record["validation_loss"] for record in matching])),
            "mean_target_gc_pearson": float(np.mean(correlations)),
            "std_target_gc_pearson": float(np.std(correlations)),
            "mean_kmer_3_js": float(np.mean([record["mean_kmer_3_js"] for record in matching])),
        })
    summary.sort(key=lambda record: (record["mean_kmer_3_js"], -abs(record["mean_target_gc_pearson"])))
    result = {
        "beta": args.beta,
        "latent_size": args.latent_size,
        "epochs": args.epochs,
        "samples_per_target": args.samples_per_target,
        "trials": trials,
        "summary": summary,
        "selection_rule": "lowest mean 3-mer JS, then strongest mean absolute GC response correlation",
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "best": summary[0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
