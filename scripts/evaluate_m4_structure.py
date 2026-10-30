#!/usr/bin/env python3
"""Evaluate position-specific sequence distributions and exploratory motif matches."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays
from promoter_ml.generation import ContinuousConditionalAutoregressiveGenerator
from promoter_ml.models import ConditionalSequenceVAE
from promoter_ml.structure_analysis import (
    encode_sequences, position_kmer_frequency, position_js, top_kmers, motif_scan, shuffle_within_sequence
)
from generate_conditional_position_baseline import assign_strength_groups
from train_motif_cnn_predictor import one_hot_sequences

GROUPS = ("weak", "medium", "strong")


def statistics(reference: np.ndarray, sample: np.ndarray) -> dict:
    result = {
        "count": len(sample),
        "gc_mean": float(np.mean((sample == 1) | (sample == 2))),
        "position_base_l1_mean": float(np.abs(position_kmer_frequency(sample, 1) -
                                              position_kmer_frequency(reference, 1)).sum(axis=1).mean()),
    }
    for k in (3, 5):
        ref_freq = position_kmer_frequency(reference, k)
        sample_freq = position_kmer_frequency(sample, k)
        result[f"position_{k}mer_js_mean"] = float(position_js(ref_freq, sample_freq).mean())
    result["motif_match"] = {
        "center_radius1": motif_scan(sample, shift=0, radius=1),
        "shift_minus5_radius1": motif_scan(sample, shift=-5, radius=1),
        "shift_plus5_radius1": motif_scan(sample, shift=5, radius=1),
        "center_radius5": motif_scan(sample, shift=0, radius=5),
    }
    return result


def motif_profiles(reference: np.ndarray, generated: dict[str, np.ndarray], count: int = 5) -> dict:
    profiles = {}
    for k in (3, 5):
        ref_freq = position_kmer_frequency(reference, k)
        generated_freq = {key: position_kmer_frequency(value, k) for key, value in generated.items()}
        profiles[str(k)] = {
            token: {
                "reference_start_frequency": ref_freq[:, code].tolist(),
                **{f"{name}_start_frequency": frequencies[:, code].tolist()
                   for name, frequencies in generated_freq.items()},
            }
            for code, token in top_kmers(ref_freq, k, count)
        }
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--vae-seed", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--samples-per-group", type=int, default=500)
    parser.add_argument("--vae-epochs", type=int, default=120)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples_per_group < 2:
        raise ValueError("At least two samples per group are required")
    cfg = load_config(ROOT / "configs/base.toml")
    data = cfg["data"]
    sequences, strengths = load_promoter_arrays(args.data_dir, data["sequence_file"], data["label_file"], 50)
    train_idx, validation_idx, _ = load_fixed_split_indices(ROOT / data["split_file"], len(sequences))
    selected = np.concatenate([train_idx, validation_idx])
    n_train = len(train_idx)
    features = one_hot_sequences(sequences[selected])
    targets = np.log10(strengths[selected])
    train_sequences = sequences[train_idx]
    train_targets = targets[:n_train]
    labels, boundaries = assign_strength_groups(train_targets)
    auto = ContinuousConditionalAutoregressiveGenerator(order=3, condition_bins=3).fit(train_sequences, train_targets)
    vae = ConditionalSequenceVAE(hidden_size=128, latent_size=16, beta=0.2, epochs=args.vae_epochs, seed=args.vae_seed)
    print(f"Training VAE seed={args.vae_seed}", flush=True)
    vae.fit(features[:n_train], train_targets, validation=(features[n_train:], targets[n_train:]))
    groups = {}
    for offset, group in enumerate(GROUPS):
        print(f"Sampling group={group}", flush=True)
        ref_sequences = train_sequences[labels == group]
        target = float(train_targets[labels == group].mean())
        if len(ref_sequences) <= args.samples_per_group:
            raise ValueError("Reference group must be larger than the sample count")
        random = np.random.default_rng(args.sample_seed + 1000 + offset)
        permutation = random.permutation(len(ref_sequences))
        real_sample = encode_sequences(ref_sequences[permutation[:args.samples_per_group]])
        real_complement = encode_sequences(ref_sequences[permutation[args.samples_per_group:]])
        ref = encode_sequences(ref_sequences)
        generated_auto = encode_sequences(auto.sample(target, args.samples_per_group, args.sample_seed + offset))
        generated_vae = encode_sequences(vae.sample(np.full(args.samples_per_group, target), args.sample_seed + offset))
        shuffled = shuffle_within_sequence(real_sample, args.sample_seed + 2000 + offset)
        groups[group] = {
            "target_log_strength": target, "reference_count": len(ref),
            "reference": {"gc_mean": float(np.mean((ref == 1) | (ref == 2))),
                          "motif_match": {
                              "center_radius1": motif_scan(ref, shift=0, radius=1),
                              "shift_minus5_radius1": motif_scan(ref, shift=-5, radius=1),
                              "shift_plus5_radius1": motif_scan(ref, shift=5, radius=1),
                              "center_radius5": motif_scan(ref, shift=0, radius=5),
                          }},
            "real_subsample_vs_remainder": statistics(real_complement, real_sample),
            "shuffled_real_subsample_vs_remainder": statistics(real_complement, shuffled),
            "autoregressive_vs_full_training_group": statistics(ref, generated_auto),
            "vae_vs_full_training_group": statistics(ref, generated_vae),
            "position_base_profiles": {
                "reference": position_kmer_frequency(ref, 1).tolist(),
                "autoregressive": position_kmer_frequency(generated_auto, 1).tolist(),
                "vae": position_kmer_frequency(generated_vae, 1).tolist(),
                "alphabet_order": "ACGT",
            },
            "top_training_motif_profiles": motif_profiles(ref, {"autoregressive": generated_auto, "vae": generated_vae}),
        }
    result = {
        "protocol": "Each group: training target mean; 500 samples per generator; fixed training group reference. "
                    "VAE model uses development validation for early stopping. The real subsample is compared "
                    "against the remainder of the training group. In-sequence shuffle preserves base composition only.",
        "interpretation_boundary": "Training-distribution fit is not independent validation. Consensus matches are exploratory "
                                   "sequence patterns under a hypothesized TSS alignment, not experimentally confirmed motifs or activity.",
        "configuration": {"vae_seed": args.vae_seed, "sample_seed": args.sample_seed,
                          "vae_epochs_limit": args.vae_epochs, "vae_epochs_run": len(vae.history),
                          "vae_hidden_size": 128, "vae_latent_size": 16, "vae_beta": 0.2,
                          "ar_order": 3, "ar_condition_bins": 3, "samples_per_group": args.samples_per_group},
        "dataset_sha256": {
            "sequence": hashlib.sha256((Path(args.data_dir) / data["sequence_file"]).read_bytes()).hexdigest(),
            "strength": hashlib.sha256((Path(args.data_dir) / data["label_file"]).read_bytes()).hexdigest(),
            "split": hashlib.sha256((ROOT / data["split_file"]).read_bytes()).hexdigest(),
        },
        "numpy_version": np.__version__,
        "training_tertile_boundaries": boundaries,
        "assumed_alignment": "50 bp ordered 5-prime to 3-prime, positions -50 through -1; TSS +1",
        "scan_rule": "TTGACA start -35; TATAAT start -12; allow <=1 mismatch, scan radius +/-1 bp. "
                     "Sensitivity uses both starts shifted -5/+5 bp and radius +/-5 bp. Paired match requires any hit in each window; "
                     "it does not enforce independent spacer or validate a binding site.",
        "groups": groups,
    }
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
