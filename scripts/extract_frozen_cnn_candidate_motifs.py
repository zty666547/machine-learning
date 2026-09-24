#!/usr/bin/env python3
"""Extract data-driven 7-mer patterns from the frozen motif CNN's training responses."""

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
from promoter_ml.models import MotifCNNStrengthPredictor
from promoter_ml.reproducibility import set_random_seed
from train_motif_cnn_predictor import one_hot_sequences


BASES = np.array(list("ACGT"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m2_motif_cnn_selected.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m2_frozen_cnn_candidate_motifs.json")
    parser.add_argument("--top-patterns", type=int, default=5)
    parser.add_argument("--top-activations", type=int, default=100)
    return parser.parse_args()


def consensus(sequences: list[str]) -> tuple[str, list[dict[str, float]]]:
    matrix = np.array([[base for base in sequence] for sequence in sequences])
    rows = []
    for position in range(matrix.shape[1]):
        fractions = {base: float(np.mean(matrix[:, position] == base)) for base in BASES}
        rows.append(fractions)
    return "".join(max(row, key=row.get) for row in rows), rows


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    data_config = config["data"]
    model_config = config["motif_cnn"]
    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    sequences, strengths = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, validation_idx, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    features = one_hot_sequences(sequences)
    targets = np.log10(strengths)
    model = MotifCNNStrengthPredictor(seed=seed, **model_config)
    model.fit(features[train_idx], targets[train_idx], validation=(features[validation_idx], targets[validation_idx]))
    _, cache = model._forward(features[train_idx])
    _, convolution, _, _, _, position_groups = cache
    activations = np.maximum(convolution, 0.0)
    head_proxy = (model.parameters["head_weight"] @ model.parameters["output_weight"])[:, 0].reshape(model.location_bins, model.filters)
    ranked = sorted(
        ((abs(float(head_proxy[bin_index, filter_index])), bin_index, filter_index) for bin_index in range(model.location_bins) for filter_index in range(model.filters)),
        reverse=True,
    )[: args.top_patterns]
    patterns = []
    training_sequences = sequences[train_idx]
    for _, bin_index, filter_index in ranked:
        positions = position_groups[bin_index]
        selected = activations[:, positions, filter_index].reshape(-1)
        top_indices = np.argsort(selected)[-args.top_activations :][::-1]
        sequence_indices = top_indices // len(positions)
        position_offsets = positions[top_indices % len(positions)]
        kmers = [training_sequences[sequence_index][position : position + model.kernel_size] for sequence_index, position in zip(sequence_indices, position_offsets)]
        consensus_sequence, position_frequencies = consensus(kmers)
        patterns.append({
            "filter_index": int(filter_index),
            "position_bin": int(bin_index + 1),
            "position_start_range_zero_based": [int(positions.min()), int(positions.max())],
            "output_head_proxy": float(head_proxy[bin_index, filter_index]),
            "mean_top_activation": float(selected[top_indices].mean()),
            "consensus_7mer": consensus_sequence,
            "position_base_frequencies": position_frequencies,
            "top_observed_7mers": kmers[:10],
        })
    result = {
        "status": "data_driven_candidate_motif_extraction",
        "architecture": model_config,
        "data_usage": "Training sequences are used only to inspect activations of the already frozen model; no test-set sequences or new model selection are used.",
        "interpretation_boundary": "These are activation-derived candidate patterns, not experimentally validated transcription-factor binding motifs.",
        "patterns": patterns,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
