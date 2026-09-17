#!/usr/bin/env python3
"""Compare similarity thresholds and create the formal fixed data split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import (
    assign_clusters_to_splits,
    load_promoter_arrays,
    similarity_cluster_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.80, 0.85, 0.90])
    parser.add_argument("--selected-threshold", type=float, default=0.90)
    parser.add_argument("--output-dir", default="data/splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(
        args.data_dir,
        sequence_file=data_config["sequence_file"],
        label_file=data_config["label_file"],
        sequence_length=int(data_config["sequence_length"]),
    )
    thresholds = sorted(set(args.thresholds))
    if args.selected_threshold not in thresholds:
        raise ValueError("selected-threshold must be included in thresholds")

    comparisons: list[dict[str, int | float]] = []
    selected_labels: np.ndarray | None = None
    for threshold in thresholds:
        labels, summary = similarity_cluster_labels(sequences, threshold)
        comparisons.append(summary)
        if threshold == args.selected_threshold:
            selected_labels = labels
    if selected_labels is None:
        raise RuntimeError("Could not build selected clustering")

    assignment = assign_clusters_to_splits(
        selected_labels,
        float(data_config["train_ratio"]),
        float(data_config["validation_ratio"]),
        float(data_config["test_ratio"]),
    )
    output_dir = REPOSITORY_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "ecoli_similarity_split.csv"
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "split", "cluster_id"])
        writer.writeheader()
        for index, (split, cluster) in enumerate(zip(assignment, selected_labels)):
            writer.writerow({"sample_index": index, "split": split, "cluster_id": int(cluster)})

    train_mask = assignment == "train"
    train_log_strength = np.log10(strengths[train_mask])
    metadata = {
        "source": "promoter.npy and gene_expression.npy",
        "sample_count": len(sequences),
        "sequence_length": int(data_config["sequence_length"]),
        "selected_threshold": args.selected_threshold,
        "threshold_comparison": comparisons,
        "split_counts": {name: int(np.sum(assignment == name)) for name in ("train", "validation", "test")},
        "training_log10_strength": {
            "mean": float(train_log_strength.mean()),
            "std": float(train_log_strength.std()),
            "tertile_boundaries": [float(value) for value in np.quantile(train_log_strength, [1 / 3, 2 / 3])],
        },
    }
    metadata_path = output_dir / "ecoli_similarity_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote split definition to {split_path}")
    print(f"Wrote split metadata to {metadata_path}")


if __name__ == "__main__":
    main()
