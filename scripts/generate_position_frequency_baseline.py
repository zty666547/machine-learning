#!/usr/bin/env python3
"""Generate and evaluate the M3 position-frequency baseline using training data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays
from promoter_ml.generation import PositionFrequencyGenerator
from promoter_ml.logging_utils import configure_logging
from promoter_ml.sequence_metrics import generation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/position_frequency_baseline")
    parser.add_argument("--sample-count", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    seed = int(config["project"]["seed"])
    logger = configure_logging("generate_position_frequency_baseline", REPOSITORY_ROOT / config["logging"]["directory"])
    data_config = config["data"]
    sequences, _ = load_promoter_arrays(args.data_dir, data_config["sequence_file"], data_config["label_file"], int(data_config["sequence_length"]))
    train_idx, _, _ = load_fixed_split_indices(REPOSITORY_ROOT / data_config["split_file"], len(sequences))
    training_sequences = sequences[train_idx]
    generated = PositionFrequencyGenerator().fit(training_sequences).sample(args.sample_count, seed=seed)
    metrics = {
        "generator": "position_independent_base_frequency",
        "sample_count": args.sample_count,
        "reference": "training_split_only",
        "metrics": generation_metrics(generated, training_sequences),
    }
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generated_sequences.txt").write_text("\n".join(generated.tolist()) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Position-frequency generation metrics: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
