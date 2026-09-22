#!/usr/bin/env python3
"""Verify that the frozen M3 evidence package is internally consistent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays


REQUIRED_REPORTS = (
    "reports/m3_review_2026-09-21.md",
    "reports/m3_vae_width_stability_2026-09-21.md",
    "reports/m3_conditional_generator_comparison_selected.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m3_release_check_2026-09-22.json")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_metric(model_results: dict, metric: str) -> float:
    return float(np.mean([model_results[group][metric] for group in ("weak", "medium", "strong")]))


def main() -> None:
    args = parse_args()
    sequences, _ = load_promoter_arrays(args.data_dir, "promoter.npy", "gene_expression.npy", 50)
    train_idx, validation_idx, test_idx = load_fixed_split_indices(REPOSITORY_ROOT / "data/splits/ecoli_similarity_split.csv", len(sequences))
    split_sets = [set(indices.tolist()) for indices in (train_idx, validation_idx, test_idx)]
    if [len(indices) for indices in (train_idx, validation_idx, test_idx)] != [8318, 1783, 1783]:
        raise AssertionError("Fixed split sizes differ from the frozen M3 split")
    if any(left & right for position, left in enumerate(split_sets) for right in split_sets[position + 1 :]):
        raise AssertionError("Fixed split sets overlap")

    comparison = load_json(REPOSITORY_ROOT / "reports/m3_conditional_generator_comparison_selected.json")
    expected_configuration = {"hidden_size": 128, "latent_size": 16, "beta": 0.20, "epochs": 120}
    if comparison["vae_configuration"] != expected_configuration:
        raise AssertionError("Comparison does not use the selected VAE configuration")
    autoregressive = comparison["models"]["conditional_autoregressive"]
    vae = comparison["models"]["continuous_condition_vae"]
    for name, results in (("conditional_autoregressive", autoregressive), ("continuous_condition_vae", vae)):
        for group in ("weak", "medium", "strong"):
            for metric in ("valid_fraction", "unique_fraction", "novel_fraction"):
                if results[group][metric] != 1.0:
                    raise AssertionError(f"{name} {group} has failed {metric}")
    ar_kmer_js = mean_metric(autoregressive, "kmer_3_js_divergence")
    vae_kmer_js = mean_metric(vae, "kmer_3_js_divergence")
    if not ar_kmer_js < vae_kmer_js:
        raise AssertionError("M3 primary baseline conclusion no longer holds")

    missing_reports = [report for report in REQUIRED_REPORTS if not (REPOSITORY_ROOT / report).is_file()]
    if missing_reports:
        raise AssertionError(f"Missing frozen M3 evidence: {missing_reports}")
    result = {
        "status": "passed",
        "data": {"sample_count": len(sequences), "split_sizes": {"train": len(train_idx), "validation": len(validation_idx), "test": len(test_idx)}},
        "selected_vae_configuration": expected_configuration,
        "model_comparison": {
            "conditional_autoregressive_mean_kmer_3_js": ar_kmer_js,
            "continuous_vae_mean_kmer_3_js": vae_kmer_js,
            "primary_m3_generator": "conditional_autoregressive",
        },
        "checks": ["fixed_split", "selected_vae_configuration", "generation_validity_uniqueness_novelty", "primary_baseline_conclusion", "required_reports"],
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
