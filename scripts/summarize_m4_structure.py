#!/usr/bin/env python3
"""Summarize repeated structure comparisons without selecting by their results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

GROUPS = ("weak", "medium", "strong")
METHODS = ("real_subsample_vs_remainder", "shuffled_real_subsample_vs_remainder",
           "autoregressive_vs_full_training_group", "vae_vs_full_training_group")
METRICS = ("gc_mean", "position_base_l1_mean", "position_3mer_js_mean", "position_5mer_js_mean")
WINDOWS = ("center_radius1", "shift_minus5_radius1", "shift_plus5_radius1", "center_radius5")
HITS = ("minus35_hit_fraction", "minus10_hit_fraction", "both_hit_fraction")


def moments(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std_population": float(a.std())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    if len(runs) < 2 or len({x["configuration"]["vae_seed"] for x in runs}) != len(runs):
        raise ValueError("Need at least two distinct VAE seeds")
    if len({x["configuration"]["sample_seed"] for x in runs}) != len(runs):
        raise ValueError("Sampling seeds must be distinct")
    keys = ("vae_epochs_limit", "vae_hidden_size", "vae_latent_size", "vae_beta",
            "ar_order", "ar_condition_bins", "samples_per_group")
    for run in runs[1:]:
        if any(run["configuration"][k] != runs[0]["configuration"][k] for k in keys):
            raise ValueError("Generator configurations differ")
        if run["training_tertile_boundaries"] != runs[0]["training_tertile_boundaries"]:
            raise ValueError("Training boundaries differ")
        if run["dataset_sha256"] != runs[0]["dataset_sha256"]:
            raise ValueError("Dataset or split checksums differ")
        for group in GROUPS:
            if (run["groups"][group]["target_log_strength"] != runs[0]["groups"][group]["target_log_strength"] or
                run["groups"][group]["reference"] != runs[0]["groups"][group]["reference"]):
                raise ValueError("Training reference differs")

    groups = {}
    for group in GROUPS:
        items = [r["groups"][group] for r in runs]
        first = items[0]
        methods = {}
        for method in METHODS:
            methods[method] = {metric: moments([x[method][metric] for x in items]) for metric in METRICS}
            methods[method]["motif_match"] = {
                window: {hit: moments([x[method]["motif_match"][window][hit] for x in items]) for hit in HITS}
                for window in WINDOWS
            }
        profiles = {}
        base_profile = {
            "alphabet_order": first["position_base_profiles"]["alphabet_order"],
            "reference": first["position_base_profiles"]["reference"],
            "autoregressive_mean": np.mean([x["position_base_profiles"]["autoregressive"] for x in items], axis=0).tolist(),
            "vae_mean": np.mean([x["position_base_profiles"]["vae"] for x in items], axis=0).tolist(),
        }
        if any(x["position_base_profiles"]["reference"] != base_profile["reference"] for x in items):
            raise ValueError("Training base profiles differ")
        for k in ("3", "5"):
            profiles[k] = {}
            for token, first_profile in first["top_training_motif_profiles"][k].items():
                other = [item["top_training_motif_profiles"][k][token] for item in items]
                if any(item["reference_start_frequency"] != first_profile["reference_start_frequency"] for item in other):
                    raise ValueError("Top motif reference profiles differ")
                profiles[k][token] = {
                    "reference_start_frequency": first_profile["reference_start_frequency"],
                    "autoregressive_start_frequency_mean": np.mean([p["autoregressive_start_frequency"] for p in other], axis=0).tolist(),
                    "vae_start_frequency_mean": np.mean([p["vae_start_frequency"] for p in other], axis=0).tolist(),
                }
        groups[group] = {"reference_count": first["reference_count"], "target_log_strength": first["target_log_strength"],
                         "reference": first["reference"], "methods": methods, "position_base_profiles": base_profile,
                         "top_training_motif_profiles": profiles}
    overall = {method: {
        metric: float(np.mean([run["groups"][g][method][metric] for run in runs for g in GROUPS]))
        for metric in METRICS
    } for method in METHODS}
    result = {
        "protocol": "Descriptive means across groups and runs; no hypothesis testing or model selection. "
                    "AR varies only sampling seed; VAE varies training and sampling seed. Training-group references are not independent tests.",
        "sample_count_per_group": runs[0]["configuration"]["samples_per_group"],
        "vae_seeds": [r["configuration"]["vae_seed"] for r in runs],
        "sample_seeds": [r["configuration"]["sample_seed"] for r in runs],
        "dataset_sha256": runs[0]["dataset_sha256"],
        "vae_epochs_run": [r["configuration"]["vae_epochs_run"] for r in runs],
        "training_tertile_boundaries": runs[0]["training_tertile_boundaries"],
        "assumed_alignment": runs[0]["assumed_alignment"], "scan_rule": runs[0]["scan_rule"],
        "overall_mean_across_groups_and_runs": overall, "groups": groups,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"overall": overall, "epochs": result["vae_epochs_run"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
