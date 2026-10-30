#!/usr/bin/env python3
"""Reproduce M4 samples and inspect response using the fixed M3 screening blend."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.config import load_config
from promoter_ml.data import KmerFeaturizer, PositionKmerFeaturizer, load_fixed_split_indices, load_promoter_arrays
from promoter_ml.generation import ContinuousConditionalAutoregressiveGenerator
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import ConditionalSequenceVAE, MotifCNNStrengthPredictor, RidgeStrengthPredictor
from promoter_ml.structure_analysis import encode_sequences, position_js, position_kmer_frequency
from generate_conditional_position_baseline import assign_strength_groups
from train_motif_cnn_predictor import one_hot_sequences
from evaluate_m4_structure import statistics

GROUPS = ("weak", "medium", "strong")
SEEDS = ((20260912, 20261030), (20260913, 20261130), (20260914, 20261230))
COMPONENTS = ("motif_cnn", "global_kmer_ridge", "position_ridge")


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected finite nonempty one-dimensional values")
    return {"mean": float(values.mean()), "std": float(values.std()),
            "p10": float(np.quantile(values, 0.1)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9))}


def scores_summary(scores: np.ndarray, boundaries: np.ndarray) -> dict:
    bins = np.digitize(scores, boundaries, right=False)
    return {**describe(scores), "screening_group_fraction":
            {name: float(np.mean(bins == index)) for index, name in enumerate(GROUPS)}}


def score(model, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cnn, global_ridge, position_ridge, weights = model
    parts = np.column_stack([
        cnn.predict(one_hot_sequences(sequences)),
        global_ridge.predict(KmerFeaturizer(k_min=1, k_max=3, include_gc=True).transform(sequences)),
        position_ridge.predict(PositionKmerFeaturizer(sequence_length=50).transform(sequences)),
    ])
    return parts @ weights, parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/m4_condition_response_2026-10-30.json")
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/m2_motif_cnn_selected.toml")
    data = cfg["data"]
    with (ROOT / "configs/m3_screening_evaluator.toml").open("rb") as stream:
        evaluator = tomllib.load(stream)["screening_evaluator"]
    weights = np.array([evaluator[f"{name}_weight"] for name in COMPONENTS], dtype=float)
    if not np.isclose(weights.sum(), 1):
        raise ValueError("Evaluator weights must sum to one")
    seqs, strengths = load_promoter_arrays(args.data_dir, data["sequence_file"], data["label_file"], 50)
    train_idx, val_idx, _ = load_fixed_split_indices(ROOT / data["split_file"], len(seqs))
    dataset_sha256 = {"sequence": hashlib.sha256((Path(args.data_dir) / data["sequence_file"]).read_bytes()).hexdigest(),
                      "strength": hashlib.sha256((Path(args.data_dir) / data["label_file"]).read_bytes()).hexdigest(),
                      "split": hashlib.sha256((ROOT / data["split_file"]).read_bytes()).hexdigest()}
    train, validation = seqs[train_idx], seqs[val_idx]
    y_train, y_val = np.log10(strengths[train_idx]), np.log10(strengths[val_idx])
    labels, boundaries_list = assign_strength_groups(y_train)
    boundaries = np.asarray(boundaries_list)
    val_labels = np.array(GROUPS)[np.digitize(y_val, boundaries)]
    references = {g: encode_sequences(train[labels == g]) for g in GROUPS}
    ref_profiles = {g: position_kmer_frequency(references[g], 3) for g in GROUPS}
    targets = {g: float(y_train[labels == g].mean()) for g in GROUPS}
    intermediate_targets = {
        "weak_medium": (targets["weak"] + targets["medium"]) / 2,
        "medium_strong": (targets["medium"] + targets["strong"]) / 2,
    }

    print("Training fixed M3 screening model", flush=True)
    cnn = MotifCNNStrengthPredictor(seed=int(cfg["project"]["seed"]), **cfg["motif_cnn"])
    cnn.fit(one_hot_sequences(train), y_train, validation=(one_hot_sequences(validation), y_val))
    global_model = RidgeStrengthPredictor(alpha=10.0).fit(KmerFeaturizer(k_min=1, k_max=3, include_gc=True).transform(train), y_train)
    position_model = RidgeStrengthPredictor(alpha=10.0).fit(PositionKmerFeaturizer(sequence_length=50).transform(train), y_train)
    predictor = (cnn, global_model, position_model, weights)
    val_score, val_parts = score(predictor, validation)
    val_groups = {}
    for group in GROUPS:
        mask = val_labels == group
        val_groups[group] = {
            "count": int(mask.sum()),
            "observed_log_strength": describe(y_val[mask]),
            "screening_score": scores_summary(val_score[mask], boundaries),
            "score_minus_observed_mean": float(np.mean(val_score[mask] - y_val[mask])),
            "mae": float(np.mean(np.abs(val_score[mask] - y_val[mask]))),
        }
    val_total = regression_metrics(y_val, val_score)
    # M3 selection used this same development set. Stop if the fixed evaluator changes.
    if abs(val_total["pearson"] - float(evaluator["selection_value"])) > 1e-8:
        raise ValueError(f"Reproduced evaluator differs from saved M3 validation Pearson: {val_total['pearson']}")
    auto = ContinuousConditionalAutoregressiveGenerator(order=3, condition_bins=3).fit(train, y_train)
    runs = []
    for vae_seed, sample_seed in SEEDS:
        print(f"Generating run {vae_seed}", flush=True)
        vae = ConditionalSequenceVAE(hidden_size=128, latent_size=16, beta=0.2, epochs=120, seed=vae_seed)
        vae.fit(one_hot_sequences(train), y_train, validation=(one_hot_sequences(validation), y_val))
        previous = json.loads((ROOT / f"reports/m4_structure_seed_{vae_seed}.json").read_text(encoding="utf-8"))
        if previous["dataset_sha256"] != dataset_sha256:
            raise ValueError("Data or split differs from the preceding structure run")
        run = {"vae_seed": vae_seed, "sample_seed": sample_seed, "vae_epochs_run": len(vae.history), "groups": {}}
        for offset, group in enumerate(GROUPS):
            target = targets[group]
            generated = {
                "autoregressive": auto.sample(target, 500, sample_seed + offset),
                "vae": vae.sample(np.full(500, target), sample_seed + offset),
            }
            run["groups"][group] = {}
            for model_name, sample in generated.items():
                encoded = encode_sequences(sample)
                current = statistics(references[group], encoded)
                old = previous["groups"][group][f"{model_name}_vs_full_training_group"]
                for key in ("gc_mean", "position_base_l1_mean", "position_3mer_js_mean", "position_5mer_js_mean"):
                    if not np.isclose(current[key], old[key], rtol=0, atol=1e-12):
                        raise ValueError(f"Samples differ from prior structure run: {vae_seed} {group} {model_name} {key}")
                values, parts = score(predictor, sample)
                sample_profile = position_kmer_frequency(encoded, 3)
                run["groups"][group][model_name] = {
                    "target_log_strength": target,
                    "sample_count": len(sample),
                    "sample_sequence_sha256": hashlib.sha256(("\n".join(sample.tolist()) + "\n").encode()).hexdigest(),
                    "screening_score": scores_summary(values, boundaries),
                    "component_mean": {name: float(parts[:, i].mean()) for i, name in enumerate(COMPONENTS)},
                    "gc_mean": current["gc_mean"],
                    "position_3mer_js_against_training_groups":
                        {ref_group: float(position_js(ref_profiles[ref_group], sample_profile).mean()) for ref_group in GROUPS},
                }
        run["intermediate"] = {}
        for offset, (name, target) in enumerate(intermediate_targets.items()):
            generated = {
                "autoregressive": auto.sample(target, 500, sample_seed + 3000 + offset),
                "vae": vae.sample(np.full(500, target), sample_seed + 3000 + offset),
            }
            run["intermediate"][name] = {"target_log_strength": target}
            for model_name, sample in generated.items():
                values, parts = score(predictor, sample)
                encoded = encode_sequences(sample)
                run["intermediate"][name][model_name] = {
                    "sample_count": len(sample),
                    "sample_sequence_sha256": hashlib.sha256(("\n".join(sample.tolist()) + "\n").encode()).hexdigest(),
                    "screening_score": scores_summary(values, boundaries),
                    "component_mean": {key: float(parts[:, i].mean()) for i, key in enumerate(COMPONENTS)},
                    "gc_mean": float(np.mean((encoded == 1) | (encoded == 2))),
                }
        runs.append(run)

    summary = {}
    for group in GROUPS:
        summary[group] = {"target_log_strength": targets[group]}
        for model_name in ("autoregressive", "vae"):
            entries = [run["groups"][group][model_name] for run in runs]
            summary[group][model_name] = {
                "screening_mean_across_runs": describe(np.array([x["screening_score"]["mean"] for x in entries])),
                "component_mean_across_runs": {name: float(np.mean([x["component_mean"][name] for x in entries])) for name in COMPONENTS},
                "screening_group_fraction_across_runs":
                    {name: float(np.mean([x["screening_score"]["screening_group_fraction"][name] for x in entries])) for name in GROUPS},
                "gc_mean_across_runs": float(np.mean([x["gc_mean"] for x in entries])),
                "position_3mer_js_against_training_groups":
                    {name: float(np.mean([x["position_3mer_js_against_training_groups"][name] for x in entries])) for name in GROUPS},
            }
    intermediate_summary = {}
    for group, target in intermediate_targets.items():
        intermediate_summary[group] = {"target_log_strength": target}
        for model_name in ("autoregressive", "vae"):
            entries = [run["intermediate"][group][model_name] for run in runs]
            intermediate_summary[group][model_name] = {
                "screening_mean_across_runs": describe(np.array([x["screening_score"]["mean"] for x in entries])),
                "component_mean_across_runs": {name: float(np.mean([x["component_mean"][name] for x in entries])) for name in COMPONENTS},
                "gc_mean_across_runs": float(np.mean([x["gc_mean"] for x in entries])),
            }
    result = {
        "protocol": "M4 structure experiment settings and seeds fixed; 500 samples per model per group per run. "
                    "Two additional conditions are the midpoints of adjacent group target means, sampled with separate seeds. "
                    "Same model weights as M3; fitted on training data; CNN early stopping and blend weight selection use reused development validation data. "
                    "No test metrics or model selection this round.",
        "boundary": "Screening scores are model outputs, not experimentally measured strengths; predictor and generators share training data. "
                    "Development validation has been repeatedly used, and prior test data was explored historically. No independent validation is claimed.",
        "dataset_sha256": dataset_sha256,
        "numpy_version": np.__version__,
        "weights": dict(zip(COMPONENTS, weights.tolist())),
        "training_group_boundaries": boundaries_list,
        "validation": {"overall": val_total, "groups_by_observed_label": val_groups,
                       "component_overall": {name: regression_metrics(y_val, val_parts[:, i]) for i, name in enumerate(COMPONENTS)}},
        "summary": summary, "intermediate_summary": intermediate_summary, "runs": runs,
    }
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"validation": val_total, "summary": summary, "intermediate": intermediate_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
