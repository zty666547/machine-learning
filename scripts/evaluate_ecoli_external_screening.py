#!/usr/bin/env python3
"""Apply the fixed M3 screen to the frozen, filtered external candidate set."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
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
from promoter_ml.metrics import _rankdata, regression_metrics
from promoter_ml.models import MotifCNNStrengthPredictor, RidgeStrengthPredictor
from train_motif_cnn_predictor import one_hot_sequences
from evaluate_m4_condition_response import score, COMPONENTS


def ranking_metrics(values, score_values, labels):
    """Spearman with average tied ranks; AUROC and threshold-grouped average precision."""
    values, scores = np.asarray(values, float), np.asarray(score_values, float)
    labels = np.asarray(labels, int)
    if values.ndim != 1 or len(values) != len(scores) or len(values) != len(labels) or len(values) < 2:
        raise ValueError("Expected matching non-empty one-dimensional arrays")
    if not np.isfinite(values).all() or not np.isfinite(scores).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("Metrics require finite values and both binary classes")
    value_ranks, score_ranks = _rankdata(values), _rankdata(scores)
    spearman = float(np.corrcoef(value_ranks, score_ranks)[0, 1]) if np.std(value_ranks) and np.std(score_ranks) else None
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    # _rankdata starts at zero; convert to conventional 1-based ranks.
    auroc = float((score_ranks[labels == 1].sum() + positives - positives * (positives + 1) / 2) / (positives * negatives))
    order = np.argsort(-scores, kind="stable")
    tp = fp = 0
    ap = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        new_tp = int(labels[order[start:end]].sum())
        tp += new_tp
        fp += end - start - new_tp
        ap += (new_tp / positives) * (tp / (tp + fp))
        start = end
    return {"count": len(values), "active": positives, "inactive": negatives,
            "active_fraction": positives / len(labels), "spearman": spearman,
            "auroc": auroc, "average_precision": float(ap)}


def unique_sequences(rows):
    """Median raw MPRA values per 50 bp; require consistent binary labels."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["sequence_50bp"]].append(row)
    result = []
    for sequence in sorted(groups):
        group = groups[sequence]
        labels = {r["active"] for r in group}
        if len(labels) != 1 or not labels <= {"active", "inactive"}:
            raise ValueError("Conflicting or invalid activity labels for repeated input")
        result.append({"sequence_50bp": sequence, "active": next(iter(labels)),
                       "mpra_expression_scaled": float(np.median([float(r["mpra_expression_scaled"]) for r in group])),
                       "record_count": len(group)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--protocol", default="configs/m4_external_eval_protocol_2026-11-03.toml")
    parser.add_argument("--output", default="reports/m4_external_screening_results_2026-11-03.json")
    parser.add_argument("--rows-output", default="outputs/m4_external/screening_scores_2026-11-03.csv")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    with protocol_path.open("rb") as f: protocol = tomllib.load(f)["evaluation"]
    if protocol["use_external_data_for_fitting_or_selection"] or protocol["primary_regression_metric"] != "spearman_on_original_scaled_mpra_values":
        raise ValueError("Unexpected protocol")
    csv_path = Path(args.candidate_csv)
    csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if csv_sha != protocol["candidate_csv_sha256"]:
        raise ValueError("Candidate CSV differs from frozen protocol")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if int(r["nearest_course_interval_gap_bp"]) > protocol["minimum_course_gap_bp_exclusive"]]
    if len(rows) != protocol["expected_candidate_rows"] or len({r["sequence_50bp"] for r in rows}) != protocol["expected_unique_sequences"]:
        raise ValueError("Candidate set differs from frozen protocol")
    if any(r["coordinate_status"] != "verified" or r["candidate_after_overlap_screen"] != "True" or
           r["shifted_approximate_match"] != "False" for r in rows):
        raise ValueError("Candidate provenance filter inconsistent")
    unique = unique_sequences(rows)
    cfg = load_config(ROOT / protocol["model_config"])
    data = cfg["data"]
    with (ROOT / protocol["evaluator_config"]).open("rb") as stream:
        evaluator = tomllib.load(stream)["screening_evaluator"]
    weights = np.array([evaluator[f"{name}_weight"] for name in COMPONENTS], dtype=float)
    if not np.allclose(weights, [0.5, 0.4, 0.1], atol=0, rtol=0):
        raise ValueError("Unexpected M3 blend weights")
    sequences, strength = load_promoter_arrays(args.data_dir, data["sequence_file"], data["label_file"], 50)
    train_i, val_i, _ = load_fixed_split_indices(ROOT / data["split_file"], len(sequences))
    train, val = sequences[train_i], sequences[val_i]
    y_train, y_val = np.log10(strength[train_i]), np.log10(strength[val_i])
    print("Fitting fixed screening predictor on course train/development data", flush=True)
    cnn = MotifCNNStrengthPredictor(seed=int(cfg["project"]["seed"]), **cfg["motif_cnn"])
    cnn.fit(one_hot_sequences(train), y_train, validation=(one_hot_sequences(val), y_val))
    global_model = RidgeStrengthPredictor(alpha=10).fit(KmerFeaturizer(k_min=1, k_max=3, include_gc=True).transform(train), y_train)
    positional_model = RidgeStrengthPredictor(alpha=10).fit(PositionKmerFeaturizer(sequence_length=50).transform(train), y_train)
    predictor = (cnn, global_model, positional_model, weights)
    val_score, _ = score(predictor, val)
    validation = regression_metrics(y_val, val_score)
    if abs(validation["pearson"] - float(evaluator["selection_value"])) > 1e-8:
        raise ValueError("Fixed predictor does not reproduce M3 development Pearson")
    ordered_sequences = np.array([r["sequence_50bp"] for r in unique])
    unique_score, unique_components = score(predictor, ordered_sequences)
    index = {sequence: i for i, sequence in enumerate(ordered_sequences)}
    positions = np.array([index[r["sequence_50bp"]] for r in rows], dtype=int)

    def metrics(group, selected_scores, parts):
        observed = np.array([float(row["mpra_expression_scaled"]) for row in group])
        binary = np.array([int(row["active"] == "active") for row in group])
        return {"blend": ranking_metrics(observed, selected_scores, binary),
                "components": {name: ranking_metrics(observed, parts[:, i], binary)
                               for i, name in enumerate(COMPONENTS)}}
    record_metrics = metrics(rows, unique_score[positions], unique_components[positions])
    unique_metrics = metrics(unique, unique_score, unique_components)
    result = {
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "candidate_csv_sha256": csv_sha,
        "input_sha256": {"sequence": hashlib.sha256((Path(args.data_dir) / data["sequence_file"]).read_bytes()).hexdigest(),
                         "strength": hashlib.sha256((Path(args.data_dir) / data["label_file"]).read_bytes()).hexdigest(),
                         "split": hashlib.sha256((ROOT / data["split_file"]).read_bytes()).hexdigest()},
        "model": {"seed": cfg["project"]["seed"], "weights": dict(zip(COMPONENTS, weights.tolist())),
                  "development_validation": validation},
        "record_level_primary": record_metrics,
        "unique_50bp_median_sensitivity": unique_metrics,
        "external_value_summary": {"negative": sum(float(r["mpra_expression_scaled"]) < 0 for r in rows),
                                   "zero": sum(float(r["mpra_expression_scaled"]) == 0 for r in rows),
                                   "positive": sum(float(r["mpra_expression_scaled"]) > 0 for r in rows),
                                   "activity_counts": dict(Counter(r["active"] for r in rows))},
        "interpretation": "Exploratory cross-assay ranking on previously filtered existing external sequences, not model selection or experimental validation of generated sequences. "
                          "External 150 bp reporter context vs 50 bp model input, potential relatedness, and label/assay differences remain.",
    }
    output = Path(args.output)
    output = output if output.is_absolute() else ROOT / output
    rows_output = Path(args.rows_output)
    rows_output = rows_output if rows_output.is_absolute() else ROOT / rows_output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with rows_output.open("w", encoding="utf-8", newline="") as f:
        fields = ["tss_name", "tss_position", "strand", "sequence_50bp", "mpra_expression_scaled", "active",
                  "nearest_course_interval_gap_bp", "record_count_for_same_50bp", "screening_score", *COMPONENTS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        counts = Counter(r["sequence_50bp"] for r in rows)
        for row, i in zip(rows, positions):
            out = {k: row[k] for k in fields[:7]}
            out["record_count_for_same_50bp"] = counts[row["sequence_50bp"]]
            out["screening_score"] = float(unique_score[i])
            out.update({name: float(unique_components[i, j]) for j, name in enumerate(COMPONENTS)})
            writer.writerow(out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
