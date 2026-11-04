#!/usr/bin/env python3
"""Screen previously evaluated external records for shifted global edit-distance overlap."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays
from promoter_ml.edit_distance_overlap import EditDistanceIndex, edlib
from promoter_ml.external_overlap import reverse_complement
from audit_ecoli_mpra_external import EXPECTED_BLOB, SPLITS, git_blob, tss_window
from evaluate_ecoli_external_screening import ranking_metrics, unique_sequences


def source_key(row):
    return (row["tss_name"], row["tss_position"], row["strand"],
            float(row["expn_med_fitted_scaled"] if "expn_med_fitted_scaled" in row else row["mpra_expression_scaled"]),
            row["active"])


def metrics(rows):
    if not rows or len({r["active"] for r in rows}) != 2:
        return None
    return ranking_metrics([float(r["mpra_expression_scaled"]) for r in rows],
                           [float(r["screening_score"]) for r in rows],
                           [int(r["active"] == "active") for r in rows])


def unique_metrics(rows):
    if not rows: return None
    unique = unique_sequences(rows)
    by_sequence = defaultdict(set)
    for row in rows: by_sequence[row["sequence_50bp"]].add(float(row["screening_score"]))
    if any(len(scores) != 1 for scores in by_sequence.values()):
        raise ValueError("The same 50 bp input has inconsistent fixed scores")
    for row in unique: row["screening_score"] = next(iter(by_sequence[row["sequence_50bp"]]))
    return metrics(unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mpra", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--protocol", default="configs/m4_edit_distance_protocol_2026-11-04.toml")
    parser.add_argument("--output", default="reports/m4_external_edit_distance_2026-11-04.json")
    parser.add_argument("--rows-output", default="outputs/m4_external/edit_distance_audit_2026-11-04.csv")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    with protocol_path.open("rb") as f: protocol = tomllib.load(f)["audit"]
    if (protocol["window_length"], protocol["max_edit_distance"], protocol["max_window_shift"]) != (50, 5, 5):
        raise ValueError("Unexpected frozen window, edit threshold or shift")
    if protocol["orientations"] != ["forward", "reverse_complement"]:
        raise ValueError("Unexpected frozen orientation rule")
    if protocol["distance"] != "global_levenshtein_substitution_insertion_deletion_each_cost_1" or not protocol["use_previous_scores_without_retraining"]:
        raise ValueError("Unexpected audit rule")
    scores_path, mpra_path, genome_path = Path(args.scores), Path(args.mpra), Path(args.genome)
    score_sha = hashlib.sha256(scores_path.read_bytes()).hexdigest()
    if score_sha != protocol["scores_csv_sha256"]:
        raise ValueError("Scored candidate file differs from frozen input")
    if git_blob(mpra_path) != EXPECTED_BLOB["mpra"] or git_blob(genome_path) != EXPECTED_BLOB["genome"]:
        raise ValueError("Public source files do not match previous audit")
    previous = json.loads((ROOT / "reports/m4_external_screening_results_2026-11-03.json").read_text(encoding="utf-8"))
    cfg = load_config(ROOT / "configs/base.toml")["data"]
    for name, path in (("sequence", Path(args.data_dir) / cfg["sequence_file"]),
                       ("strength", Path(args.data_dir) / cfg["label_file"]),
                       ("split", ROOT / cfg["split_file"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != previous["input_sha256"][name]:
            raise ValueError(f"Course {name} differs from previous scoring")
    with scores_path.open(encoding="utf-8", newline="") as f: scored_rows = list(csv.DictReader(f))
    if len(scored_rows) != protocol["external_records"]:
        raise ValueError("Unexpected number of scored external records")
    baseline = metrics(scored_rows)
    for metric in ("spearman", "auroc", "average_precision"):
        if abs(baseline[metric] - previous["record_level_primary"]["blend"][metric]) > 1e-12:
            raise ValueError("Previous recorded metric differs from input scores")
    with mpra_path.open(encoding="utf-8", newline="") as f:
        source_rows = [r for r in csv.DictReader(f, delimiter="\t") if r["category"] == "tss"]
    source_lookup = defaultdict(list)
    for row in source_rows: source_lookup[source_key(row)].append(row)
    lines = genome_path.read_text(encoding="utf-8").splitlines()
    if not lines[0].startswith(">U00096.2 "): raise ValueError("Unexpected genome")
    genome = "".join(lines[1:]).upper()
    sequences, _ = load_promoter_arrays(args.data_dir, cfg["sequence_file"], cfg["label_file"], 50)
    indices = load_fixed_split_indices(ROOT / cfg["split_file"], len(sequences))
    memberships = defaultdict(set)
    for split, subset in zip(SPLITS, indices):
        for i in subset: memberships[str(sequences[i]).upper()].add(split)
    reference = sorted(memberships)
    index = EditDistanceIndex(reference, protocol["max_edit_distance"])
    result_rows, kept = [], []
    counts, split_counts, distance_counts = Counter(), Counter(), Counter()
    for number, row in enumerate(scored_rows):
        matches = source_lookup.get(source_key(row), [])
        if len(matches) != 1:
            raise ValueError("External source row is missing or ambiguous")
        original = matches[0]
        if tss_window(original, genome) != row["sequence_50bp"]:
            raise ValueError("External sequence differs from validated source")
        variant = original["variant"].upper()
        best = None
        splits = set()
        for shift in range(-protocol["max_window_shift"], protocol["max_window_shift"] + 1):
            candidate = variant[70 + shift:120 + shift]
            for orientation, query in (("forward", candidate), ("reverse_complement", reverse_complement(candidate))):
                for ref_idx, distance in index.find(query).items():
                    witness = (distance, abs(shift), shift, orientation, ref_idx)
                    if best is None or witness < best: best = witness
                    splits.update(memberships[reference[ref_idx]])
        counts["records"] += 1
        counts["records_with_match"] += best is not None
        counts["records_without_match"] += best is None
        if best is None: kept.append(row)
        else:
            distance_counts[str(best[0])] += 1
            for split in splits: split_counts[split] += 1
        result_rows.append({**row, "edit_match": best is not None,
                            "minimum_edit_distance_up_to_5": "" if best is None else best[0],
                            "best_shift": "" if best is None else best[2],
                            "best_orientation": "" if best is None else best[3],
                            "matching_course_splits": ";".join(sorted(splits))})
        if (number + 1) % 200 == 0: print(f"Checked {number + 1}/{len(scored_rows)}", flush=True)
    backend_version = importlib.metadata.version("edlib") if edlib is not None else None
    result = {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "score_csv_sha256": score_sha,
        "course_data_sha256": previous["input_sha256"],
        "method": {"distance": protocol["distance"], "max_edits": 5, "shift_range": [-5, 5],
                   "reference_splits": list(SPLITS), "edlib_version": backend_version,
                   "fallback": "exact banded Python dynamic programming when edlib is unavailable",
                   "limitations": "Only global 50 bp windows, shifts +/-5, edit distance <=5. No longer-range or broader homology guarantee."},
        "counts": {**dict(counts), "matching_course_split_counts": dict(split_counts),
                   "minimum_edit_distance_counts": dict(distance_counts),
                   "split_counts_can_overlap": True,
                   "unique_50bp_with_match": len({r["sequence_50bp"] for r in result_rows if r["edit_match"]}),
                   "unique_50bp_without_match": len({r["sequence_50bp"] for r in result_rows if not r["edit_match"]})},
        "previous_fixed_record_metrics": baseline,
        "posthoc_without_edit_matches_record_metrics": metrics(kept),
        "posthoc_without_edit_matches_unique_median_metrics": unique_metrics(kept),
        "interpretation": "No retraining. Subset metrics are post-hoc sensitivity analysis on previously evaluated data; no model selection or new independent test.",
    }
    output, rows_output = Path(args.output), Path(args.rows_output)
    if not output.is_absolute(): output = ROOT / output
    if not rows_output.is_absolute(): rows_output = ROOT / rows_output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with rows_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
