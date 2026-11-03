#!/usr/bin/env python3
"""Extend the previous overlap screen to shifted, approximate 50 bp windows."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays
from promoter_ml.external_overlap import AlignedNeighborIndex, reverse_complement
from audit_ecoli_mpra_external import EXPECTED_BLOB, SPLITS, git_blob, tss_window

RADIUS = 5
MAX_MISMATCHES = 5


def shifted_neighbours(variant, index, membership):
    """All fixed-length shifted alignments; no gaps. Return minimal mismatch per split."""
    if len(variant) != 150 or set(variant) - set("ACGT"):
        raise ValueError("Expected a validated 150 bp sequence")
    best = None
    by_split = {}
    for shift in range(-RADIUS, RADIUS + 1):
        window = variant[70 + shift:120 + shift]
        for orientation, query in (("forward", window), ("reverse_complement", reverse_complement(window))):
            for reference_index, distance in index.find(query).items():
                witness = (distance, abs(shift), shift, orientation, reference_index)
                if best is None or witness < best:
                    best = witness
                for split in membership[index.sequences[reference_index]]:
                    if split not in by_split or witness < by_split[split]:
                        by_split[split] = witness
    return best, by_split


def duplicate_summary(rows):
    """Describe repeated 50 bp inputs without aggregating their experimental values."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["sequence_50bp"]].append(row)
    repeated = [group for group in groups.values() if len(group) > 1]
    return {
        "rows": len(rows), "unique_50bp": len(groups),
        "duplicate_50bp_groups": len(repeated),
        "rows_in_duplicate_groups": sum(map(len, repeated)),
        "extra_rows_beyond_unique_50bp": len(rows) - len(groups),
        "groups_with_both_activity_labels": sum(len({r["active"] for r in group}) > 1 for group in repeated),
        "groups_with_different_scaled_values": sum(len({r["mpra_expression_scaled"] for r in group}) > 1 for group in repeated),
        "groups_with_different_150bp_context": sum(len({r["_variant"] for r in group}) > 1 for group in repeated),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mpra", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--prior-rows", required=True)
    parser.add_argument("--output", default="reports/m4_ecoli_mpra_refined_audit_2026-11-03.json")
    parser.add_argument("--rows-output", default="outputs/m4_external/urtecho_tss_refined_overlap.csv")
    args = parser.parse_args()
    mpra, genome_path, prior_path = Path(args.mpra), Path(args.genome), Path(args.prior_rows)
    if git_blob(mpra) != EXPECTED_BLOB["mpra"] or git_blob(genome_path) != EXPECTED_BLOB["genome"]:
        raise ValueError("Upstream file does not match recorded Git blob")
    previous = json.loads((ROOT / "reports/m4_ecoli_mpra_audit_2026-11-03.json").read_text(encoding="utf-8"))
    config = load_config(ROOT / "configs/base.toml")["data"]
    data_dir = Path(args.data_dir)
    for name, path in (("sequence_sha256", data_dir / config["sequence_file"]),
                       ("strength_sha256", data_dir / config["label_file"]),
                       ("split_sha256", ROOT / config["split_file"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != previous["course"][name]:
            raise ValueError(f"Course provenance differs from previous audit: {name}")
    with mpra.open(encoding="utf-8", newline="") as stream:
        source_rows = [r for r in csv.DictReader(stream, delimiter="\t") if r["category"] == "tss"]
    with prior_path.open(encoding="utf-8", newline="") as stream:
        prior_rows = list(csv.DictReader(stream))
    if len(source_rows) != len(prior_rows):
        raise ValueError("Prior row count does not match source")
    fasta = genome_path.read_text(encoding="utf-8").splitlines()
    if not fasta[0].startswith(">U00096.2 "): raise ValueError("Unexpected reference genome")
    genome = "".join(fasta[1:]).upper()
    seqs, _ = load_promoter_arrays(data_dir, config["sequence_file"], config["label_file"], 50)
    indices = load_fixed_split_indices(ROOT / config["split_file"], len(seqs))
    membership = defaultdict(set)
    for split, subset in zip(SPLITS, indices):
        for i in subset: membership[str(seqs[i]).upper()].add(split)
    index = AlignedNeighborIndex(sorted(membership), MAX_MISMATCHES)
    candidates, output_rows = [], []
    counts, split_counts = Counter(), Counter()
    for i, (original, prior) in enumerate(zip(source_rows, prior_rows)):
        for name in ("tss_name", "tss_position", "strand", "active"):
            if original[name] != prior[name]:
                raise ValueError(f"Prior CSV differs from source at row {i}: {name}")
        if float(original["expn_med_fitted_scaled"]) != float(prior["mpra_expression_scaled"]):
            raise ValueError("Prior CSV expression value differs from source")
        if prior["candidate_after_overlap_screen"] != "True":
            continue
        if prior["coordinate_status"] != "verified":
            raise ValueError("Candidate has unverified coordinate")
        window = tss_window(original, genome)
        if window != prior["sequence_50bp"]:
            raise ValueError("Candidate sequence differs from source")
        variant = original["variant"].upper()
        if index.find(window) or index.find(reverse_complement(window)):
            raise ValueError("Prior candidate has a fixed-window neighbour within 5 mismatches")
        for shift in range(-RADIUS, RADIUS + 1):
            shifted = variant[70 + shift:120 + shift]
            if shifted in membership or reverse_complement(shifted) in membership:
                raise ValueError("Prior candidate has an exact shifted match")
        best, splits = shifted_neighbours(variant, index, membership)
        result = dict(prior)
        result["shifted_approximate_match"] = bool(best)
        result["minimum_shifted_mismatches_up_to_5"] = "" if best is None else best[0]
        result["best_shift"] = "" if best is None else best[2]
        result["best_orientation"] = "" if best is None else best[3]
        result["course_splits_with_shifted_approximate_match"] = ";".join(sorted(splits))
        for split in SPLITS:
            result[f"nearest_{split}_shifted_mismatches_up_to_5"] = splits[split][0] if split in splits else ""
            split_counts[split] += split in splits
        counts["prior_candidates"] += 1
        counts["new_shifted_approximate_match"] += best is not None
        counts["remaining_after_refined_screen"] += best is None
        if best is not None:
            counts[f"mismatch_{best[0]}"] += 1
        else:
            result["_variant"] = variant
            candidates.append(result)
        output_rows.append(result)
        if counts["prior_candidates"] % 1000 == 0:
            print(f"Checked {counts['prior_candidates']} prior candidates", flush=True)
    if counts["prior_candidates"] != previous["overlap"]["candidate_after_combined_screen"]["rows"]:
        raise ValueError("Prior candidate count differs from recorded audit")
    duplicates = duplicate_summary(candidates)
    activity = Counter(row["active"] for row in candidates)
    nonpositive = sum(float(row["mpra_expression_scaled"]) <= 0 for row in candidates)
    candidate_groups = defaultdict(list)
    for row in candidates:
        candidate_groups[row["sequence_50bp"]].append(row)
    for row in output_rows:
        group = candidate_groups.get(row["sequence_50bp"], []) if not row["shifted_approximate_match"] else []
        row["remaining_same_50bp_row_count"] = len(group) if group else ""
        row["remaining_distinct_150bp_contexts"] = len({item["_variant"] for item in group}) if group else ""
    # Keep 150 bp source context out of the derived CSV; original file can be joined by identifier.
    for row in output_rows: row.pop("_variant", None)
    result = {
        "protocol": {"population": "Only prior audit candidates; preceding audit fully regenerated and checked separately.",
                     "shift_range": [-RADIUS, RADIUS], "max_substitutions": MAX_MISMATCHES,
                     "sequence_length": 50, "orientations": ["forward", "reverse_complement"],
                     "reference": "All course sequences in train, validation, and test; test used only for sequence overlap.",
                     "limitations": "Aligned substitutions only; no gaps, homology or genomic-neighbour exclusion. No model evaluated."},
        "inputs": {"prior_csv_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                   "mpra_sha256": hashlib.sha256(mpra.read_bytes()).hexdigest(),
                   "genome_sha256": hashlib.sha256(genome_path.read_bytes()).hexdigest(),
                   **previous["course"]},
        "counts": dict(counts), "matches_by_course_split": dict(split_counts),
        "split_counts_can_overlap": True,
        "remaining": {"activity_counts": dict(activity), "nonpositive_scaled_expression": nonpositive,
                      "duplicate_summary": duplicates},
    }
    for key in (args.output, args.rows_output):
        path = Path(key)
        if not path.is_absolute(): path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows_path = Path(args.rows_output)
    if not rows_path.is_absolute(): rows_path = ROOT / rows_path
    with rows_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
