#!/usr/bin/env python3
"""Audit TSS reporter data, its sequence coordinates, and overlap with course data."""
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

EXPECTED_BLOB = {"mpra": "f8ea0e51fd7da4b68ca0c29bcf1250a199a36dc3",
                 "genome": "9cf442042b81aaa4854e2b7e24d051d44d06ae2e"}
SPLITS = ("train", "validation", "test")

class GenomeMismatchError(ValueError):
    """The published sequence differs from its stated genome interval."""

class GenomeBoundsError(ValueError):
    """The published interval crosses the reference FASTA's linear boundary."""


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def tss_window(row: dict, genome: str) -> str:
    if row["category"] != "tss":
        raise ValueError("Only original TSS records have a guaranteed alignment")
    variant = row["variant"].upper()
    start, end, tss = int(row["start"]), int(row["end"]), int(row["tss_position"])
    if len(variant) != 150 or end - start != 150:
        raise ValueError("Invalid 150 bp genomic interval")
    if not 0 <= start < end <= len(genome):
        raise GenomeBoundsError("Interval outside linear genome bounds")
    genomic = genome[start:end]
    if row["strand"] == "+":
        expected = genomic
        if tss - start != 120: raise ValueError("Unexpected plus TSS coordinate")
    elif row["strand"] == "-":
        expected = reverse_complement(genomic)
        if end - tss != 120: raise ValueError("Unexpected minus TSS coordinate")
    else:
        raise ValueError("Unknown strand")
    if variant != expected:
        raise GenomeMismatchError("Sequence does not match the reported genomic interval and strand")
    return variant[70:120]  # Relative to the sequence TSS index 120: -50 through -1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mpra", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--output", default="reports/m4_ecoli_mpra_audit_2026-11-03.json")
    parser.add_argument("--rows-output", default="outputs/m4_external/urtecho_tss_overlap.csv")
    args = parser.parse_args()
    mpra, genome_path = Path(args.mpra), Path(args.genome)
    if git_blob(mpra) != EXPECTED_BLOB["mpra"] or git_blob(genome_path) != EXPECTED_BLOB["genome"]:
        raise ValueError("Upstream file does not match the recorded Git blob SHA-1")
    fasta = genome_path.read_text(encoding="utf-8").splitlines()
    if not fasta[0].startswith(">U00096.2 "):
        raise ValueError("Unexpected genome accession")
    genome = "".join(fasta[1:]).upper()
    with mpra.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    tss_rows = [row for row in rows if row["category"] == "tss"]
    cfg = load_config(ROOT / "configs/base.toml")["data"]
    seqs, _ = load_promoter_arrays(args.data_dir, cfg["sequence_file"], cfg["label_file"], 50)
    indices = load_fixed_split_indices(ROOT / cfg["split_file"], len(seqs))
    membership = defaultdict(set)
    for name, subset in zip(SPLITS, indices):
        for i in subset: membership[str(seqs[i]).upper()].add(name)
    reference = sorted(membership)
    search = AlignedNeighborIndex(reference, max_mismatches=5)
    counts = Counter()
    split_counts = {name: Counter() for name in SPLITS}
    values = []
    unique_windows = set()
    candidate_windows = set()
    candidate_counts = Counter()
    nearest_shift_counts = Counter()
    output_rows = []
    for number, row in enumerate(tss_rows):
        value = float(row["expn_med_fitted_scaled"])
        if not np.isfinite(value):
            raise ValueError("Non-finite reporter value")
        try:
            window = tss_window(row, genome)
        except (GenomeMismatchError, GenomeBoundsError) as error:
            status = "genome_interval_sequence_mismatch" if isinstance(error, GenomeMismatchError) else "genome_interval_out_of_bounds"
            counts[status] += 1
            output_rows.append({"tss_name": row["tss_name"], "tss_position": row["tss_position"],
                                "strand": row["strand"], "coordinate_status": status,
                                "sequence_50bp": "", "mpra_expression_scaled": value, "active": row["active"],
                                "nearest_mismatches_either_strand_up_to_5": "", "exact_forward_any_course_split": "",
                                "exact_shift5_course_splits": "", "nearest_exact_shift": "",
                                "nearest_exact_shift_abs": "", "candidate_after_overlap_screen": "",
                                **{f"nearest_{name}_mismatches_up_to_5": "" for name in SPLITS}})
            continue
        unique_windows.add(window)
        values.append(value)
        forward = search.find(window)
        reverse = search.find(reverse_complement(window))
        all_hits = forward.copy()
        for key, mismatches in reverse.items():
            all_hits[key] = min(all_hits.get(key, 51), mismatches)
        nearest = min(all_hits.values(), default=None)
        forward_nearest = min(forward.values(), default=None)
        shifted_splits = set()
        shift_distances = []
        variant = row["variant"].upper()
        for shift in range(-5, 6):
            shifted = variant[70 + shift:120 + shift]
            for query in (shifted, reverse_complement(shifted)):
                if query in membership:
                    shifted_splits.update(membership[query])
                    shift_distances.append(shift)
        nearest_shift = min(shift_distances, key=lambda x: (abs(x), x)) if shift_distances else None
        if nearest_shift is not None:
            nearest_shift_counts[nearest_shift] += 1
        candidate = nearest is None and not shifted_splits
        if candidate:
            candidate_windows.add(window)
            candidate_counts["rows"] += 1
            candidate_counts["active"] += row["active"] == "active"
            candidate_counts["inactive"] += row["active"] == "inactive"
            candidate_counts["positive_scaled_expression"] += value > 0
            candidate_counts["negative_scaled_expression"] += value < 0
        counts["tss_rows"] += 1
        counts["exact_forward"] += forward_nearest == 0
        counts["exact_either_strand"] += nearest == 0
        counts["within_5_mismatches_either_strand"] += nearest is not None
        counts["no_match_within_5_either_strand"] += nearest is None
        counts["exact_within_tss_shift5_either_strand"] += bool(shifted_splits)
        counts["no_aligned_90pct_or_shift5_exact"] += nearest is None and not shifted_splits
        counts["positive_scaled_expression"] += value > 0
        counts["negative_scaled_expression"] += value < 0
        counts["active"] += row["active"] == "active"
        line = {"tss_name": row["tss_name"], "tss_position": row["tss_position"],
                "strand": row["strand"], "coordinate_status": "verified", "sequence_50bp": window,
                "mpra_expression_scaled": value, "active": row["active"],
                "nearest_mismatches_either_strand_up_to_5": "" if nearest is None else nearest,
                "exact_forward_any_course_split": forward_nearest == 0,
                "exact_shift5_course_splits": ";".join(sorted(shifted_splits)),
                "nearest_exact_shift": "" if nearest_shift is None else nearest_shift,
                "nearest_exact_shift_abs": "" if nearest_shift is None else abs(nearest_shift),
                "candidate_after_overlap_screen": candidate}
        for name in SPLITS:
            distances = [distance for i, distance in all_hits.items() if name in membership[reference[i]]]
            closest = min(distances, default=None)
            split_counts[name]["exact"] += closest == 0
            split_counts[name]["within_5"] += closest is not None
            split_counts[name]["exact_shift5"] += name in shifted_splits
            line[f"nearest_{name}_mismatches_up_to_5"] = "" if closest is None else closest
        output_rows.append(line)
        if number % 3000 == 0: print(f"Checked {number}/{len(tss_rows)}", flush=True)
    result = {
        "source": {"repository": "https://github.com/KosuriLab/ecoli_promoter_mpra",
                   "commit": "484abdb6ba2d398501009ee6eb25ca7f190d3b4c",
                   "mpra_git_blob_sha1": git_blob(mpra), "genome_git_blob_sha1": git_blob(genome_path),
                   "mpra_sha256": hashlib.sha256(mpra.read_bytes()).hexdigest(),
                   "genome_sha256": hashlib.sha256(genome_path.read_bytes()).hexdigest()},
        "course": {"sequence_sha256": hashlib.sha256((Path(args.data_dir) / cfg["sequence_file"]).read_bytes()).hexdigest(),
                   "strength_sha256": hashlib.sha256((Path(args.data_dir) / cfg["label_file"]).read_bytes()).hexdigest(),
                   "split_sha256": hashlib.sha256((ROOT / cfg["split_file"]).read_bytes()).hexdigest(),
                   "rows": len(seqs), "unique_sequences": len(reference)},
        "external": {"all_rows": len(rows), "category_counts": dict(Counter(row["category"] for row in rows)),
                     "tss_150bp_validated_rows": counts["tss_rows"], "unique_tss_windows_50bp": len(unique_windows),
                     "tss_strands": dict(Counter(row["strand"] for row in tss_rows)),
                     "scaled_expression_min": float(min(values)), "scaled_expression_max": float(max(values)),
                     "scaled_expression_median": float(np.median(values)), "counts": dict(counts)},
        "overlap": {"definition": "Aligned Hamming distance of 50 bp TSS-upstream windows, query in both forward and reverse-complement orientations. Separately, check exact matches with the 50 bp window shifted from -5 to +5 bases. No gaps or near matches at shifted positions.",
                    "counts_by_course_split": {key: dict(value) for key, value in split_counts.items()},
                    "nearest_exact_shift_counts": {str(k): nearest_shift_counts[k] for k in sorted(nearest_shift_counts)},
                    "candidate_after_combined_screen": {**dict(candidate_counts), "unique_windows_50bp": len(candidate_windows)},
                    "split_counts_can_overlap": True},
        "limitations": "Rows whose interval crosses linear FASTA bounds or whose sequence differs from the declared genomic interval are excluded from overlap analysis and retained in the row audit. "
                       "Different experiment and reporter context; scaled MPRA values may be negative and are not comparable to course log10 labels. "
                       "Sequence non-match at 90% aligned identity is not proof of genomic independence. "
                       "Rows and sequences may repeat. Test split used only for overlap audit, not predictive evaluation.",
    }
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows_output = Path(args.rows_output)
    if not rows_output.is_absolute(): rows_output = ROOT / rows_output
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    with rows_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
