#!/usr/bin/env python3
"""Check whether screened external windows lie near mapped course sequences."""
from __future__ import annotations
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays
from promoter_ml.external_overlap import reverse_complement
from audit_ecoli_mpra_external import EXPECTED_BLOB, SPLITS, git_blob

GAPS = (0, 50, 100, 200)  # Diagnostic thresholds chosen before computing results.
WINDOW = 50


def exact_genome_hits(genome, sequence):
    """Return every circular-genome start and orientation of an exact match."""
    sequence = sequence.upper()
    if not genome or not sequence or len(sequence) > len(genome) or set(sequence) - set("ACGT"):
        raise ValueError("Invalid reference genome or query")
    expanded = genome + genome[:len(sequence) - 1]
    hits = []
    for orientation, query in (("+", sequence), ("-", reverse_complement(sequence))):
        start = expanded.find(query)
        while 0 <= start < len(genome):
            hits.append((start, orientation))
            start = expanded.find(query, start + 1)
    return hits


def nearest_gap(start, starts, genome_length, width=50):
    """Minimum base gap between equal-length half-open intervals on a circular genome."""
    if not starts:
        return None
    if not 0 <= start < genome_length:
        raise ValueError("Start outside genome")
    i = bisect_left(starts, start)
    candidates = (starts[i % len(starts)], starts[(i - 1) % len(starts)])
    distance = min(min(abs(start - x), genome_length - abs(start - x)) for x in candidates)
    return max(0, distance - width)


def external_start(row, genome):
    """Validate the original audited TSS-relative 50 bp window on either strand."""
    tss = int(row["tss_position"])
    if row["strand"] == "+":
        start, end = tss - 50, tss
        sequence = genome[start:end]
    elif row["strand"] == "-":
        start, end = tss, tss + 50
        sequence = reverse_complement(genome[start:end])
    else:
        raise ValueError("Unexpected strand")
    if not 0 <= start < end <= len(genome) or sequence != row["sequence_50bp"]:
        raise ValueError("External window differs from genome or crosses reference boundary")
    return start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--refined-rows", required=True)
    parser.add_argument("--output", default="reports/m4_ecoli_genome_proximity_2026-11-03.json")
    parser.add_argument("--rows-output", default="outputs/m4_external/urtecho_genome_proximity.csv")
    args = parser.parse_args()
    previous = json.loads((ROOT / "reports/m4_ecoli_mpra_refined_audit_2026-11-03.json").read_text())
    config = load_config(ROOT / "configs/base.toml")["data"]
    data_dir, genome_path, input_path = Path(args.data_dir), Path(args.genome), Path(args.refined_rows)
    if git_blob(genome_path) != EXPECTED_BLOB["genome"]:
        raise ValueError("Genome Git blob does not match")
    for name, path in (("sequence_sha256", data_dir / config["sequence_file"]),
                       ("strength_sha256", data_dir / config["label_file"]),
                       ("split_sha256", ROOT / config["split_file"]),
                       ("genome_sha256", genome_path)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != previous["inputs"][name]:
            raise ValueError(f"Provenance mismatch: {name}")
    lines = genome_path.read_text(encoding="utf-8").splitlines()
    if not lines[0].startswith(">U00096.2 "): raise ValueError("Unexpected genome")
    genome = "".join(lines[1:]).upper()
    sequences, _ = load_promoter_arrays(data_dir, config["sequence_file"], config["label_file"], WINDOW)
    split_indices = load_fixed_split_indices(ROOT / config["split_file"], len(sequences))
    split_of = {}
    for split, indices in zip(SPLITS, split_indices):
        split_of.update({int(i): split for i in indices})
    starts = {split: set() for split in SPLITS}
    mapping = Counter()
    for i, sequence in enumerate(sequences):
        hits = exact_genome_hits(genome, str(sequence))
        split = split_of[i]
        mapping["course_records"] += 1
        mapping[f"{split}_records"] += 1
        if hits:
            mapping["mapped_records"] += 1
            mapping[f"{split}_mapped_records"] += 1
            mapping["records_with_multiple_genomic_starts"] += len({pos for pos, _ in hits}) > 1
            mapping["records_with_multiple_orientation_hits"] += len(hits) > 1
            for pos, _ in hits: starts[split].add(pos)
        else:
            mapping["unmapped_records"] += 1
            mapping[f"{split}_unmapped_records"] += 1
        if i and i % 2000 == 0: print(f"Mapped {i}/{len(sequences)} course sequences", flush=True)
    starts = {split: sorted(values) for split, values in starts.items()}
    all_starts = sorted(set().union(*[set(values) for values in starts.values()]))
    with input_path.open(encoding="utf-8", newline="") as stream:
        prior_rows = list(csv.DictReader(stream))
    if len(prior_rows) != previous["counts"]["prior_candidates"]:
        raise ValueError("Unexpected refined row count")
    candidate_rows = [r for r in prior_rows if r["shifted_approximate_match"] == "False"]
    if len(candidate_rows) != previous["counts"]["remaining_after_refined_screen"]:
        raise ValueError("Unexpected refined candidate count")
    counts, by_split = Counter(), {split: Counter() for split in SPLITS}
    passed, output_rows = [], []
    for row in candidate_rows:
        if row["coordinate_status"] != "verified" or row["candidate_after_overlap_screen"] != "True":
            raise ValueError("Invalid candidate status")
        start = external_start(row, genome)
        gap = nearest_gap(start, all_starts, len(genome), WINDOW)
        if gap is None: raise ValueError("No course sequences mapped")
        counts["input_records"] += 1
        out = dict(row)
        out["genome_start_0based"] = start
        out["nearest_course_interval_gap_bp"] = gap
        out["near_course_200bp"] = gap <= 200
        for limit in GAPS: counts[f"within_gap_{limit}_bp"] += gap <= limit
        for split in SPLITS:
            value = nearest_gap(start, starts[split], len(genome), WINDOW)
            out[f"nearest_{split}_interval_gap_bp"] = "" if value is None else value
            for limit in GAPS:
                by_split[split][f"within_gap_{limit}_bp"] += value is not None and value <= limit
        if gap > 200:
            passed.append(row)
        output_rows.append(out)
    result = {
        "protocol": {"rule": "Exact-match full 50 bp course sequence to U00096.2 in either orientation, allowing genome-origin wrapping. "
                             "For previously screened external windows, compute minimal circular interval gap to mapped course sequences. "
                             "200 bp is the prespecified conservative screening threshold; other thresholds are sensitivity counts.",
                     "thresholds_bp": list(GAPS), "genome": "U00096.2", "genome_length": len(genome),
                     "split_boundary": "Test sequences used only for overlap/proximity screening; no model evaluation.",
                     "limitations": "Exact genomic mapping only; unmapped course sequences, TSS annotation, indels, "
                                    "long-range regulation and homology remain unresolved. 200 bp is a heuristic, not proof of independence."},
        "input_sha256": {"refined_rows": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                         "genome": hashlib.sha256(genome_path.read_bytes()).hexdigest(),
                         "course": previous["inputs"]["sequence_sha256"], "split": previous["inputs"]["split_sha256"]},
        "course_mapping": {**dict(mapping), "unique_genome_positions_by_split": {k: len(v) for k, v in starts.items()}},
        "candidate_proximity": {**dict(counts), "by_course_split": {k: dict(v) for k, v in by_split.items()},
                                "split_counts_can_overlap": True},
        "outside_200bp_of_mapped_course_windows": {
            "records": len(passed), "unique_50bp": len({r["sequence_50bp"] for r in passed}),
            "activity_counts": dict(Counter(r["active"] for r in passed)),
            "nonpositive_scaled_expression": sum(float(r["mpra_expression_scaled"]) <= 0 for r in passed),
        },
    }
    output = Path(args.output)
    rows_output = Path(args.rows_output)
    if not output.is_absolute(): output = ROOT / output
    if not rows_output.is_absolute(): rows_output = ROOT / rows_output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with rows_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
