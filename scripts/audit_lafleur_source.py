#!/usr/bin/env python3
"""Assess a published synthetic promoter library without using model predictions."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays
from promoter_ml.edit_distance_overlap import EditDistanceIndex, edlib
from promoter_ml.external_overlap import reverse_complement

MODULES = ("UP", "h35", "spacs", "h10", "disc", "ITR")


def nominal_window(record, shift=0):
    """Design-time ITR start only; this does not establish an experimental TSS."""
    parts = [str(record[name] or "").upper() for name in MODULES]
    sequence = "".join(parts)
    if not all(parts) or set(sequence) - set("ACGT"):
        raise ValueError("Missing or invalid promoter module")
    start = sum(len(part) for part in parts[:-1])
    end = start + shift
    if end < 50 or end > len(sequence):
        return None, start, len(sequence)
    return sequence[end - 50:end], start, len(sequence)


def main():
    import openpyxl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--previous-scores", required=True)
    parser.add_argument("--protocol", default="configs/m4_lafleur_source_audit_2026-11-05.toml")
    parser.add_argument("--output", default="reports/m4_lafleur_source_audit_2026-11-05.json")
    parser.add_argument("--rows-output", default="outputs/m4_external/lafleur_source_audit.csv")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    with protocol_path.open("rb") as f: protocol = tomllib.load(f)["source_audit"]
    if (protocol["window_length"], protocol["shift_max_bp"], protocol["max_global_edit_distance"]) != (50, 5, 5):
        raise ValueError("Unexpected frozen filter")
    if protocol["search_orientations"] != ["forward", "reverse_complement"] or protocol["use_strength_labels_for_model_evaluation"]:
        raise ValueError("Unexpected frozen scope")
    workbook_path, scores_path = Path(args.workbook), Path(args.previous_scores)
    for path, name in ((workbook_path, "supplement_sha256"), (scores_path, "previous_scores_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[name]:
            raise ValueError(f"Frozen input checksum mismatch: {name}")
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    sheet = workbook[protocol["workbook_sheet"]]
    values = iter(sheet.iter_rows(values_only=True))
    header = next(values)
    if not set(MODULES + ("ID", "high quality", "Txave [au]")).issubset(header):
        raise ValueError("Missing required workbook fields")
    records = [dict(zip(header, row)) for row in values if row[0] is not None]
    quality = [r for r in records if r["high quality"] == protocol["author_quality_flag"]]
    if len({r["ID"] for r in records}) != len(records):
        raise ValueError("Duplicate source IDs")
    cfg = load_config(ROOT / "configs/base.toml")["data"]
    source_data = ROOT / "reports/m4_external_screening_results_2026-11-03.json"
    previous = json.loads(source_data.read_text(encoding="utf-8"))
    for name, path in (("sequence", Path(args.data_dir) / cfg["sequence_file"]),
                       ("strength", Path(args.data_dir) / cfg["label_file"]),
                       ("split", ROOT / cfg["split_file"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != previous["input_sha256"][name]:
            raise ValueError(f"Course input differs from previous audit: {name}")
    sequences, _ = load_promoter_arrays(args.data_dir, cfg["sequence_file"], cfg["label_file"], 50)
    split_indices = load_fixed_split_indices(ROOT / cfg["split_file"], len(sequences))
    membership = defaultdict(set)
    for split, indices in zip(("train", "validation", "test"), split_indices):
        for idx in indices: membership[str(sequences[idx]).upper()].add(f"course_{split}")
    with scores_path.open(encoding="utf-8", newline="") as f: previous_rows = list(csv.DictReader(f))
    if len(previous_rows) != 1552: raise ValueError("Unexpected previous evaluated set")
    for row in previous_rows: membership[row["sequence_50bp"]].add("previously_evaluated_external")
    refs = sorted(membership)
    index = EditDistanceIndex(refs, protocol["max_global_edit_distance"])

    @lru_cache(maxsize=30000)
    def find(query):
        return index.find(query)

    counts, split_counts, shifts_hist, length_hist, nominal_hist = Counter(), Counter(), Counter(), Counter(), Counter()
    output_rows = []
    all_windows, clear_windows = set(), set()
    window_contexts = defaultdict(list)
    ids = set()
    for number, row in enumerate(quality):
        nominal, start, length = nominal_window(row)
        if nominal is None:
            counts["insufficient_nominal_upstream"] += 1
            continue
        if row["ID"] in ids: raise ValueError("Duplicate quality ID")
        ids.add(row["ID"])
        label = row["Txave [au]"]
        counts["positive_finite_strength_values"] += isinstance(label, (int, float)) and math.isfinite(label) and label > 0
        all_windows.add(nominal)
        window_contexts[nominal].append("".join(str(row[name] or "").upper() for name in MODULES))
        length_hist[str(length)] += 1
        nominal_hist[str(start)] += 1
        best = None
        hit_groups = set()
        valid_shifts = 0
        for shift in range(-5, 6):
            window, _, _ = nominal_window(row, shift)
            if window is None: continue
            valid_shifts += 1
            for orientation, query in (("forward", window), ("reverse_complement", reverse_complement(window))):
                for ref_idx, distance in find(query).items():
                    witness = (distance, abs(shift), shift, orientation, ref_idx)
                    if best is None or witness < best: best = witness
                    hit_groups.update(membership[refs[ref_idx]])
        shifts_hist[str(valid_shifts)] += 1
        counts["records_screened"] += 1
        course_hit = any(g.startswith("course_") for g in hit_groups)
        eval_hit = "previously_evaluated_external" in hit_groups
        counts["records_match_course"] += course_hit
        counts["records_match_previously_evaluated_external"] += eval_hit
        counts["records_match_either"] += bool(hit_groups)
        counts["records_no_match_in_rule"] += not hit_groups
        if not hit_groups: clear_windows.add(nominal)
        for group in hit_groups: split_counts[group] += 1
        output_rows.append({
            "source_id": row["ID"], "design_time_itr_start_0based": start,
            "full_sequence_length": length, "nominal_50bp": nominal, "valid_shift_count": valid_shifts,
            "match_course": course_hit, "match_previously_evaluated_external": eval_hit,
            "matching_reference_groups": ";".join(sorted(hit_groups)),
            "minimum_edits_up_to_5": "" if best is None else best[0],
            "best_shift": "" if best is None else best[2],
            "best_orientation": "" if best is None else best[3],
            "candidate_after_screen": not hit_groups,
        })
        if (number + 1) % 500 == 0: print(f"Screened {number+1}/{len(quality)}", flush=True)
    duplicate_groups = [contexts for contexts in window_contexts.values() if len(contexts) > 1]
    for row in output_rows:
        contexts = window_contexts[row["nominal_50bp"]]
        row["same_nominal_50bp_record_count"] = len(contexts)
        row["distinct_full_sequence_contexts"] = len(set(contexts))
    result = {
        "source": {"publication": "https://doi.org/10.1038/s41467-022-32829-5",
                   "supplement": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-32829-5/MediaObjects/41467_2022_32829_MOESM5_ESM.xlsx",
                   "workbook_sha256": protocol["supplement_sha256"], "sheet": protocol["workbook_sheet"],
                   "openpyxl_version": openpyxl.__version__, "edlib_version": importlib.metadata.version("edlib") if edlib is not None else None},
        "inputs": {"prior_external_scores_sha256": protocol["previous_scores_sha256"],
                   "course_data_sha256": previous["input_sha256"],
                   "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()},
        "method": {"selection": "Author-provided high quality flag Yes; no model prediction or target-score evaluation performed.",
                   "nominal_alignment": "Concatenate UP,h35,spacs,h10,disc,ITR; define design-time ITR start as nominal TSS; extract preceding 50 bp. Actual TSS and indexing convention not established for each record.",
                   "comparison": "Available nominal-window shifts +/-5, both orientations, global edit distance <=5 against all course sequences and previously evaluated external windows. Boundary shifts that lack 50 bp are skipped.",
                   "limitations": "In-vitro RNAP/sigma70 transcription measurements differ from prior cellular assays; synthetic library has shared scaffold and repeated sequences. This screen does not prove independence or alignment validity."},
        "counts": {"source_rows": len(records), "author_high_quality_rows": len(quality), **dict(counts),
                   "unique_nominal_50bp_all": len(all_windows), "unique_nominal_50bp_no_match": len(clear_windows),
                   "duplicate_50bp_groups": len(duplicate_groups),
                   "rows_in_duplicate_50bp_groups": sum(map(len, duplicate_groups)),
                   "duplicate_groups_with_different_full_sequence": sum(len(set(g)) > 1 for g in duplicate_groups),
                   "matches_by_reference_group": dict(split_counts), "reference_group_counts_can_overlap": True,
                   "available_shift_count_distribution": dict(sorted(shifts_hist.items(), key=lambda x:int(x[0]))),
                   "full_length_distribution": dict(sorted(length_hist.items(), key=lambda x:int(x[0]))),
                   "nominal_start_distribution": dict(sorted(nominal_hist.items(), key=lambda x:int(x[0])))},
        "interpretation": "Source feasibility audit only. Data quality flags and presence of positive numeric labels were inspected; no model scoring or predictive metric was computed. Do not call a fully blind independent test.",
    }
    output, rows_output = Path(args.output), Path(args.rows_output)
    if not output.is_absolute(): output = ROOT / output
    if not rows_output.is_absolute(): rows_output = ROOT / rows_output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with rows_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
