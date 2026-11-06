#!/usr/bin/env python3
"""Describe ambiguity in LaFleur TSS table positions without treating either convention as confirmed."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_lafleur_source import nominal_window


def classify_peak(position: int, nominal: int, length: int):
    """Compare interpretations, not a declaration of actual TSS coordinates."""
    result = {}
    for name, index in (("zero_based", position), ("one_based", position - 1)):
        result[name] = {"hypothetical_index": index,
                        "within_sequence": 0 <= index < length,
                        "can_extract_preceding_50bp": 50 <= index < length,
                        "difference_from_nominal_itr_start": index - nominal}
    return result


def peak_summary(names, counts):
    values = []
    for name, raw in zip(names, counts):
        if not isinstance(name, str) or not name.startswith("TSS_pos_"):
            raise ValueError("Unexpected TSS column name")
        p = int(name.rsplit("_", 1)[1])
        if raw is None: raw = 0
        if not isinstance(raw, (int, float)) or not np.isfinite(raw) or raw < 0:
            raise ValueError("TSS read counts must be finite and nonnegative")
        values.append((p, float(raw)))
    best = max(value for _, value in values)
    locations = [p for p, value in values if value == best]
    return best, sorted(locations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--prior-rows", required=True)
    parser.add_argument("--output", default="reports/m4_lafleur_tss_diagnostic_2026-11-06.json")
    parser.add_argument("--rows-output", default="outputs/m4_external/lafleur_tss_diagnostic.csv")
    args = parser.parse_args()
    source, prior = Path(args.workbook), Path(args.prior_rows)
    expected = json.loads((ROOT / "reports/m4_lafleur_source_audit_2026-11-05.json").read_text(encoding="utf-8"))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != expected["source"]["workbook_sha256"]:
        raise ValueError("Workbook checksum differs from source audit")
    with prior.open(encoding="utf-8", newline="") as f:
        prior_rows = list(csv.DictReader(f))
    previous = {str(r["source_id"]): r for r in prior_rows}
    if len(previous) != len(prior_rows) or len(previous) != expected["counts"]["author_high_quality_rows"]:
        raise ValueError("Prior IDs or count invalid")
    workbook = openpyxl.load_workbook(source, read_only=True, data_only=True)
    sheet = workbook["La Fleur et al (Fig 1d, S1)"]
    iterator = iter(sheet.iter_rows(values_only=True))
    header = next(iterator)
    if len(set(header)) != len(header): raise ValueError("Duplicate source headers")
    reference = {}
    for row in iterator:
        if row[0] is None: continue
        record = dict(zip(header, row))
        if record["high quality"] != "Yes": continue
        key = str(record["ID"])
        if key in reference or key not in previous: raise ValueError("Prior/source ID mismatch")
        window, nominal, length = nominal_window(record)
        if (window != previous[key]["nominal_50bp"] or nominal != int(previous[key]["design_time_itr_start_0based"])
                or length != int(previous[key]["full_sequence_length"])):
            raise ValueError("Prior window differs from original modules")
        reference[key] = (nominal, length)
    if set(reference) != set(previous): raise ValueError("Missing high-quality source IDs")
    sheet = workbook["TSS Data (Fig 1c, S2)"]
    iterator = iter(sheet.iter_rows(values_only=True))
    headers = next(iterator)
    if headers[0] != "ID" or len(set(headers)) != len(headers):
        raise ValueError("Unexpected TSS table headers")
    positions = [int(s.rsplit("_", 1)[1]) for s in headers[1:]]
    if any(not isinstance(s, str) or not s.startswith("TSS_pos_") for s in headers[1:]):
        raise ValueError("Unexpected TSS column")
    seen, counts, diffs = set(), Counter(), {"zero_based": Counter(), "one_based": Counter()}
    output = []
    for raw in iterator:
        key = str(raw[0])
        if raw[0] is None or key not in reference: continue
        if key in seen: raise ValueError("Duplicate TSS ID")
        seen.add(key)
        nominal, length = reference[key]
        highest, peaks = peak_summary(headers[1:], raw[1:])
        counts["matched_high_quality_ids"] += 1
        counts["zero_maximum"] += highest == 0
        counts["tied_maximum"] += len(peaks) > 1
        row = {"source_id": key, "full_sequence_length": length, "nominal_itr_start_0based": nominal,
               "raw_maximum_count": highest, "peak_position_count": len(peaks),
               "raw_peak_positions": ";".join(map(str, peaks))}
        if highest > 0 and len(peaks) == 1:
            counts["unique_nonzero_maximum"] += 1
            info = classify_peak(peaks[0], nominal, length)
            for name, data in info.items():
                diffs[name][str(data["difference_from_nominal_itr_start"])] += 1
                counts[f"{name}_within_sequence"] += data["within_sequence"]
                counts[f"{name}_can_extract_50bp"] += data["can_extract_preceding_50bp"]
                counts[f"{name}_within_1bp_nominal"] += abs(data["difference_from_nominal_itr_start"]) <= 1
                counts[f"{name}_within_5bp_nominal"] += abs(data["difference_from_nominal_itr_start"]) <= 5
                for k, v in data.items(): row[f"{name}_{k}"] = v
        else:
            for name in ("zero_based", "one_based"):
                for k in ("hypothetical_index", "within_sequence", "can_extract_preceding_50bp", "difference_from_nominal_itr_start"):
                    row[f"{name}_{k}"] = ""
        output.append(row)
    if seen != set(reference): raise ValueError("Missing TSS rows for high-quality IDs")
    result = {
        "source": {"workbook_sha256": source_sha, "prior_rows_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
                   "tss_sheet": "TSS Data (Fig 1c, S2)", "data_sheet": "La Fleur et al (Fig 1d, S1)",
                   "openpyxl_version": openpyxl.__version__},
        "rule": {"population": "Author high quality Yes records; no model or activity label evaluation",
                 "peak": "Single largest nonzero numeric count across all TSS_pos columns. Rows with ties or zero maximum are reported but not assigned a position.",
                 "hypotheses": "Interpret raw number as zero-based or as one-based by subtracting one. Neither is asserted correct.",
                 "caveat": "Raw maximum position might use a longer template, hidden flanking context, or other conventions. High-quality flag uses the author's methodology; this diagnostic does not reproduce it."},
        "tss_columns": {"count": len(positions), "min_raw_position": min(positions), "max_raw_position": max(positions)},
        "counts": dict(counts),
        "difference_from_nominal_counts": {k: dict(sorted(v.items(), key=lambda x:int(x[0]))) for k, v in diffs.items()},
        "conclusion": "Coordinate origin and template are unverified. Neither naive indexing hypothesis is validated by this summary; no experimentally aligned windows should be claimed without source documentation or validation.",
    }
    target, rows_target = Path(args.output), Path(args.rows_output)
    if not target.is_absolute(): target = ROOT / target
    if not rows_target.is_absolute(): rows_target = ROOT / rows_target
    target.parent.mkdir(parents=True, exist_ok=True)
    rows_target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with rows_target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
