#!/usr/bin/env python3
"""Import course Bacillus subtilis data as separate, auditable datasets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path, fields: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = fields.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence(value: str, row: int) -> str:
    value = value.strip().upper()
    if not value or set(value) - set("ACGT"):
        raise ValueError(f"row {row}: empty sequence or non-ACGT character")
    return value


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def positive_number(value: str, row: int, field: str) -> str:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row}: invalid {field}: {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"row {row}: {field} must be finite and positive")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="course promoter directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/bacillus_subtilis"))
    args = parser.parse_args()
    classification_file = args.data_root / "reg_and_gen/Datasets/Bacillus subtilis/Dataset.csv"
    strength_file = args.data_root / "strenth/data/kucao_test.csv"
    classification = read_csv(classification_file, {"seq_id", "seq_type", "seq", "label"})
    strength = read_csv(strength_file, {"调控基因名称", "一致化后强度", "启动子序列", "荧光表达强度"})

    binary_rows = []
    for index, row in enumerate(classification, 2):
        label = row["label"].strip()
        if label not in {"0", "1"}:
            raise ValueError(f"row {index}: unexpected classification label {label!r}")
        binary_rows.append({"source_row": str(index), "source_id": row["seq_id"], "source_seq_type": row["seq_type"],
                            "sequence": sequence(row["seq"], index), "label": label})

    strength_rows = []
    for index, row in enumerate(strength, 2):
        strength_rows.append({
            "source_row": str(index), "gene_name": row["调控基因名称"],
            "sequence": sequence(row["启动子序列"], index),
            "normalized_strength": positive_number(row["一致化后强度"].strip(), index, "normalized_strength"),
            "fluorescence_strength": positive_number(row["荧光表达强度"].strip(), index, "fluorescence_strength"),
        })

    duplicates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in strength_rows:
        duplicates[row["sequence"]].append(row)
    repeated = [group for group in duplicates.values() if len(group) > 1]
    conflicting = [group for group in repeated if len({float(r["normalized_strength"]) for r in group}) > 1]

    def summary(rows: list[dict[str, str]]) -> dict:
        sequences = [row["sequence"] for row in rows]
        return {"rows": len(rows), "unique_sequences": len(set(sequences)),
                "sequence_lengths": dict(sorted(Counter(map(len, sequences)).items()))}

    report = {
        "species": "Bacillus subtilis",
        "classification": {**summary(binary_rows), "source": str(classification_file.relative_to(args.data_root)),
                           "sha256": digest(classification_file), "labels": dict(Counter(r["label"] for r in binary_rows))},
        "strength": {**summary(strength_rows), "source": str(strength_file.relative_to(args.data_root)),
                     "sha256": digest(strength_file), "duplicate_sequence_groups": len(repeated),
                     "conflicting_normalized_strength_groups": len(conflicting),
                     "conflicting_source_rows": [[r["source_row"] for r in group] for group in conflicting]},
        "policy": "Separate species and tasks; preserve full sequences and both original strength fields. "
                  "No cropping, deduplication, data split or model training. Original strength file is a test resource.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "classification.csv", ["source_row", "source_id", "source_seq_type", "sequence", "label"], binary_rows)
    write_csv(args.output_dir / "strength.csv", ["source_row", "gene_name", "sequence", "normalized_strength", "fluorescence_strength"], strength_rows)
    (args.output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
