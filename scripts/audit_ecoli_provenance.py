#!/usr/bin/env python3
"""Verify course arrays against pinned upstream Git blob identities and text."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

UPSTREAM = {
    "promoter.npy": "14cf8fb3979f8824cf612870bae7cd46d875a347",
    "gene_expression.npy": "264a60dd2cf439b711a24413fc10383b9f8cb598",
}
REPOSITORY = "https://github.com/wangxinglong1990/Promoter_design"
COMMIT = "56c1b331bfbca0665b2c3f6b02345de794fe25df"
CAPE_COMMIT = "7a9a0e903b46813a62b234e0a062e22def6dff44"
CAPE_SEQUENCE_BLOB = "14cf8fb3979f8824cf612870bae7cd46d875a347"


def file_hashes(path: Path) -> dict:
    content = path.read_bytes()
    git_blob = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "git_blob_sha1": git_blob}


def compare_text(path: Path, sequences: np.ndarray, strengths: np.ndarray) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = [item.strip().lower() for item in next(reader)]
        if header != ["strength", "promoter"]:
            raise ValueError(f"Unexpected text columns: {header}")
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    if any(len(row) != 2 for row in rows):
        raise ValueError("Every text record must have exactly two columns")
    if len(rows) != len(sequences):
        raise ValueError("Text row count differs from array row count")
    text_strength = np.array([float(row[0].strip()) for row in rows])
    text_sequences = np.array([row[1].strip().upper() for row in rows])
    return {
        "rows": len(rows),
        "sequence_mismatches": int(np.count_nonzero(text_sequences != sequences)),
        "strength_mismatches": int(np.count_nonzero(text_strength != strengths)),
        "header": header,
    }


def audit(data_dir: Path) -> dict:
    paths = {name: data_dir / name for name in UPSTREAM}
    seqs = np.char.upper(np.load(paths["promoter.npy"], allow_pickle=False).astype(str))
    labels = np.load(paths["gene_expression.npy"], allow_pickle=False).astype(np.float64)
    if seqs.ndim != 1 or labels.ndim != 1 or len(seqs) == 0 or len(seqs) != len(labels):
        raise ValueError("Expected nonempty aligned 1D arrays")
    if not np.all(np.isfinite(labels)) or np.any(labels <= 0):
        raise ValueError("Strengths must be positive finite numbers")
    invalid = sum(len(seq) != 50 or bool(set(seq) - set("ACGT")) for seq in seqs)
    identities = {}
    for name, path in paths.items():
        checksums = file_hashes(path)
        identities[name] = {**checksums, "upstream_git_blob_sha1": UPSTREAM[name],
                            "matches_pinned_upstream": checksums["git_blob_sha1"] == UPSTREAM[name]}
    text = compare_text(data_dir / "supplementary/E_coli.txt", seqs, labels)
    return {
        "upstream_repository": REPOSITORY, "upstream_commit": COMMIT,
        "upstream_directory": "PromoS_and_PromoA_strength_prediction/PromoA_train",
        "cape_sequence_source": {
            "repository": "https://github.com/BobYHY/CAPE", "commit": CAPE_COMMIT,
            "path": "CAPE_Final/data/Ecoli_seq.npy", "git_blob_sha1": CAPE_SEQUENCE_BLOB,
            "matches_local_sequence": identities["promoter.npy"]["git_blob_sha1"] == CAPE_SEQUENCE_BLOB,
            "scope": "Sequence identity only; CAPE label file was not compared.",
        },
        "scope": "Source identity and integrity audit, not a model performance experiment.",
        "sample_count": len(seqs), "invalid_sequences": invalid,
        "strength_min": float(labels.min()), "strength_max": float(labels.max()),
        "arrays": identities, "supplementary_text_comparison": text,
        "interpretation": "Byte identity establishes a data link; biological meaning and sequence position come from cited literature, not from a hash.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/m4_ecoli_provenance_audit_2026-10-29.json"))
    args = parser.parse_args()
    result = audit(args.data_dir)
    if (result["invalid_sequences"] or not result["cape_sequence_source"]["matches_local_sequence"] or
            any(not item["matches_pinned_upstream"] for item in result["arrays"].values())):
        raise ValueError("Data do not match pinned upstream arrays or contain invalid sequences")
    text = result["supplementary_text_comparison"]
    if text["sequence_mismatches"] or text["strength_mismatches"]:
        raise ValueError("Text and arrays do not match")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
