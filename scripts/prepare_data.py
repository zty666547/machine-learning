#!/usr/bin/env python3
"""Validate the raw course arrays, record checksums and write processed copies.

This script is the first step of the shared data contract. It never modifies
anything under ``data/raw``; the processed output is written to ``data/processed``
(ignored by Git) and the manifest is written next to the raw data so it can be
reviewed and committed.

Raw arrays are expected to be NumPy ``.npy`` files. Fixed-width unicode arrays
(``<U50``) load with ``allow_pickle=False`` and are the recommended format.
Legacy object arrays saved with pickle are refused by default; pass
``--allow-pickle`` once to convert such a file into the safe fixed-width format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

DNA_ALPHABET = "ACGT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", help="directory holding the course .npy files")
    parser.add_argument("--processed-dir", default="data/processed", help="where validated arrays are written")
    parser.add_argument(
        "--sequence-file",
        default="promoter.npy",
        help="file name of the one-dimensional DNA sequence array",
    )
    parser.add_argument(
        "--label-file",
        default="gene_expression.npy",
        help="file name of the one-dimensional positive strength array",
    )
    parser.add_argument("--sequence-length", type=int, default=50, help="required sequence length in bp")
    parser.add_argument("--allow-pickle", action="store_true", help="permit loading a legacy pickled object array")
    parser.add_argument("--dry-run", action="store_true", help="validate and checksum only; write nothing")
    return parser.parse_args()


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_array(path: Path, allow_pickle: bool) -> np.ndarray:
    """Load a .npy array, converting legacy object arrays into fixed-width text."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    try:
        return np.load(path, allow_pickle=False)
    except ValueError as error:
        if not allow_pickle:
            raise ValueError(
                f"{path.name} is a pickled object array and cannot be loaded safely. "
                "Re-run with --allow-pickle to convert it into a fixed-width unicode array."
            ) from error
        legacy = np.load(path, allow_pickle=True)
        if legacy.dtype != object:
            raise
        return np.array([str(item) for item in legacy], dtype=f"<U{max(len(str(item)) for item in legacy)}")


def validate_sequences(sequences: np.ndarray, sequence_length: int) -> np.ndarray:
    if sequences.ndim != 1:
        raise ValueError(f"Sequences must be one-dimensional, found shape {sequences.shape}")
    if sequences.dtype.kind in {"S", "O"}:
        sequences = np.array([item.decode() if isinstance(item, bytes) else str(item) for item in sequences])
    sequences = np.char.upper(sequences.astype(str))

    invalid = [
        (index, sequence)
        for index, sequence in enumerate(sequences)
        if len(sequence) != sequence_length or set(sequence) - set(DNA_ALPHABET)
    ]
    if invalid:
        index, sequence = invalid[0]
        raise ValueError(
            f"{len(invalid)} invalid sequence(s); first at index {index}: {sequence!r} "
            f"(expected {sequence_length} characters from {DNA_ALPHABET})"
        )
    return sequences


def validate_strengths(strengths: np.ndarray) -> np.ndarray:
    if strengths.ndim != 1:
        raise ValueError(f"Strengths must be one-dimensional, found shape {strengths.shape}")
    strengths = strengths.astype(np.float64)
    if not np.all(np.isfinite(strengths)):
        raise ValueError("Strength labels contain non-finite values")
    if np.any(strengths <= 0):
        raise ValueError("Strength labels must be strictly positive so that log10 is defined")
    return strengths


def describe_array(array: np.ndarray) -> dict[str, object]:
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def main() -> int:
    args = parse_args()
    raw_dir = REPOSITORY_ROOT / args.raw_dir
    processed_dir = REPOSITORY_ROOT / args.processed_dir
    sequence_path = raw_dir / args.sequence_file
    label_path = raw_dir / args.label_file

    print(f"Raw directory:       {raw_dir}")
    print(f"Sequence file:       {args.sequence_file}")
    print(f"Label file:          {args.label_file}")
    print(f"Required length:     {args.sequence_length} bp")
    print(f"Pickled input:       {'allowed' if args.allow_pickle else 'refused'}")
    print("-" * 60)

    if not raw_dir.is_dir():
        print(f"FAILED: raw directory does not exist: {raw_dir}")
        print("Place the course arrays in data/raw and record their source in data/raw/README.md.")
        return 1

    for path in (sequence_path, label_path):
        if not path.is_file():
            print(f"FAILED: missing required file: {path}")
            print("Expected promoter.npy and gene_expression.npy; see data/raw/README.md for the contract.")
            return 1

    try:
        sequences = validate_sequences(load_array(sequence_path, args.allow_pickle), args.sequence_length)
        strengths = validate_strengths(load_array(label_path, args.allow_pickle))
    except (ValueError, FileNotFoundError) as error:
        print(f"FAILED: {error}")
        return 1

    if len(sequences) != len(strengths):
        print(f"FAILED: sequence count {len(sequences)} != label count {len(strengths)}")
        return 1
    if len(sequences) == 0:
        print("FAILED: the dataset is empty")
        return 1

    manifest = {
        "dataset": {
            "sample_count": int(len(sequences)),
            "sequence_length": int(args.sequence_length),
            "alphabet": DNA_ALPHABET,
            "label_transform": "log10",
            "strength_min": float(strengths.min()),
            "strength_max": float(strengths.max()),
            "strength_mean": float(strengths.mean()),
            "mean_gc_fraction": float(np.mean([(s.count("G") + s.count("C")) / len(s) for s in sequences])),
            "duplicate_sequences": int(len(sequences) - len(set(sequences.tolist()))),
        },
        "source_files": {
            args.sequence_file: {
                "sha256": sha256_of(sequence_path),
                "bytes": sequence_path.stat().st_size,
                **describe_array(sequences),
            },
            args.label_file: {
                "sha256": sha256_of(label_path),
                "bytes": label_path.stat().st_size,
                **describe_array(strengths),
            },
        },
    }

    print(f"Sequences:           {len(sequences)} x {args.sequence_length} bp, dtype {sequences.dtype}")
    print(f"Duplicate sequences: {manifest['dataset']['duplicate_sequences']}")
    print(
        "Strength (raw):      "
        f"min={manifest['dataset']['strength_min']:.6g} "
        f"max={manifest['dataset']['strength_max']:.6g} "
        f"mean={manifest['dataset']['strength_mean']:.6g}"
    )
    print(f"Mean GC fraction:    {manifest['dataset']['mean_gc_fraction']:.4f}")
    for name, info in manifest["source_files"].items():
        print(f"sha256 {name}: {info['sha256']}")

    if args.dry_run:
        print("-" * 60)
        print("Dry run: validation passed, no files were written.")
        return 0

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / args.sequence_file, sequences)
    np.save(processed_dir / args.label_file, strengths)
    manifest_path = raw_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("-" * 60)
    print(f"Wrote processed arrays to {processed_dir}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
