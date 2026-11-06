#!/usr/bin/env python3
"""Score 50 bp sequences with a saved experimental Fourier ridge model."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.data import PositionKmerFeaturizer
from promoter_ml.models import RandomFourierRidge
from score_m3_screening_evaluator import load_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--output", default="outputs/m4_ranking_rff/fourier_candidate_scores.csv")
    args = parser.parse_args()
    checkpoint = Path(args.model)
    seq = load_candidates(args.candidate_file)
    model = RandomFourierRidge.load(checkpoint)
    values = model.predict(PositionKmerFeaturizer(sequence_length=50).transform(seq))
    if not np.isfinite(values).all(): raise ValueError("Nonfinite scores")
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "predicted_log10_strength"])
        writer.writerows(zip(seq.tolist(), values.tolist()))
    print(f"Saved {len(seq)} experimental scores to {output}")
    print("Checkpoint SHA-256:", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    print("Model estimates on the course label scale, not experimentally measured activity.")


if __name__ == "__main__": main()
