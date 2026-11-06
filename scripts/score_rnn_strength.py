#!/usr/bin/env python3
"""Score 50 bp DNA sequences using a saved experimental RNN checkpoint."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.models import RNNStrengthPredictor
from score_m3_screening_evaluator import load_candidates
from train_motif_cnn_predictor import one_hot_sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Saved experimental RNN .npz checkpoint.")
    parser.add_argument("--candidate-file", required=True, help="One 50 bp A/C/G/T sequence per line.")
    parser.add_argument("--output", default="outputs/m4_rnn/candidate_scores.csv")
    args = parser.parse_args()
    checkpoint, candidate_path = Path(args.model), Path(args.candidate_file)
    candidates = load_candidates(str(candidate_path))
    model = RNNStrengthPredictor.load(checkpoint)
    prediction = model.predict(one_hot_sequences(candidates))
    if not np.isfinite(prediction).all(): raise ValueError("Model produced nonfinite predictions")
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "predicted_log10_strength"])
        writer.writerows(zip(candidates.tolist(), prediction.tolist()))
    print(f"Saved {len(candidates)} experimental scores to {output}")
    print("Checkpoint SHA-256:", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    print("Prediction is a model estimate on the course label scale, not an experimental measurement.")


if __name__ == "__main__": main()
