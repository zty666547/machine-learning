#!/usr/bin/env python3
"""Export averaged positional base and top k-mer profiles as a CSV."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "kind", "token", "start_1based", "assumed_tss_relative_start",
                         "training_frequency", "autoregressive_mean", "vae_mean"])
        for group, info in summary["groups"].items():
            base = info["position_base_profiles"]
            for index, letter in enumerate(base["alphabet_order"]):
                for p, ref in enumerate(base["reference"]):
                    writer.writerow([group, "base", letter, p + 1, p - 50,
                                     ref[index], base["autoregressive_mean"][p][index], base["vae_mean"][p][index]])
            for k, tokens in info["top_training_motif_profiles"].items():
                for token, profile in tokens.items():
                    for p, ref in enumerate(profile["reference_start_frequency"]):
                        writer.writerow([group, f"{k}-mer", token, p + 1, p - 50, ref,
                                         profile["autoregressive_start_frequency_mean"][p],
                                         profile["vae_start_frequency_mean"][p]])
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
