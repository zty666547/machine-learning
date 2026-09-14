"""Shared pytest fixtures: small synthetic arrays written into a temporary directory."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def make_sequences(sample_count: int = 40, sequence_length: int = 50, seed: int = 5) -> np.ndarray:
    """Return random but valid upper-case DNA sequences."""
    rng = np.random.default_rng(seed)
    return np.array(
        ["".join(rng.choice(list("ACGT"), size=sequence_length)) for _ in range(sample_count)],
        dtype=f"<U{sequence_length}",
    )


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """A valid raw data directory with matching sequences and positive strengths."""
    sequences = make_sequences()
    strengths = np.linspace(0.1, 25.0, len(sequences))
    np.save(tmp_path / "promoter.npy", sequences)
    np.save(tmp_path / "gene_expression.npy", strengths)
    return tmp_path, sequences, strengths
