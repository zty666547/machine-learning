"""Data contract tests: loading, validation and the prepare_data entry point."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_sequences
from promoter_ml.data import KmerFeaturizer, load_promoter_arrays
from prepare_data import load_array, sha256_of, validate_sequences, validate_strengths


def test_load_promoter_arrays_accepts_valid_arrays(synthetic_raw_dir):
    raw_dir, sequences, strengths = synthetic_raw_dir
    loaded_sequences, loaded_strengths = load_promoter_arrays(raw_dir)
    assert loaded_sequences.tolist() == sequences.tolist()
    assert np.allclose(loaded_strengths, strengths)


def test_load_promoter_arrays_rejects_invalid_base(tmp_path):
    sequences = make_sequences(sample_count=3)
    sequences[1] = "N" + sequences[1][1:]
    np.save(tmp_path / "promoter.npy", sequences)
    np.save(tmp_path / "gene_expression.npy", np.ones(3))
    with pytest.raises(ValueError, match="invalid sequences"):
        load_promoter_arrays(tmp_path)


def test_load_promoter_arrays_rejects_non_positive_strength(tmp_path):
    sequences = make_sequences(sample_count=3)
    np.save(tmp_path / "promoter.npy", sequences)
    np.save(tmp_path / "gene_expression.npy", np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="finite positive"):
        load_promoter_arrays(tmp_path)


def test_validate_sequences_upper_cases_fixed_width_input():
    mixed = np.array(["acgtacgtac", "ACGTACGTAC"])
    validated = validate_sequences(mixed, sequence_length=10)
    assert validated.tolist() == ["ACGTACGTAC", "ACGTACGTAC"]


def test_validate_sequences_reports_first_offender():
    sequences = np.array(["ACGT", "ACGN"])
    with pytest.raises(ValueError, match="index 1"):
        validate_sequences(sequences, sequence_length=4)


def test_validate_strengths_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        validate_strengths(np.array([1.0, np.nan]))


def test_load_array_refuses_pickled_object_array(tmp_path):
    legacy_path = tmp_path / "promoter.npy"
    np.save(legacy_path, np.array(["ACGT"], dtype=object))
    with pytest.raises(ValueError, match="allow-pickle"):
        load_array(legacy_path, allow_pickle=False)
    converted = load_array(legacy_path, allow_pickle=True)
    assert converted.dtype.kind == "U"
    assert converted.tolist() == ["ACGT"]


def test_sha256_of_is_stable(tmp_path):
    target = tmp_path / "payload.bin"
    target.write_bytes(b"promoter")
    assert sha256_of(target) == sha256_of(target)
    assert len(sha256_of(target)) == 64


def test_featurizer_row_sums_are_one_per_kmer_order():
    sequences = make_sequences(sample_count=2)
    features = KmerFeaturizer(k_min=1, k_max=2, include_gc=True).transform(sequences)
    # 4 + 16 k-mer columns (each normalized to sum to 1) plus one GC column
    assert features.shape == (2, 4 + 16 + 1)
    for row in features[:, :-1]:
        assert np.isclose(row[:4].sum(), 1.0)
        assert np.isclose(row[4:].sum(), 1.0)
