# Formal data splits

This directory stores small, reviewable index files shared by every model.

Generate the E. coli split with:

```bash
python scripts/prepare_similarity_split.py \
  --data-dir /path/to/promoter/strenth/data \
  --selected-threshold 0.90
```

The script compares 80%, 85% and 90% aligned Hamming-identity thresholds,
clusters connected similar sequences, and assigns whole clusters to the
train/validation/test sets. The selected split, its threshold comparison, and
training-only label statistics are versioned here after review.
