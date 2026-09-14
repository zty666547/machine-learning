# Data splits

Small, reviewable definitions of the train/validation/test split shared by the
generator and the independent evaluator.

Planned content (W1):

- `similarity_v1.json` — cluster-based split: parameter block (k-mer size,
  similarity threshold, linkage, seed), the cluster assignment, and the index
  lists for train / validation / test.

Rules:

- a split file is committed and never edited in place; a change means a new version;
- the generator and the independent strength evaluator must use the same version;
- the development random split in `promoter_ml.data.split_indices` is a
  placeholder only and must not appear in reported results.

Use the `split_strategy` field in the configuration to select the active version;
`random_dev` is the current placeholder value in `configs/base.toml`.
