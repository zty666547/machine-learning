# Results

Every experiment writes its outputs here so that tables and figures can be
reviewed without rerunning training.

```text
results/figures/     one figure per experiment, named exp<NN>_<topic>.png
results/tables/      one file per experiment, named exp<NN>_<topic>.csv
results/generated/   FASTA files with generation headers (model/condition/seed/mode)
results/<name>_metrics.json   one summary json per run, with the config used
```

Rules:

- a result file is committed only together with the script and config that produced it;
- every json/csv records the seed, the split version and the config path;
- training checkpoints and raw model weights stay in `outputs/` and out of Git;
- generated FASTA files are published through the final `best_candidates.fasta`,
  while bulk candidate sets stay local.

An empty directory carries a `.gitkeep` so the layout is visible in a fresh clone.
