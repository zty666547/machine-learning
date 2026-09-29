# Raw data

Place the original course files in this directory without modifying them.

## Contract

| File | Contents | Requirements |
| --- | --- | --- |
| `promoter.npy` | one-dimensional DNA sequences | exactly 50 characters, alphabet `ACGT` (upper or lower case) |
| `gene_expression.npy` | one-dimensional expression strength | strictly positive, finite, same length as the sequence array |

The arrays must be NumPy `.npy` files. Fixed-width unicode (`<U50`) is the
recommended and safest dtype. A legacy pickled object array is refused by
`scripts/prepare_data.py` unless the one-off `--allow-pickle` flag is passed, in
which case it is converted into a fixed-width array.

Check and normalize the files before any experiment:

```bash
python scripts/prepare_data.py --dry-run   # validate and print checksums
python scripts/prepare_data.py             # also write data/processed/*
```

The script writes `data/raw/dataset_manifest.json` next to the source files. That
manifest is small and reviewable: commit it together with the dataset notes so
every member trains on the identical arrays.

## Dataset notes

Record the source, licence, species, retrieval date and sha256 of each file
below before the first formal experiment:

- Source:
- Species / strain:
- Strength unit and how it was measured:
- Retrieval date:
- sha256 (`promoter.npy`):
- sha256 (`gene_expression.npy`):

Raw data files are ignored by Git; the manifest, this README and the notes are
tracked.
