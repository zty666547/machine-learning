# Environment

The project supports two local setups. Day-to-day edits and light scripts run on
Windows; the CUDA training for the W7 generation models runs on Ubuntu with the
GPU. Both must be able to run the same repository, so keep commands portable
(plain `python scripts/...`) and never rely on a shell-specific path.

## Requirements

| Component | Requirement | Purpose |
| --- | --- | --- |
| Python | >= 3.10 (3.13 is used locally) | all scripts |
| numpy | >= 1.24 | data, features, baselines |
| torch | >= 2.2, CUDA build on Ubuntu | W7 conditional generators |
| scikit-learn | optional (`analysis` extra) | W3 evaluator candidates, W2 EDA |
| matplotlib | optional (`analysis` extra) | figures under `results/figures/` |

The project is installed as an editable package, so `promoter_ml` is importable
without `sys.path` tweaks in new scripts.

## Ubuntu with CUDA

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,dev]'

# CUDA build of PyTorch: pick the index URL matching the installed driver.
# cu124 and cu121 are the common choices; see https://pytorch.org/get-started/locally/
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then confirm the whole toolchain in one command:

```bash
python scripts/check_environment.py
```

A healthy Ubuntu report ends with
`Environment check passed: required packages and a CUDA device are available.`
and prints the detected GPU name and VRAM. If it prints
`cuda not available`, the wheel is a CPU build or the driver is too old for it.

Notes for this machine: the GPU is an RTX 4060 Laptop with 8 GiB VRAM. Keep the
W7 models small (single condition scalar, 50 bp sequences, batch size chosen so
peak memory stays well below the limit) and record the batch size in the config
that produced each result.

## Windows (editing, data preparation, light scripts)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[analysis,dev]"
python scripts/check_environment.py
python scripts/smoke_test.py
```

PyTorch is not required on Windows: everything except the W7 training scripts
must stay importable and runnable with numpy alone. `scripts/check_environment.py`
reports a missing PyTorch as a failure and a missing CUDA device as a warning, so
the same command is usable on both platforms.

## Verified layout

`python scripts/check_environment.py` checks the components; the repository layout
is checked mechanically with:

```bash
python scripts/check_layout.py
```

That script is read-only and fails when a directory or file listed below is
missing, or when `pyproject.toml` stops declaring the `gen`, `analysis` or `dev`
extra. Run both checks before opening a merge request.

```text
configs/            shared experiment configuration
data/raw/           untouched course files + dataset_manifest.json (tracked)
data/processed/     validated arrays written by scripts/prepare_data.py
data/splits/        small, reviewable split definitions (tracked)
docs/               environment and data contract notes
experiments/        one reproducible script per experiment (exp01_... etc.)
results/            figures, tables and manifests coming from experiments (tracked)
scripts/            command line entry points (prepare, check, train, evaluate)
src/promoter_ml/    shared library code
tests/              fast contract and layout tests (python -m pytest)
outputs/            local checkpoints and large artifacts (not tracked)
logs/               local run logs (not tracked)
```

## W0 acceptance checks

```bash
python scripts/check_layout.py        # directory skeleton + declared extras
python scripts/check_environment.py   # python, numpy, torch, CUDA, optional tools
python -m pytest                      # 17 fast tests over the data contract and layout
python scripts/smoke_test.py          # end-to-end pipeline on synthetic data
```

## Configuration policy

Experiment settings live in `configs/base.toml` and in additional TOML files, never
only in a personal command line. `configs/base.toml` keeps the shared keys; a new
experiment adds `configs/<experiment>.toml` instead of editing the shared defaults.
The configuration schema is intentionally unchanged in W0; W7 adds the model,
device and sampling sections together with the scripts that read them.
