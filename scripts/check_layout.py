#!/usr/bin/env python3
"""Verify the W0 directory skeleton and the dependency declaration.

Read-only: this script never creates, moves or deletes anything. It answers two
questions before a merge request:

1. does the repository have the layout described in ``docs/ENVIRONMENT.md``;
2. does ``pyproject.toml`` declare the extras the later work packages expect.
"""

from __future__ import annotations

import argparse
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = (
    "configs",
    "data/raw",
    "data/processed",
    "data/splits",
    "docs",
    "experiments",
    "results/figures",
    "results/tables",
    "results/generated",
    "scripts",
    "src/promoter_ml",
    "tests",
)

REQUIRED_FILES = (
    "configs/base.toml",
    "data/raw/README.md",
    "data/processed/README.md",
    "data/splits/README.md",
    "docs/ENVIRONMENT.md",
    "results/README.md",
    "pyproject.toml",
    "README.md",
    "scripts/check_environment.py",
    "scripts/check_layout.py",
    "scripts/prepare_data.py",
    "scripts/smoke_test.py",
    "tests/conftest.py",
    "tests/test_data_contract.py",
    "tests/test_layout_and_environment.py",
)

REQUIRED_EXTRAS = {"gen": "torch", "analysis": "scikit-learn", "dev": "pytest"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPOSITORY_ROOT), help="repository root to check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    print(f"Checking layout under {root}")
    print("-" * 60)

    missing: list[str] = []
    for relative in REQUIRED_DIRECTORIES:
        path = root / relative
        if path.is_dir():
            print(f"[ ok ] directory  {relative}")
        else:
            print(f"[FAIL] directory  {relative} is missing")
            missing.append(relative)

    for relative in REQUIRED_FILES:
        path = root / relative
        if path.is_file():
            print(f"[ ok ] file       {relative}")
        else:
            print(f"[FAIL] file       {relative} is missing")
            missing.append(relative)

    print("-" * 60)
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.is_file():
        print("FAILED: pyproject.toml is missing, cannot check extras")
        return 1

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    for extra, package in REQUIRED_EXTRAS.items():
        declared = extras.get(extra, [])
        if any(package in requirement for requirement in declared):
            print(f"[ ok ] extra      [{extra}] declares {package}")
        else:
            print(f"[FAIL] extra      [{extra}] does not declare {package}")
            missing.append(f"extra:{extra}")

    print("-" * 60)
    if missing:
        print(f"W0 layout check failed: {len(missing)} item(s) missing -> {sorted(missing)}")
        return 1
    print("W0 layout check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
