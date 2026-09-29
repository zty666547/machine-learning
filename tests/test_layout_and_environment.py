"""Repository layout and toolchain-check behaviour (the W0 acceptance checks)."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

import check_environment
import check_layout


def test_repository_layout_is_complete():
    assert check_layout.main() == 0


def test_layout_check_fails_on_a_truncated_copy(tmp_path, monkeypatch):
    # A copy containing only pyproject.toml must be reported as incomplete.
    shutil.copy(check_layout.REPOSITORY_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    monkeypatch.setattr("sys.argv", ["check_layout.py", "--root", str(tmp_path)])
    assert check_layout.main() == 1


def test_module_version_parsing():
    assert check_environment.parse_version("3.13.5") == (3, 13, 5)
    assert check_environment.parse_version("2.4.0+cu124") == (2, 4, 0)
    assert check_environment.parse_version("1.26.4rc1") == (1, 26, 4)
    assert check_environment.parse_version("2.2") == (2, 2)
    assert check_environment.parse_version("unknown") == ()


def test_describe_package_reports_numpy_and_missing_package():
    assert check_environment.describe_package("numpy") is not None
    assert check_environment.describe_package("definitely_not_installed_package") is None


def test_required_and_optional_package_tables_agree():
    assert set(check_environment.REQUIRED_PACKAGES) == {"numpy", "torch"}
    assert "matplotlib" in check_environment.OPTIONAL_PACKAGES
    for module, distribution in {**check_environment.REQUIRED_PACKAGES, **check_environment.OPTIONAL_PACKAGES}.items():
        assert module and distribution


@pytest.mark.parametrize("relative", ["docs/ENVIRONMENT.md", "results/README.md", "data/raw/README.md"])
def test_documentation_files_are_tracked(relative: str):
    path: Path = check_layout.REPOSITORY_ROOT / relative
    assert path.is_file()
    assert "README" in path.name or relative.endswith("ENVIRONMENT.md")
