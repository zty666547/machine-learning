#!/usr/bin/env python3
"""Report the local toolchain so every member can compare environments.

The check is intentionally read-only and never installs anything. It prints one
line per component and exits non-zero only when a required component is missing.

Required: Python >= 3.10, numpy, torch.
Optional: scikit-learn, matplotlib, biopython, and a CUDA device for training.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from typing import Iterable

# Module name -> distribution name, so versions are read from metadata without
# importing the package. Importing matplotlib or torch would write caches and
# make this read-only check slow and side-effecting.
REQUIRED_PACKAGES = {"numpy": "numpy", "torch": "torch"}
OPTIONAL_PACKAGES = {"sklearn": "scikit-learn", "matplotlib": "matplotlib", "Bio": "biopython"}
MINIMUM_PYTHON = (3, 10)

INSTALL_HINTS = {
    "torch": [
        "No PyTorch found. Install the CUDA build that matches the driver, e.g.",
        "  python -m pip install torch --index-url https://download.pytorch.org/whl/cu124",
        "See docs/ENVIRONMENT.md for the CPU fallback and for the Ubuntu setup.",
    ],
    "numpy": ["Install numpy with: python -m pip install 'numpy>=1.24'"],
}


def parse_version(text: str) -> tuple[int, ...]:
    """Return the leading numeric release so versions compare reliably.

    Suffixes such as ``+cu124`` (local version label) or ``rc1`` (pre-release) are
    ignored: ``"2.4.0+cu124"`` yields ``(2, 4, 0)``.
    """
    release = text.split("+", 1)[0]
    numbers: list[int] = []
    for part in release.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


def describe_package(module_name: str) -> str | None:
    """Report a version without importing: None when the package is absent."""
    if importlib.util.find_spec(module_name) is None:
        return None
    distribution = REQUIRED_PACKAGES.get(module_name) or OPTIONAL_PACKAGES.get(module_name) or module_name
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "installed (version unknown)"


def report_required(names: Iterable[str], missing: list[str]) -> None:
    for name in names:
        version = describe_package(name)
        if version is None:
            print(f"[FAIL] {name:<13} not installed")
            missing.append(name)
        else:
            print(f"[ ok ] {name:<13} {version}")


def report_optional(names: Iterable[str]) -> None:
    for name in names:
        version = describe_package(name)
        status = version if version is not None else "not installed (optional)"
        print(f"[ -- ] {name:<13} {status}")


def report_torch_runtime(required: Iterable[str]) -> bool:
    """Report the torch build and CUDA availability; True when a GPU is usable."""
    if "torch" in required:
        print("[ -- ] torch runtime   skipped: PyTorch is not installed")
        return False

    # PyTorch must be imported for CUDA queries; this happens only when present.
    import torch

    cuda_version = getattr(torch.version, "cuda", None)
    print(f"[ -- ] torch build     {torch.__version__} (cuda={cuda_version or 'cpu-only'})")
    if not torch.cuda.is_available():
        print("[ -- ] cuda            not available; CPU training only")
        print("       W7 training expects a GPU; confirm the driver and the CUDA wheel.")
        return False

    device_count = torch.cuda.device_count()
    print(f"[ ok ] cuda            available, {device_count} device(s)")
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / 1024**3
        print(
            f"[ ok ] gpu[{index}]         {properties.name}, "
            f"compute capability {properties.major}.{properties.minor}, "
            f"{memory_gib:.1f} GiB"
        )
    return True


def report_machine() -> None:
    print(f"[ -- ] platform        {platform.platform()}")
    print(f"[ -- ] python          {platform.python_version()} ({sys.executable})")
    print(f"[ -- ] cpu threads     {os.cpu_count()}")
    if platform.system() == "Linux":
        kernel = f"{platform.system()} {platform.release()}"
        print(f"[ -- ] kernel          {kernel}")


def main() -> int:
    print("PR03-01 environment check")
    print("-" * 60)
    report_machine()

    if sys.version_info < MINIMUM_PYTHON:
        print(f"[FAIL] python          requires >= {'.'.join(map(str, MINIMUM_PYTHON))}")
        return 1
    print("[ ok ] python version  meets the >= 3.10 requirement")

    missing: list[str] = []
    report_required(REQUIRED_PACKAGES, missing)
    report_optional(OPTIONAL_PACKAGES)

    has_gpu = report_torch_runtime(missing)

    print("-" * 60)
    if missing:
        for name in missing:
            for line in INSTALL_HINTS.get(name, [f"Install {name} before running the pipeline."]):
                print(line)
        print("Environment check failed: required components are missing.")
        return 1

    if not has_gpu:
        print("Environment check passed with warnings: required packages found, no usable CUDA device.")
        return 0

    print("Environment check passed: required packages and a CUDA device are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
