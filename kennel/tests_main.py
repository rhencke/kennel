"""Project test runner entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def ensure_rocq_python_artifacts() -> None:
    if os.environ.get("KENNEL_ROCQ_PYTEST_ARTIFACTS") == "prepared":
        return

    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(
        [str(repo_root / "rocq-python-extraction" / "export_pytest_generated.sh")],
        check=True,
        cwd=repo_root,
    )


def main() -> int:
    import pytest

    ensure_rocq_python_artifacts()
    args = [
        "--cov",
        "--cov-report=term-missing",
        "--cov-fail-under=100",
        *sys.argv[1:],
    ]
    return pytest.main(args)
