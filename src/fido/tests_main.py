"""Project test runner entrypoint."""

import os
import subprocess
import sys
from pathlib import Path


def ensure_rocq_python_artifacts() -> None:
    if os.environ.get("FIDO_ROCQ_PYTEST_ARTIFACTS") == "prepared":
        return

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [str(repo_root / "rocq-python-extraction" / "export_pytest_generated.sh")],
        check=True,
        cwd=repo_root,
    )


def run_pytest(argv: list[str], *, paths: list[str], coverage: list[str]) -> int:
    import pytest

    args = [
        *(f"--cov={source}" for source in coverage),
        "--cov-report=term-missing",
        "--cov-fail-under=100",
        *(argv or paths),
    ]
    return pytest.main(args)


def main() -> int:
    ensure_rocq_python_artifacts()
    return run_pytest(
        sys.argv[1:], paths=[], coverage=["fido", "rocq-python-extraction/test"]
    )
