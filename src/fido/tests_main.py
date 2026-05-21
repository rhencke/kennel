"""Project test runner entrypoint."""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def ensure_rocq_python_artifacts(
    *,
    _run: Callable[..., object] = subprocess.run,
) -> None:
    if os.environ.get("FIDO_ROCQ_PYTEST_ARTIFACTS") == "prepared":
        return

    repo_root = Path(__file__).resolve().parents[2]
    _run(
        [str(repo_root / "rocq-python-extraction" / "export_pytest_generated.sh")],
        check=True,
        cwd=repo_root,
    )


def run_pytest(
    argv: list[str],
    *,
    paths: list[str],
    coverage: list[str],
    _pytest_main: Callable[[list[str]], int] | None = None,
) -> int:
    if _pytest_main is None:  # pragma: no cover
        import pytest

        _pytest_main = pytest.main

    # Cap xdist parallelism at 2 workers by default to keep total memory
    # footprint bounded under the 4 GiB cgroup cap on the test container
    # (#1248 + PR #1254).  pytest-xdist's per-worker memory grows with
    # the test heap; on free-threaded Python 3.14t this means 4 workers
    # × ~3 GiB peak = soft-locked box.  Two workers gives us parallel
    # speedup without trading the box for it.  Honor an explicit ``-n``
    # in argv (e.g. ``./fido tests -n 1`` or ``-n 4``) so callers can
    # override; only inject the default when the caller hasn't specified.
    user_argv = argv or paths
    has_n_flag = any(
        a == "-n" or a.startswith(("-n=", "--numprocesses")) for a in user_argv
    )
    xdist_args = ["-n", "2"] if not has_n_flag else []
    args = [
        *(f"--cov={source}" for source in coverage),
        "--cov-report=term-missing",
        "--cov-fail-under=100",
        *xdist_args,
        *user_argv,
    ]
    return _pytest_main(args)


def main(
    argv: list[str] | None = None,
    *,
    _ensure: Callable[[], None] | None = None,
    _pytest_main: Callable[[list[str]], int] | None = None,
) -> int:
    actual_argv = sys.argv[1:] if argv is None else argv
    ensure = _ensure if _ensure is not None else ensure_rocq_python_artifacts
    ensure()
    return run_pytest(
        actual_argv,
        paths=[],
        coverage=["fido", "rocq-python-extraction/test"],
        _pytest_main=_pytest_main,
    )


if __name__ == "__main__":
    raise SystemExit(main())
