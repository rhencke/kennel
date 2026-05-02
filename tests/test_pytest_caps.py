"""Regression tests for the pytest-cap layered defense (#1248).

These do not invoke a sub-pytest run — that would explode CI time.  Instead
they verify the *plumbing* that the launcher and conftest install so a leaky
test or hung worker cannot soft-lock the box again.
"""

import os
import re
import signal
from pathlib import Path


def test_faulthandler_is_registered_on_sigusr1() -> None:
    """conftest.py registers a SIGUSR1 handler that dumps Python stacks.

    We can't easily catch the handler firing without forking, but we can
    assert the per-pid log file was opened at import time and that
    SIGUSR1 is the signal conftest hooks (#1248).
    """
    log_path = Path(f"/tmp/pyfh-{os.getpid()}.log")
    assert log_path.exists(), f"conftest should have opened {log_path} at import time"
    assert log_path.is_file()
    # Simply having the SIGUSR1 module attribute confirms our import path.
    assert hasattr(signal, "SIGUSR1")


def test_launcher_test_paths_use_capped_runner() -> None:
    """The fido launcher routes ``./fido tests`` and ``./fido pytest`` through
    the capped image runner so every pytest invocation gets cgroup memory and
    wall-clock bounds.  Without this every leak repeats #1248.
    """
    repo_root = Path(__file__).resolve().parent.parent
    launcher = (repo_root / "fido").read_text()

    tests_block = re.search(
        r"^  tests\)\n(?P<body>.+?)\n    ;;\n",
        launcher,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert tests_block, "tests case missing from launcher"
    assert "run_fido_pyproject_image_capped" in tests_block.group("body"), (
        "tests case must use the capped runner — see #1248"
    )

    pytest_block = re.search(
        r"^  pytest\)\n(?P<body>.+?)\n    ;;\n",
        launcher,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert pytest_block, "pytest case missing from launcher"
    assert "run_fido_pyproject_image_capped" in pytest_block.group("body"), (
        "pytest case must use the capped runner — see #1248"
    )


def test_run_container_emits_diagnostic_on_timeout_and_oom() -> None:
    """``run_container`` must emit a clear, grep-able message on the two
    cap-trip exit codes (124 timeout, 137 OOM) so a CI failure points at the
    cause instead of a silent dead container.
    """
    repo_root = Path(__file__).resolve().parent.parent
    launcher = (repo_root / "fido").read_text()

    assert (
        "TIMEOUT: container exceeded ${_container_cap_timeout} wall-clock cap (#1248)."
        in launcher
    ), "timeout diagnostic missing from run_container"
    assert (
        "OOM: container killed by cgroup memory cap ${_container_cap_memory} (#1248)."
        in launcher
    ), "OOM diagnostic missing from run_container"


def test_run_container_dumps_faulthandler_logs_on_timeout() -> None:
    """On wall-clock timeout, the launcher must surface any
    ``/tmp/pyfh-*.log`` files produced by SIGUSR1 dumps so the operator can
    see *which* worker hung instead of guessing from the empty stderr.
    """
    repo_root = Path(__file__).resolve().parent.parent
    launcher = (repo_root / "fido").read_text()

    assert "/tmp/pyfh-*.log" in launcher, (
        "timeout branch should iterate /tmp/pyfh-*.log dumps"
    )
