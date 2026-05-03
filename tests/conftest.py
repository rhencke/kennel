"""Shared pytest fixtures for fido tests."""

import atexit
import faulthandler
import gc
import os
import signal
import tracemalloc
from typing import TextIO

import pytest

from fido import provider

# Set FIDO_TEST_TRACEMALLOC=1 to capture per-pid allocation snapshots and
# dump the top 30 allocators at process exit.  Used to investigate the
# pytest-xdist memory leak (#1248).  Off by default — tracemalloc adds
# allocation-tracking overhead.
#
# The dump goes to ``$FIDO_TEST_TRACEMALLOC_DIR/pytrace-<pid>.log``
# (default: the repo root, which is host-mounted into the test container
# so the file survives container exit).
if os.environ.get("FIDO_TEST_TRACEMALLOC") == "1":
    tracemalloc.start(25)  # frames per traceback

    def _dump_tracemalloc_top(*_args: object) -> None:
        gc.collect()
        snapshot = tracemalloc.take_snapshot()
        all_stats = snapshot.statistics("lineno")
        top_stats = all_stats[:30]
        out_dir = os.environ.get("FIDO_TEST_TRACEMALLOC_DIR") or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"pytrace-{os.getpid()}.log")
        total = sum(s.size for s in all_stats)
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                f"=== tracemalloc top 30 (pid={os.getpid()}, "
                f"total_allocated={total / 1024 / 1024:.1f} MiB, "
                f"site_count={len(all_stats)}) ===\n"
            )
            for stat in top_stats:
                f.write(f"{stat}\n")

    atexit.register(_dump_tracemalloc_top)
    # Also dump on SIGUSR2 for live capture (SIGUSR1 is reserved for
    # faulthandler stack dumps).
    signal.signal(signal.SIGUSR2, _dump_tracemalloc_top)


def _open_faulthandler_log() -> TextIO:
    """Open a per-pid log file for SIGUSR1-driven Python stack dumps.

    Mirrors the production hook at :data:`fido.server` so any pytest worker
    can be inspected mid-hang via ``kill -SIGUSR1 <pid>`` (#1248).  The file
    is intentionally left open for the lifetime of the process — closing it
    would defeat the handler when it fires.
    """
    return open(f"/tmp/pyfh-{os.getpid()}.log", "w", encoding="utf-8", buffering=1)


# Each pytest process — controller and every xdist worker — registers its own
# SIGUSR1 handler at module import.  Importing conftest is the earliest hook
# pytest runs in a worker, well before any test collection or teardown.
_FAULTHANDLER_LOG = _open_faulthandler_log()
faulthandler.register(
    signal.SIGUSR1,
    file=_FAULTHANDLER_LOG,
    all_threads=True,
    chain=False,
)


@pytest.fixture(autouse=True)
def _reset_claude_talker_registry():
    """Clear the global :class:`~fido.provider.SessionTalker` registry between
    tests so entries from one test can't leak into the next and cause a
    spurious :class:`~fido.provider.SessionLeakError`.  Also clears any
    thread-local repo_name the test may have set via
    :func:`fido.provider.set_thread_repo`.
    """
    yield
    with provider._talkers_lock:
        provider._talkers.clear()
    provider.set_thread_repo(None)
