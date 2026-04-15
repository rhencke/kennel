"""Infrastructure port protocols and real implementations.

Each port groups one low-level concern — process execution, time/sleep,
filesystem queries, or OS-level process control — so callers can accept
a single typed collaborator instead of a bag of raw stdlib callables.

Real implementations delegate directly to the stdlib with no added logic.
Tests inject fakes or mocks constructed at the call site.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------


class ProcessRunner(Protocol):
    """Runs external processes.

    Wraps :func:`subprocess.run` so callers can be tested without spawning
    real subprocesses.
    """

    def run(
        self,
        cmd: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Execute *cmd*, forwarding all keyword arguments to :func:`subprocess.run`."""
        ...


class RealProcessRunner:
    """Real :class:`ProcessRunner` that delegates to :func:`subprocess.run`."""

    def run(
        self,
        cmd: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, **kwargs)


# ---------------------------------------------------------------------------
# Clock / time
# ---------------------------------------------------------------------------


class Clock(Protocol):
    """Time and sleep operations.

    Wraps :func:`time.sleep` and :func:`time.monotonic` so callers can be
    tested without real wall-clock delays.
    """

    def sleep(self, secs: float) -> None:
        """Pause execution for *secs* seconds."""
        ...

    def monotonic(self) -> float:
        """Return a monotonic clock value in fractional seconds."""
        ...


class RealClock:
    """Real :class:`Clock` that delegates to :mod:`time`."""

    def sleep(self, secs: float) -> None:
        time.sleep(secs)

    def monotonic(self) -> float:
        return time.monotonic()


# ---------------------------------------------------------------------------
# Filesystem queries
# ---------------------------------------------------------------------------


class Filesystem(Protocol):
    """Filesystem queries.

    Wraps :func:`shutil.which` and :meth:`pathlib.Path.is_dir` so callers
    can be tested against a fake filesystem.
    """

    def which(self, name: str) -> str | None:
        """Return the full path to *name* on PATH, or ``None`` if not found."""
        ...

    def is_dir(self, path: Path) -> bool:
        """Return ``True`` if *path* is an existing directory."""
        ...


class RealFilesystem:
    """Real :class:`Filesystem` that delegates to :mod:`shutil` and :mod:`pathlib`."""

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def is_dir(self, path: Path) -> bool:
        return path.is_dir()


# ---------------------------------------------------------------------------
# OS-level process control
# ---------------------------------------------------------------------------


class OsProcess(Protocol):
    """OS-level process control.

    Wraps :func:`os.execvp`, :func:`os.chdir`, and :func:`signal.signal`
    so callers can be tested without replacing the running process or
    mutating global OS state.
    """

    def execvp(self, file: str, args: list[str]) -> None:
        """Replace the running process image — equivalent to :func:`os.execvp`."""
        ...

    def chdir(self, path: Path | str) -> None:
        """Change the process working directory — equivalent to :func:`os.chdir`."""
        ...

    def install_signal(self, signum: int, handler: Any) -> Any:
        """Install a signal handler — equivalent to :func:`signal.signal`.

        Returns the previous handler.
        """
        ...


class RealOsProcess:
    """Real :class:`OsProcess` that delegates to :mod:`os` and :mod:`signal`."""

    def execvp(self, file: str, args: list[str]) -> None:
        os.execvp(file, args)

    def chdir(self, path: Path | str) -> None:
        os.chdir(path)

    def install_signal(self, signum: int, handler: Any) -> Any:
        return signal.signal(signum, handler)


# ---------------------------------------------------------------------------
# Grouped bundle
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Infra:
    """All four infrastructure ports bundled as a single injectable collaborator.

    Callers that need all ports accept one :class:`Infra` instead of four
    separate arguments.  Tests construct an :class:`Infra` with fakes and
    inject the whole bundle at the composition root.
    """

    proc: ProcessRunner
    clock: Clock
    fs: Filesystem
    os_proc: OsProcess


def real_infra() -> Infra:
    """Construct an :class:`Infra` wired to the real stdlib implementations."""
    return Infra(
        proc=RealProcessRunner(),
        clock=RealClock(),
        fs=RealFilesystem(),
        os_proc=RealOsProcess(),
    )
