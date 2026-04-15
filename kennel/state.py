"""State file and git-dir utilities shared between worker and tasks."""

from __future__ import annotations

import fcntl
import json
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class JsonFileStore(ABC):
    """Abstract base class for JSON-backed file stores.

    Provides a :meth:`modify` context manager for atomic read-modify-write
    under an exclusive ``flock``.  Subclasses must implement
    :attr:`_data_path`; they may optionally override :attr:`_lock_path`
    (defaults to the same file as the data) and :meth:`_default` (the value
    yielded when the data file is absent or empty, defaults to ``{}``).

    Usage::

        class MyStore(JsonFileStore):
            @property
            def _data_path(self) -> Path:
                return self._dir / "data.json"

        with MyStore().modify() as data:
            data["key"] = "value"
    """

    @property
    @abstractmethod
    def _data_path(self) -> Path:
        """Path to the JSON data file."""

    @property
    def _lock_path(self) -> Path:
        """Path to the flock target file.  Defaults to :attr:`_data_path`."""
        return self._data_path

    def _default(self) -> Any:
        """Value yielded when the data file is absent or empty."""
        return {}

    def _validate(self, data: Any) -> None:
        """Validate loaded data.  Raise :exc:`ValueError` if invalid.

        Called by :meth:`modify` after deserialising the JSON and before
        yielding to the caller.  The default implementation is a no-op;
        override in subclasses to add schema checks.
        """

    @contextmanager
    def modify(self) -> Generator[Any, None, None]:
        """Atomic read-modify-write: hold the exclusive flock for the entire block.

        Yields the current JSON data (or the result of :meth:`_default` when
        the file is absent or empty).  Any mutations are written back when the
        ``with`` block exits, while the exclusive lock is still held —
        preventing interleaved concurrent modifications.

        Raises :exc:`ValueError` if the file contains invalid JSON or if
        :meth:`_validate` rejects the loaded data.
        """
        lock_path = self._lock_path
        data_path = self._data_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch(exist_ok=True)
        with open(lock_path) as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            text = data_path.read_text() if data_path.exists() else ""
            if text.strip():
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as e:
                    raise ValueError(f"corrupt {data_path.name}: {e}") from e
            else:
                data = self._default()
            self._validate(data)
            yield data
            data_path.write_text(json.dumps(data))


class State(JsonFileStore):
    """Encapsulates fido state.json operations for a single worker directory.

    Abstracts all file access so callers never touch the filesystem directly.
    Instantiate with the fido_dir path and inject wherever state is needed.

    Inherits :meth:`~JsonFileStore.modify` for atomic read-modify-write.
    The lock is held on ``state.lock`` (separate from the data file) so that
    shared reads via :meth:`load` are not blocked by concurrent
    ``modify`` calls.
    """

    def __init__(self, fido_dir: Path) -> None:
        self._fido_dir = fido_dir

    @property
    def _data_path(self) -> Path:
        return self._fido_dir / "state.json"

    @property
    def _lock_path(self) -> Path:
        return self._fido_dir / "state.lock"

    @contextmanager
    def _flock(self, exclusive: bool = False) -> Generator[None, None, None]:
        """Hold a flock on ``state.lock`` for the duration of the block."""
        lock_path = self._lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch(exist_ok=True)
        with open(lock_path) as lock_fd:  # noqa: SIM115
            fcntl.flock(lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield

    def load(self) -> dict[str, Any]:
        """Return state dict, or {} when the directory or state file is absent."""
        if not self._fido_dir.exists():
            return {}
        with self._flock():
            if not self._data_path.exists():
                return {}
            return json.loads(self._data_path.read_text())

    def save(self, data: dict[str, Any]) -> None:
        """Write *data* to state.json."""
        with self._flock(exclusive=True):
            self._data_path.write_text(json.dumps(data))

    def clear(self) -> None:
        """Remove state.json."""
        with self._flock(exclusive=True):
            self._data_path.unlink(missing_ok=True)


def _resolve_git_dir(  # pyright: ignore[reportUnusedFunction]  # imported by tasks/worker
    work_dir: Path,
    *,
    _run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    """Return the absolute .git directory for *work_dir*."""
    result = _run(
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())
