"""AtomicReference[T] — single-lock pointer-swap primitive for lock-free reads.

Writers hold ``_lock`` for the duration of the pointer swap so concurrent
writes are serialized and no update is lost.  Readers call :meth:`get`
without acquiring any lock — a single object-reference load is atomic on all
CPython-supported architectures, including the free-threaded (no-GIL) build.

``T`` should be an immutable value (frozen dataclass, namedtuple, ``None``,
etc.) so that readers who hold a reference to the snapshot can inspect all
fields without encountering races inside the value itself.
"""

import threading
from collections.abc import Callable
from typing import Generic, TypeVar

_T = TypeVar("_T")


class AtomicReference(Generic[_T]):
    """A reference cell whose pointer swap is serialized by a single lock.

    Readers are observationally lock-free — :meth:`get` is a bare attribute
    read with no lock acquire.  Writers hold the internal ``_lock`` only for
    the atomic swap, keeping write-side latency minimal.

    The typical usage pattern is *update with retry*::

        ref = AtomicReference(initial_snapshot)

        def bump(old: Snapshot) -> Snapshot:
            return replace(old, counter=old.counter + 1)

        ref.update(bump)  # retries automatically if a concurrent writer raced

    ``fn`` passed to :meth:`update` must be a **pure function** — the retry
    loop may call it more than once under write contention.
    """

    def __init__(self, initial: _T) -> None:
        self._value: _T = initial
        self._lock = threading.Lock()

    def get(self) -> _T:
        """Return the current value.  Lock-free; safe to call from any thread."""
        return self._value

    def set(self, value: _T) -> None:
        """Unconditionally replace the stored reference.

        Serialized by the internal lock so concurrent :meth:`set` and
        :meth:`compare_and_set` calls never interleave.
        """
        with self._lock:
            self._value = value

    def compare_and_set(self, expected: _T, new_value: _T) -> bool:
        """Swap to *new_value* only if the current reference **is** *expected*.

        Uses identity (``is``) rather than equality (``==``) for the check —
        the comparison is always O(1) and unambiguous regardless of how ``_T``
        defines ``__eq__``.

        Returns ``True`` on success, ``False`` when the current value is not
        *expected* (indicating a concurrent write raced ahead).
        """
        with self._lock:
            if self._value is not expected:
                return False
            self._value = new_value
            return True

    def update(self, fn: Callable[[_T], _T]) -> _T:
        """Apply *fn* to the current value and install the result atomically.

        Reads the current reference, calls ``fn(current)`` to produce the
        next value, then attempts a compare-and-set.  If a concurrent writer
        changed the reference first, the loop retries with the new current.

        Returns the value that was successfully installed.

        ``fn`` must be a pure function — it may be called more than once
        under write contention.
        """
        while True:
            old = self.get()
            new = fn(old)
            if self.compare_and_set(old, new):
                return new
