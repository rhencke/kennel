"""AtomicReference[T] — single-lock pointer-swap primitive for lock-free reads.

Writers hold ``_lock`` for the duration of the pointer swap so concurrent
writes are serialized and no update is lost.  Readers call :meth:`get`
without acquiring any lock — a single object-reference load is atomic on all
CPython-supported architectures, including the free-threaded (no-GIL) build.

``T`` should be an immutable value (frozen dataclass, namedtuple, ``None``,
etc.) so that readers who hold a reference to the snapshot can inspect all
fields without encountering races inside the value itself.

:class:`AtomicReader` and :class:`AtomicUpdater` are the two narrow
Protocol faces of :class:`AtomicReference`.  Collaborators should accept
one of those, not the concrete class — a callee that holds both is usually
doing something that belongs inside the owner (code smell).
"""

import threading
from collections.abc import Callable
from typing import Generic, Protocol, TypeVar

from fido.lens import Lens

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_upd = TypeVar("_T_upd")


class AtomicReader(Protocol[_T_co]):
    """Read-only face of an atomic reference cell.

    Pass this to collaborators that only need to observe the latest snapshot.
    A collaborator that also calls :meth:`~AtomicUpdater.update` is likely
    crossing an ownership boundary — prefer passing an :class:`AtomicUpdater`
    to the writer and letting the registry expose snapshots via a separate read
    path.
    """

    def get(self) -> _T_co:
        """Return the current value.  Lock-free; safe to call from any thread."""
        ...


class AtomicUpdater(Protocol[_T_upd]):
    """Write-only face of an atomic reference cell.

    Pass this to collaborators that only need to publish updates.  Receiving
    an :class:`AtomicUpdater` makes the intent clear: *this object writes,
    it does not read back what it wrote*.  If a collaborator needs to read
    the current snapshot, it should receive an :class:`AtomicReader` as a
    separate dependency — and if it truly needs both, that is usually a sign
    the logic belongs in the owner that holds the full :class:`AtomicReference`.
    """

    def update(
        self,
        selector: "Callable[[Lens[_T_upd]], Lens[_T_upd]]",
        value: object,
    ) -> "_T_upd":
        """Install *value* at the path described by *selector* atomically."""
        ...


class AtomicReference(Generic[_T]):
    """A reference cell whose pointer swap is serialized by a single lock.

    Readers are observationally lock-free — :meth:`get` is a bare attribute
    read with no lock acquire.  Writers hold the internal ``_lock`` only for
    the atomic swap, keeping write-side latency minimal.

    The typical usage pattern is *lens update*::

        ref = AtomicReference(initial_snapshot)

        ref.update(lambda root: root.counter, new_counter_value)

    *selector* passed to :meth:`update` must be a **pure function** — the
    retry loop may call it more than once under write contention.
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

    def compare_and_set(self, expected: _T, new_value: _T) -> tuple[bool, _T]:
        """Swap to *new_value* only if the current reference **is** *expected*.

        Uses identity (``is``) rather than equality (``==``) for the check —
        the comparison is always O(1) and unambiguous regardless of how ``_T``
        defines ``__eq__``.

        Returns a ``(success, value)`` tuple:

        - On success: ``(True, new_value)`` — the swap happened.
        - On failure: ``(False, current)`` — the current reference at the
          moment of rejection.  Callers (e.g. :meth:`update`) can feed
          ``current`` straight into the next attempt without an extra
          :meth:`get` round-trip.
        """
        with self._lock:
            if self._value is not expected:
                return False, self._value
            self._value = new_value
            return True, new_value

    def update(
        self,
        selector: "Callable[[Lens[_T]], Lens[_T]]",
        value: object,
    ) -> _T:
        """Install *value* at the path described by *selector* atomically.

        *selector* receives a :class:`~fido.lens.Lens`-wrapped root and
        returns a navigated :class:`~fido.lens.Lens` pointing at the target
        field.  *value* is installed there; the reconstructed root is CAS'd
        into the reference with automatic retry::

            ref.update(
                lambda root: root.repos[name],
                new_repo_state,
            )

        *selector* must be a pure function — the retry loop may call it
        more than once under write contention.  Returns the reconstructed
        root value that was successfully installed.
        """
        old = self.get()
        while True:
            new: _T = selector(Lens(old)).set(value)  # type: ignore[assignment]
            success, old = self.compare_and_set(old, new)
            if success:
                return new


__all__ = ["AtomicReader", "AtomicReference", "AtomicUpdater"]
