"""Atomic reference cell — lock-free read, lock-serialized write.

Writers hold ``_lock`` for the duration of the pointer swap so concurrent
writes are serialized and no update is lost.  Readers call :meth:`get`
without acquiring any lock — a single object-reference load is atomic on all
CPython-supported architectures, including the free-threaded (no-GIL) build.

``T`` should be an immutable value (frozen dataclass, namedtuple, ``None``,
etc.) so that readers who hold a reference to the snapshot can inspect all
fields without encountering races inside the value itself.

:class:`AtomicReader` and :class:`AtomicUpdater` are the two narrow
public Protocol faces.  Use :func:`create_atomic` to create a cell — it
returns ``(reader, updater)`` so callers accept only the face they need.
A collaborator that holds *both* faces is doing something that belongs
inside the owner of the cell (code smell).

:class:`_AtomicReference` is the private backing implementation.  Import
it only in unit tests that need to exercise ``set`` / ``compare_and_set``
internals; production code should use :func:`create_atomic` exclusively.
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


class _AtomicReference(Generic[_T]):
    """Private backing implementation of an atomic reference cell.

    Readers are observationally lock-free — :meth:`get` is a bare attribute
    read with no lock acquire.  Writers hold the internal ``_lock`` only for
    the atomic swap, keeping write-side latency minimal.

    Do not import this class in production code.  Use :func:`create_atomic`
    instead, which returns ``(AtomicReader[T], AtomicUpdater[T])`` so each
    collaborator gets exactly the face it needs.

    The typical usage pattern is *lens update*::

        reader, updater = create_atomic(initial_snapshot)
        updater.update(lambda root: root.counter, new_counter_value)
        current = reader.get()

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


def create_atomic(initial: _T) -> "tuple[AtomicReader[_T], AtomicUpdater[_T]]":
    """Create an atomic reference cell seeded with *initial*.

    Returns a ``(reader, updater)`` pair backed by a single
    :class:`_AtomicReference`.  Pass the reader to collaborators that only
    observe state; pass the updater to collaborators that publish updates.

    A collaborator that holds *both* is crossing an ownership boundary —
    prefer putting that logic in the class that called :func:`create_atomic`.
    """
    ref: _AtomicReference[_T] = _AtomicReference(initial)
    return ref, ref


__all__ = ["AtomicReader", "AtomicUpdater", "create_atomic"]
