"""Process-wide set of claimed review comment IDs.

The :class:`RepliedComments` class is a thread-safe owner of a set of
comment database IDs that have been claimed by either the webhook handler or
the worker's ``handle_threads`` path.  A single instance lives here so that
both sides can coordinate without a circular import.

**Bidirectional claim protocol**:

- ``server.py`` claims IDs before calling ``reply_to_comment()`` (atomic
  check-and-add via :meth:`RepliedComments.claim`).  ``worker.py`` uses the
  same set in :meth:`~fido.worker.Worker._filter_threads` to skip threads
  whose first comment was already claimed.

- ``worker.py`` also claims each thread's ``first_db_id`` in
  :meth:`~fido.worker.Worker.handle_threads` before launching the comments
  sub-agent.  A concurrent webhook handler that arrives after the worker has
  claimed a thread will see the claim and skip its own reply.

Whichever path claims first wins.  The other path sees the claim and skips,
preventing the comments sub-agent and the webhook handler from each posting
a duplicate reply to the same thread (fixes #672).
"""

import threading


class RepliedComments:
    """Thread-safe set of already-replied comment IDs with atomic claim.

    Eliminates the TOCTOU window between the membership check and the add
    that allowed duplicate webhook deliveries to both pass the dedup guard
    and each independently call reply_to_comment() (closes #566).

    Usage pattern::

        if not _replied_comments.claim(cid):
            return  # already handled by another delivery
        # safe to proceed — this thread holds the exclusive claim
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: set[int] = set()

    def claim(self, cid: int) -> bool:
        """Atomically claim *cid*.

        Returns ``True`` if this call is the first to claim *cid* (newly
        claimed — caller should proceed with the reply).  Returns ``False``
        if *cid* was already present (another thread or a prior delivery
        already claimed it — caller should skip the reply).
        """
        with self._lock:
            if cid in self._ids:
                return False
            self._ids.add(cid)
            return True

    def add(self, cid: int) -> None:
        """Non-atomic add — for test pre-seeding only."""
        with self._lock:
            self._ids.add(cid)

    def release(self, cid: int) -> None:
        """Release a previously claimed *cid* after a failed reply attempt.

        Removes *cid* so a subsequent webhook redelivery can claim it and
        retry the reply.  Call this in the failure path (exception handler)
        after a claim succeeded but the reply call raised.
        """
        with self._lock:
            self._ids.discard(cid)

    def discard(self, cid: int) -> None:
        """Remove *cid* if present — for test cleanup only."""
        with self._lock:
            self._ids.discard(cid)

    def __contains__(self, cid: object) -> bool:
        with self._lock:
            return cid in self._ids


#: Process-wide singleton shared by the webhook handler and the worker.
replied_comments = RepliedComments()
