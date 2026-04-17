"""Process-wide set of webhook-claimed review comment IDs.

The :class:`RepliedComments` class is a thread-safe owner of a set of
comment database IDs that the webhook handler has already claimed (or is
currently processing a reply for).  A single instance lives here so that
both the webhook handler (``server.py``) and the worker (``worker.py``)
can consult it without a circular import.

``server.py`` claims IDs before calling ``reply_to_comment()`` (atomic
check-and-add via :meth:`RepliedComments.claim`).  ``worker.py`` uses the
same set in :meth:`~kennel.worker.Worker._filter_threads` to skip threads
whose first comment was already claimed, preventing the comments sub-agent
from posting a duplicate reply even when the webhook reply is still in
flight and not yet visible in the GitHub API.
"""

from __future__ import annotations

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
