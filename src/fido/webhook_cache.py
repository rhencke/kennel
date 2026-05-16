"""Generic webhook-patched in-memory cache base (#1752).

Shared scaffolding for :class:`~fido.issue_cache.IssueCache` (and,
soon, :class:`~fido.comment_cache.CommentCache`).  Owns:

* the per-repo lock + dict of nodes (``_nodes: dict[K, V]``)
* per-node staleness check via ``last_applied_at`` so a stale
  webhook delivered after a newer one is dropped
* pre-inventory queue draining in timestamp order after
  :meth:`load_inventory`
* periodic :meth:`reconcile_with_inventory` to heal missed-webhook
  drift
* ``on_change`` callback for SCADA publication, fired *outside*
  the cache lock so the callback may acquire other locks
* shared boilerplate for the ``apply_event`` shell

Subclasses provide:

* :meth:`_node_key` — extract the dict key from a node
* :meth:`_node_key_from_payload` — extract the dict key from a
  webhook payload
* :meth:`_node_from_inventory` — parse a raw inventory item into
  a node, seeded with the snapshot's start time as
  ``last_applied_at``
* :meth:`_node_last_applied_at` — accessor (works for both
  mutable and frozen nodes)
* :meth:`_node_with_last_applied_at` — return-or-mutate the node
  with an advanced timestamp (frozen nodes return a new instance;
  mutable nodes set in place and return the same reference)
* :meth:`_dispatch_event` — apply an event-specific handler under
  the lock; returns ``True`` if state mutated (so the base's
  counters and ``last_applied_at`` advance), ``False`` to ignore
* :meth:`_nodes_equal` — field-equality for reconcile drift
  detection
* :meth:`metrics` — return the subclass's specific
  :class:`CacheMetrics` dataclass, composed with
  :meth:`_base_metric_fields`

Thread-safe under Python 3.14t (free-threaded, no GIL): a single
``threading.Lock`` guards every mutation and read.  Per-repo
caches are independent.
"""

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

log = logging.getLogger(__name__)

_K = TypeVar("_K")
_V = TypeVar("_V")
_M = TypeVar("_M")


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class WebhookCache(ABC, Generic[_K, _V, _M]):
    """Per-repo cache patched by webhook events, with inventory hydration."""

    def __init__(
        self,
        repo_name: str,
        *,
        on_change: "Callable[[_M], None] | None" = None,
        pre_inventory_queue_limit: int | None = None,
    ) -> None:
        """Create an empty cache.

        *on_change*, when supplied, is called with a fresh
        :meth:`metrics` snapshot after every mutation
        (load_inventory, apply_event, reconcile_with_inventory) —
        *outside* the cache lock so the callback may acquire other
        locks (e.g. the atomic FidoState cell) without lock-order
        inversion.

        *pre_inventory_queue_limit*, when supplied, caps the depth
        of the pre-inventory event queue.  Once the queue holds
        this many events and another arrives, the OLDEST queued
        event is dropped and ``events_dropped_queue_overflow`` is
        bumped.  ``None`` (default) means unbounded — appropriate
        for caches whose hydration is guaranteed to run at startup
        (e.g. :class:`~fido.issue_cache.IssueTreeCache`).  Caches
        that can stay un-loaded across persistent failures (e.g.
        :class:`~fido.comment_cache.CommentCache` on a permission
        outage) supply a finite cap so memory growth is bounded.
        """
        self._repo = repo_name
        self._lock = threading.Lock()
        self._nodes: dict[_K, _V] = {}
        self._inventory_loaded_at: datetime | None = None
        self._events_applied = 0
        self._events_dropped_stale = 0
        self._events_dropped_queue_overflow = 0
        self._last_event_at: datetime | None = None
        self._last_reconcile_at: datetime | None = None
        self._last_reconcile_drift = 0
        self._pre_inventory_queue: list[tuple[datetime, str, dict[str, Any]]] = []
        self._pre_inventory_queue_limit = pre_inventory_queue_limit
        self._on_change = on_change

    # ── subclass hooks ────────────────────────────────────────────────────

    @abstractmethod
    def _node_key(self, node: _V) -> _K:
        """Return the dict key for *node*.  Subclass implements."""

    @abstractmethod
    def _node_key_from_payload(self, payload: dict[str, Any]) -> _K:
        """Extract the dict key from a webhook event payload."""

    @abstractmethod
    def _node_from_inventory(
        self, raw: dict[str, Any], snapshot_started_at: datetime
    ) -> _V:
        """Parse one raw inventory item into a cache node.

        The node's ``last_applied_at`` should be seeded with
        *snapshot_started_at* so any in-flight events older than the
        snapshot are subsumed.
        """

    @abstractmethod
    def _node_last_applied_at(self, node: _V) -> datetime:
        """Return the ``last_applied_at`` timestamp of *node*."""

    @abstractmethod
    def _node_with_last_applied_at(self, node: _V, ts: datetime) -> _V:
        """Return *node* with ``last_applied_at`` advanced to *ts*.

        For mutable node types (e.g. plain dataclass), this may mutate
        in place and return the same reference.  For frozen node
        types, it must return a new instance.  The base then stores
        the returned reference back into ``self._nodes``.
        """

    @abstractmethod
    def _dispatch_event(self, event_type: str, payload: dict[str, Any]) -> bool:
        """Apply an event-specific handler under the cache lock.

        Called by :meth:`_apply_event_locked` after the staleness
        check passes.  Returns ``True`` if cache state mutated (so
        the base bumps counters and advances ``last_applied_at``),
        ``False`` for unknown / ignored event types.
        """

    @abstractmethod
    def _nodes_equal(self, a: _V, b: _V) -> bool:
        """Field-equality check for reconcile drift detection.

        Should ignore ``last_applied_at`` — that's a cache-internal
        bookkeeping field that only changes on event application,
        not in inventory data.
        """

    @abstractmethod
    def metrics(self) -> _M:
        """Return the subclass's specific metrics dataclass.

        Subclass implementations typically compose
        :meth:`_base_metric_fields` with their own fields.
        """

    # ── shared scaffolding ────────────────────────────────────────────────

    def _notify_change(self) -> None:
        """Fire :attr:`_on_change` with fresh :meth:`metrics`.

        Called from every mutation path *after* the cache lock is
        released so the callback is free to acquire other locks.
        """
        if self._on_change is None:
            return
        self._on_change(self.metrics())

    def _base_metric_fields(self) -> dict[str, Any]:
        """Snapshot of the common metric fields.

        Subclass :meth:`metrics` calls this under :attr:`_lock`
        before composing its specific dataclass — keeps the locking
        discipline obvious and lets subclasses add cache-specific
        fields without re-implementing the common bookkeeping.
        """
        return {
            "repo_name": self._repo,
            "inventory_loaded_at": self._inventory_loaded_at,
            "node_count": len(self._nodes),
            "events_applied": self._events_applied,
            "events_dropped_stale": self._events_dropped_stale,
            "events_dropped_queue_overflow": self._events_dropped_queue_overflow,
            "last_event_at": self._last_event_at,
            "last_reconcile_at": self._last_reconcile_at,
            "last_reconcile_drift": self._last_reconcile_drift,
        }

    @property
    def is_loaded(self) -> bool:
        """``True`` once :meth:`load_inventory` has run at least once."""
        with self._lock:
            return self._inventory_loaded_at is not None

    # ── inventory ────────────────────────────────────────────────────────

    def load_inventory(
        self,
        items: Iterable[dict[str, Any]],
        snapshot_started_at: datetime,
        /,
    ) -> None:
        """Replace the cache with the contents of *items* and apply any
        queued pre-inventory events.

        *snapshot_started_at* must be the time the inventory query
        was issued (not when it returned).  Events with a timestamp
        earlier than this are discarded as subsumed.
        """
        new_nodes: dict[_K, _V] = {}
        for raw in items:
            node = self._node_from_inventory(raw, snapshot_started_at)
            new_nodes[self._node_key(node)] = node
        with self._lock:
            self._nodes = new_nodes
            self._inventory_loaded_at = _now()
            queued = self._pre_inventory_queue
            self._pre_inventory_queue = []
        log.info(
            "%s[%s]: inventory loaded (%d nodes, %d events queued during boot)",
            type(self).__name__,
            self._repo,
            len(new_nodes),
            len(queued),
        )
        self._notify_change()
        for _ts, event_type, payload in sorted(queued, key=lambda x: x[0]):
            self.apply_event(event_type, payload)

    def reconcile_with_inventory(
        self,
        items: Iterable[dict[str, Any]],
        snapshot_started_at: datetime,
        /,
    ) -> int:
        """Diff the cache against a fresh inventory snapshot and apply
        every divergence.  Returns the count of corrections applied.

        Used by the hourly watchdog to heal drift caused by lost
        webhook events.  Cache nodes absent from the snapshot are
        removed; nodes in the snapshot that diverge from cache (per
        :meth:`_nodes_equal`) are replaced.
        """
        snapshot: dict[_K, _V] = {}
        for raw in items:
            node = self._node_from_inventory(raw, snapshot_started_at)
            snapshot[self._node_key(node)] = node
        with self._lock:
            drift = 0
            for key in list(self._nodes.keys()):
                if key not in snapshot:
                    del self._nodes[key]
                    drift += 1
                    continue
                if not self._nodes_equal(self._nodes[key], snapshot[key]):
                    self._nodes[key] = snapshot[key]
                    drift += 1
            for key, fresh in snapshot.items():
                if key not in self._nodes:
                    self._nodes[key] = fresh
                    drift += 1
            self._last_reconcile_at = _now()
            self._last_reconcile_drift = drift
        log.info(
            "%s[%s]: reconcile applied %d corrections",
            type(self).__name__,
            self._repo,
            drift,
        )
        self._notify_change()
        return drift

    # ── event entry point ────────────────────────────────────────────────

    def apply_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Apply a webhook-derived event to the cache.

        *payload* must contain at least ``timestamp`` (tz-aware
        :class:`datetime`).  Behavior:

        - Pre-inventory: queued with timestamp; drained later in
          timestamp order.
        - Stale (``timestamp < node.last_applied_at`` for an
          existing node): dropped; ``events_dropped_stale``
          incremented.
        - Otherwise: dispatched to the subclass's
          :meth:`_dispatch_event`; node's ``last_applied_at``
          advanced.

        The subclass-specific dispatcher is responsible for the
        per-event handler logic.
        """
        if self._apply_event_locked(event_type, payload):
            self._notify_change()

    def _apply_event_locked(self, event_type: str, payload: dict[str, Any]) -> bool:
        """Locked half of :meth:`apply_event`.

        Returns ``True`` if the cache mutated (event applied or
        dropped as stale) so the caller can fire
        :meth:`_notify_change` outside the lock.
        """
        timestamp = payload["timestamp"]
        with self._lock:
            if self._inventory_loaded_at is None:
                self._pre_inventory_queue.append((timestamp, event_type, payload))
                # Cap depth (codex P1 follow-up on #1766): when
                # hydration stays failed for a long time and webhook
                # traffic keeps arriving, an unbounded queue would
                # grow indefinitely.  Drop oldest by *timestamp*
                # (not arrival order) when over the cap — the drain
                # in :meth:`load_inventory` replays sorted by
                # timestamp anyway (events can arrive out of order),
                # so eviction has to match that ordering or it would
                # discard a newer update while keeping older stale
                # events (codex P2 follow-up on #1766).
                if (
                    self._pre_inventory_queue_limit is not None
                    and len(self._pre_inventory_queue) > self._pre_inventory_queue_limit
                ):
                    oldest_idx = min(
                        range(len(self._pre_inventory_queue)),
                        key=lambda i: self._pre_inventory_queue[i][0],
                    )
                    dropped = self._pre_inventory_queue.pop(oldest_idx)
                    self._events_dropped_queue_overflow += 1
                    log.warning(
                        "%s[%s]: pre-inventory queue at cap %d — "
                        "dropped oldest-timestamp event %s @ %s "
                        "(queue overflow=%d)",
                        type(self).__name__,
                        self._repo,
                        self._pre_inventory_queue_limit,
                        dropped[1],
                        dropped[0].isoformat(),
                        self._events_dropped_queue_overflow,
                    )
                else:
                    log.info(
                        "%s[%s]: queued %s pre-inventory (queue depth=%d)",
                        type(self).__name__,
                        self._repo,
                        event_type,
                        len(self._pre_inventory_queue),
                    )
                return False
            key = self._node_key_from_payload(payload)
            existing = self._nodes.get(key)
            if existing is not None and timestamp < self._node_last_applied_at(
                existing
            ):
                self._events_dropped_stale += 1
                log.info(
                    "%s[%s]: dropping stale %s for %r (event=%s, last_applied=%s)",
                    type(self).__name__,
                    self._repo,
                    event_type,
                    key,
                    timestamp.isoformat(),
                    self._node_last_applied_at(existing).isoformat(),
                )
                return True
            if not self._dispatch_event(event_type, payload):
                return False
            after = self._nodes.get(key)
            if after is not None:
                advanced_ts = max(self._node_last_applied_at(after), timestamp)
                self._nodes[key] = self._node_with_last_applied_at(after, advanced_ts)
            self._events_applied += 1
            self._last_event_at = timestamp
            return True


__all__ = ["WebhookCache"]
