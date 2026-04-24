"""In-memory tree cache for the picker (closes #812).

Replaces the per-iteration ``find_all_open_issues`` polling with a cache
populated once at startup, patched by webhook events, and reconciled
hourly.

Mutations are idempotent so a redelivered or duplicate webhook doesn't
corrupt state.  Events are processed in their own ``timestamp`` order
(not webhook receive order); each cache node tracks ``last_applied_at``
so a stale event arriving after a newer one is dropped.

Thread-safe under Python 3.14t (free-threaded, no GIL): a single
``threading.Lock`` guards every mutation and read.  Per-repo caches are
independent — pass each repo its own :class:`IssueTreeCache`.
"""

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


def _parse_iso(value: str | None) -> datetime:
    """Parse an ISO-8601 GitHub timestamp into a tz-aware datetime.

    Accepts ``None`` / empty as :data:`datetime.min` so missing timestamps
    sort earliest (and thus are always treated as stale next to anything
    real).
    """
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class IssueNode:
    """A single open issue tracked by the cache.

    Closed issues are removed from the cache rather than carried with a
    ``state`` flag — picker logic expects "in cache ⇒ open."

    ``sub_issues`` is the ordered list of *all* sub-issue numbers
    (including ones that have since closed); openness is determined at
    query time by membership in :attr:`IssueTreeCache._nodes`.  Storing
    closed children's numbers preserves the original GitHub ordering for
    strict first-priority descent — GitHub doesn't reorder when a child
    closes, and neither do we.
    """

    number: int
    title: str
    assignees: set[str]
    parent: int | None
    sub_issues: list[int]
    milestone: str | None
    created_at: datetime
    last_applied_at: datetime


@dataclass
class CacheMetrics:
    """Snapshot of cache health for ``fido status`` display."""

    repo_name: str
    inventory_loaded_at: datetime | None
    open_issue_count: int
    events_applied: int
    events_dropped_stale: int
    last_event_at: datetime | None
    last_reconcile_at: datetime | None
    last_reconcile_drift: int


class IssueTreeCache:
    """Per-repo issue tree cache populated by inventory + webhook events.

    Lifecycle:

    1. Construct empty.  Until :meth:`load_inventory` runs, every event
       handed to :meth:`apply_event` is *queued* and applied (in
       timestamp order) the moment inventory completes.  This avoids the
       "events arrive while we're booting" race.

    2. :meth:`load_inventory` seeds the cache from a fresh
       ``find_all_open_issues`` snapshot.  All seeded nodes share the
       same ``last_applied_at`` (the snapshot start time), which becomes
       the staleness watermark — events older than the snapshot are
       subsumed and dropped.

    3. Steady state: webhook handlers call :meth:`apply_event` for every
       relevant ``issues`` / ``pull_request`` / sub-issue event.

    4. :meth:`reconcile_with_inventory` runs hourly to heal drift from
       missed webhooks.  Diffs the cache against a fresh snapshot and
       updates ``last_reconcile_drift`` for visibility.
    """

    def __init__(self, repo_name: str) -> None:
        self._repo = repo_name
        self._lock = threading.Lock()
        self._nodes: dict[int, IssueNode] = {}
        self._inventory_loaded_at: datetime | None = None
        self._events_applied = 0
        self._events_dropped_stale = 0
        self._last_event_at: datetime | None = None
        self._last_reconcile_at: datetime | None = None
        self._last_reconcile_drift = 0
        self._pre_inventory_queue: list[tuple[datetime, str, dict[str, Any]]] = []

    # ── inventory ────────────────────────────────────────────────────────

    def load_inventory(
        self,
        issues: Iterable[dict[str, Any]],
        snapshot_started_at: datetime,
    ) -> None:
        """Replace the cache with the contents of *issues* and apply any
        queued pre-inventory events.

        *snapshot_started_at* must be the time the inventory query was
        issued (not when it returned).  Events with a timestamp earlier
        than this are discarded as subsumed.
        """
        new_nodes = {
            n.number: n
            for n in (self._node_from_inventory(i, snapshot_started_at) for i in issues)
        }
        with self._lock:
            self._nodes = new_nodes
            self._inventory_loaded_at = _now()
            queued = self._pre_inventory_queue
            self._pre_inventory_queue = []
        log.info(
            "issue-cache[%s]: inventory loaded (%d open issues, %d events queued during boot)",
            self._repo,
            len(new_nodes),
            len(queued),
        )
        for _ts, event_type, payload in sorted(queued, key=lambda x: x[0]):
            self.apply_event(event_type, payload)

    def reconcile_with_inventory(
        self,
        issues: Iterable[dict[str, Any]],
        snapshot_started_at: datetime,
    ) -> int:
        """Diff the cache against a fresh inventory snapshot and apply
        every divergence.  Returns the count of corrections applied.

        Used by the hourly watchdog to heal drift caused by lost webhook
        events.  Cache nodes absent from the snapshot are removed; nodes
        in the snapshot that diverge from cache are replaced.  Resets the
        per-node ``last_applied_at`` to *snapshot_started_at* so any
        in-flight events older than the snapshot are subsumed.
        """
        snapshot = {
            n.number: n
            for n in (self._node_from_inventory(i, snapshot_started_at) for i in issues)
        }
        with self._lock:
            drift = 0
            for number in list(self._nodes.keys()):
                if number not in snapshot:
                    del self._nodes[number]
                    drift += 1
                    continue
                cached = self._nodes[number]
                fresh = snapshot[number]
                if not _nodes_equal(cached, fresh):
                    self._nodes[number] = fresh
                    drift += 1
            for number, fresh in snapshot.items():
                if number not in self._nodes:
                    self._nodes[number] = fresh
                    drift += 1
            self._last_reconcile_at = _now()
            self._last_reconcile_drift = drift
        log.info("issue-cache[%s]: reconcile applied %d corrections", self._repo, drift)
        return drift

    @staticmethod
    def _node_from_inventory(
        issue: dict[str, Any], snapshot_started_at: datetime
    ) -> IssueNode:
        return IssueNode(
            number=issue["number"],
            title=issue.get("title", ""),
            assignees={
                a["login"]
                for a in (issue.get("assignees") or {}).get("nodes") or []
                if a.get("login")
            },
            parent=(issue.get("parent") or {}).get("number")
            if issue.get("parent")
            else None,
            sub_issues=[
                c["number"]
                for c in (issue.get("subIssues") or {}).get("nodes") or []
                if c.get("number") is not None
            ],
            milestone=(issue.get("milestone") or {}).get("title")
            if issue.get("milestone")
            else None,
            created_at=_parse_iso(issue.get("createdAt")),
            last_applied_at=snapshot_started_at,
        )

    # ── event entry point ────────────────────────────────────────────────

    def apply_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Apply a webhook-derived event to the cache.

        *payload* must contain at least ``issue_number`` (int) and
        ``timestamp`` (tz-aware datetime).  Other keys are event-typed.

        Behavior:

        - Pre-inventory: queued with timestamp; drained later.
        - Stale (``timestamp < node.last_applied_at`` for an existing
          node): dropped, ``events_dropped_stale`` incremented.
        - Otherwise: dispatched to the matching handler; node's
          ``last_applied_at`` advanced.
        """
        timestamp = payload["timestamp"]
        with self._lock:
            if self._inventory_loaded_at is None:
                self._pre_inventory_queue.append((timestamp, event_type, payload))
                log.info(
                    "issue-cache[%s]: queued %s for #%s pre-inventory (queue depth=%d)",
                    self._repo,
                    event_type,
                    payload.get("issue_number"),
                    len(self._pre_inventory_queue),
                )
                return
            number = payload["issue_number"]
            existing = self._nodes.get(number)
            if existing is not None and timestamp < existing.last_applied_at:
                self._events_dropped_stale += 1
                log.info(
                    "issue-cache[%s]: dropping stale %s for #%s "
                    "(event=%s, last_applied=%s)",
                    self._repo,
                    event_type,
                    number,
                    timestamp.isoformat(),
                    existing.last_applied_at.isoformat(),
                )
                return
            match event_type:
                case "opened":
                    self._handle_opened(payload)
                case "closed":
                    self._handle_closed(payload)
                case "reopened":
                    self._handle_reopened(payload)
                case "assigned":
                    self._handle_assigned(payload)
                case "unassigned":
                    self._handle_unassigned(payload)
                case "milestoned":
                    self._handle_milestoned(payload)
                case "edited_title":
                    self._handle_edited_title(payload)
                case "sub_issue_added":
                    self._handle_sub_issue_added(payload)
                case "sub_issue_removed":
                    self._handle_sub_issue_removed(payload)
                case _:
                    log.debug(
                        "issue-cache[%s]: ignoring unknown event %r",
                        self._repo,
                        event_type,
                    )
                    return
            after = self._nodes.get(number)
            if after is not None:
                after.last_applied_at = max(after.last_applied_at, timestamp)
            self._events_applied += 1
            self._last_event_at = timestamp
            log.info(
                "issue-cache[%s]: applied %s for #%s (open=%d, applied=%d, "
                "stale_dropped=%d)",
                self._repo,
                event_type,
                number,
                len(self._nodes),
                self._events_applied,
                self._events_dropped_stale,
            )

    # ── handlers (called under self._lock by apply_event) ────────────────

    def _handle_opened(self, payload: dict[str, Any]) -> None:
        number = payload["issue_number"]
        node = self._nodes.get(number)
        if node is not None:
            return  # idempotent: already present
        self._nodes[number] = IssueNode(
            number=number,
            title=payload.get("title", ""),
            assignees=set(payload.get("assignees", [])),
            parent=payload.get("parent"),
            sub_issues=list(payload.get("sub_issues", [])),
            milestone=payload.get("milestone"),
            created_at=payload.get("created_at", payload["timestamp"]),
            last_applied_at=payload["timestamp"],
        )

    def _handle_closed(self, payload: dict[str, Any]) -> None:
        self._nodes.pop(payload["issue_number"], None)

    def _handle_reopened(self, payload: dict[str, Any]) -> None:
        # Same shape as opened — payload carries full issue snapshot.
        self._handle_opened(payload)

    def _handle_assigned(self, payload: dict[str, Any]) -> None:
        node = self._nodes.get(payload["issue_number"])
        if node is None:
            return
        node.assignees.add(payload["login"])

    def _handle_unassigned(self, payload: dict[str, Any]) -> None:
        node = self._nodes.get(payload["issue_number"])
        if node is None:
            return
        node.assignees.discard(payload["login"])

    def _handle_milestoned(self, payload: dict[str, Any]) -> None:
        node = self._nodes.get(payload["issue_number"])
        if node is None:
            return
        node.milestone = payload.get("milestone")

    def _handle_edited_title(self, payload: dict[str, Any]) -> None:
        node = self._nodes.get(payload["issue_number"])
        if node is None:
            return
        node.title = payload.get("title", node.title)

    def _handle_sub_issue_added(self, payload: dict[str, Any]) -> None:
        parent_number = payload["issue_number"]
        child_number = payload["child"]
        parent = self._nodes.get(parent_number)
        if parent is None:
            return
        if child_number in parent.sub_issues:
            return  # idempotent
        parent.sub_issues.append(child_number)
        child = self._nodes.get(child_number)
        if child is not None:
            child.parent = parent_number

    def _handle_sub_issue_removed(self, payload: dict[str, Any]) -> None:
        parent_number = payload["issue_number"]
        child_number = payload["child"]
        parent = self._nodes.get(parent_number)
        if parent is None or child_number not in parent.sub_issues:
            return
        parent.sub_issues = [n for n in parent.sub_issues if n != child_number]
        child = self._nodes.get(child_number)
        if child is not None and child.parent == parent_number:
            child.parent = None

    # ── query API (read-locked) ──────────────────────────────────────────

    def get(self, number: int) -> IssueNode | None:
        with self._lock:
            node = self._nodes.get(number)
            if node is None:
                return None
            return _copy_node(node)

    def all_open(self) -> dict[int, IssueNode]:
        """Snapshot copy of every open issue node.  Safe to walk without
        holding the cache lock."""
        with self._lock:
            return {n: _copy_node(node) for n, node in self._nodes.items()}

    def assigned_to(self, login: str) -> list[IssueNode]:
        """All open issues currently assigned to *login*, sorted by
        creation order (oldest first) for stable picker ranking."""
        with self._lock:
            matching = [
                _copy_node(node)
                for node in self._nodes.values()
                if login in node.assignees
            ]
        matching.sort(key=lambda n: (n.created_at, n.number))
        return matching

    def metrics(self) -> CacheMetrics:
        with self._lock:
            return CacheMetrics(
                repo_name=self._repo,
                inventory_loaded_at=self._inventory_loaded_at,
                open_issue_count=len(self._nodes),
                events_applied=self._events_applied,
                events_dropped_stale=self._events_dropped_stale,
                last_event_at=self._last_event_at,
                last_reconcile_at=self._last_reconcile_at,
                last_reconcile_drift=self._last_reconcile_drift,
            )

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._inventory_loaded_at is not None


def _copy_node(node: IssueNode) -> IssueNode:
    """Return a defensive copy so callers can't mutate the cache through
    a returned reference."""
    return IssueNode(
        number=node.number,
        title=node.title,
        assignees=set(node.assignees),
        parent=node.parent,
        sub_issues=list(node.sub_issues),
        milestone=node.milestone,
        created_at=node.created_at,
        last_applied_at=node.last_applied_at,
    )


def _nodes_equal(a: IssueNode, b: IssueNode) -> bool:
    """Field-equality check for reconcile drift detection.

    Ignores ``last_applied_at`` — that's a cache-internal bookkeeping
    field that only changes on event application, not in inventory data.
    """
    return (
        a.number == b.number
        and a.title == b.title
        and a.assignees == b.assignees
        and a.parent == b.parent
        and a.sub_issues == b.sub_issues
        and a.milestone == b.milestone
        and a.created_at == b.created_at
    )


__all__ = [
    "CacheMetrics",
    "IssueNode",
    "IssueTreeCache",
]
