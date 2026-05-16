"""In-memory tree cache for the picker (closes #812).

Replaces the per-iteration ``find_all_open_issues`` polling with a cache
populated once at startup, patched by webhook events, and reconciled
hourly.

Mutations are idempotent so a redelivered or duplicate webhook doesn't
corrupt state.  Events are processed in their own ``timestamp`` order
(not webhook receive order); each cache node tracks ``last_applied_at``
so a stale event arriving after a newer one is dropped.

Inherits :class:`~fido.webhook_cache.WebhookCache` for the shared
scaffolding (lock + dict, pre-inventory queue, ``apply_event``
staleness shell, ``reconcile_with_inventory``, on_change callback,
metrics base) and implements the issue-tree-specific hooks.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fido.webhook_cache import WebhookCache


def _parse_iso(value: str | None) -> datetime:
    """Parse an ISO-8601 GitHub timestamp into a tz-aware datetime.

    Accepts ``None`` / empty as :data:`datetime.min` so missing timestamps
    sort earliest (and thus are always treated as stale next to anything
    real).
    """
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


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


class IssueTreeCache(WebhookCache[int, IssueNode, CacheMetrics]):
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

    def __init__(
        self,
        repo_name: str,
        *,
        on_change: "Callable[[CacheMetrics], None] | None" = None,
    ) -> None:
        """Create an empty cache.

        *on_change*, when supplied, is called with a fresh
        :class:`CacheMetrics` after every mutation (load_inventory,
        apply_event, reconcile_with_inventory) — *outside* the cache
        lock so the callback may acquire other locks (e.g. the atomic
        FidoState cell) without lock-order inversion.  The
        :class:`~fido.registry.WorkerRegistry` supplies a callback that
        publishes an :class:`~fido.appstate.IssueCacheSnapshot` into
        the per-repo SCADA leaf (#1696 parity).
        """
        super().__init__(repo_name, on_change=on_change)

    # ── public API aliases that preserve the ``issues=`` kwarg surface ──

    def load_inventory(
        self,
        issues: "Iterable[dict[str, Any]]",
        snapshot_started_at: datetime,
    ) -> None:
        super().load_inventory(issues, snapshot_started_at)

    def reconcile_with_inventory(
        self,
        issues: "Iterable[dict[str, Any]]",
        snapshot_started_at: datetime,
    ) -> int:
        return super().reconcile_with_inventory(issues, snapshot_started_at)

    # ── WebhookCache hooks ───────────────────────────────────────────────

    def _node_key(self, node: IssueNode) -> int:
        return node.number

    def _node_key_from_payload(self, payload: dict[str, Any]) -> int:
        return payload["issue_number"]

    def _node_from_inventory(
        self, raw: dict[str, Any], snapshot_started_at: datetime
    ) -> IssueNode:
        return IssueNode(
            number=raw["number"],
            title=raw.get("title", ""),
            assignees={
                a["login"]
                for a in (raw.get("assignees") or {}).get("nodes") or []
                if a.get("login")
            },
            parent=(raw.get("parent") or {}).get("number")
            if raw.get("parent")
            else None,
            sub_issues=[
                c["number"]
                for c in (raw.get("subIssues") or {}).get("nodes") or []
                if c.get("number") is not None
            ],
            milestone=(raw.get("milestone") or {}).get("title")
            if raw.get("milestone")
            else None,
            created_at=_parse_iso(raw.get("createdAt")),
            last_applied_at=snapshot_started_at,
        )

    def _node_last_applied_at(self, node: IssueNode) -> datetime:
        return node.last_applied_at

    def _node_with_last_applied_at(self, node: IssueNode, ts: datetime) -> IssueNode:
        # IssueNode is mutable; advance in place and return the same ref.
        node.last_applied_at = ts
        return node

    def _nodes_equal(self, a: IssueNode, b: IssueNode) -> bool:
        """Ignore ``last_applied_at`` — that's cache-internal bookkeeping."""
        return (
            a.number == b.number
            and a.title == b.title
            and a.assignees == b.assignees
            and a.parent == b.parent
            and a.sub_issues == b.sub_issues
            and a.milestone == b.milestone
            and a.created_at == b.created_at
        )

    def _dispatch_event(self, event_type: str, payload: dict[str, Any]) -> bool:
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
                return False
        return True

    def metrics(self) -> CacheMetrics:
        with self._lock:
            base = self._base_metric_fields()
        return CacheMetrics(
            repo_name=base["repo_name"],
            inventory_loaded_at=base["inventory_loaded_at"],
            open_issue_count=base["node_count"],
            events_applied=base["events_applied"],
            events_dropped_stale=base["events_dropped_stale"],
            last_event_at=base["last_event_at"],
            last_reconcile_at=base["last_reconcile_at"],
            last_reconcile_drift=base["last_reconcile_drift"],
        )

    # ── per-event handlers (called under self._lock by _dispatch_event) ──

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


__all__ = [
    "CacheMetrics",
    "IssueNode",
    "IssueTreeCache",
]
