"""Per-(repo, item) in-memory cache for comment & review data (#1748, #1754).

The reply-back filter epic (#1256) and worker comment-inspection
paths repeatedly need comment text for the same issue/PR within
a short window.  Without a shared cache, every call hits GitHub
and piles latency + rate-limit pressure on the worker.

Cache scope: one instance per ``(repo, item)`` pair, bound at
construction.  Each instance holds three resource kinds for that
single issue/PR:

* ``"issues"`` — top-level comments (``/issues/{n}/comments``,
  covers both issues and PRs since PRs ARE issues at top level)
* ``"pulls"`` — inline review-thread comments
  (``/pulls/{n}/comments``); single-file AND multi-line range
  comments distinguished only by ``line`` / ``start_line`` on the
  raw object
* ``"reviews"`` — review submissions (``/pulls/{n}/reviews``);
  state (APPROVED / COMMENTED / CHANGES_REQUESTED) + optional body

Storage: raw GitHub API objects, deep-frozen via
:func:`fido.frozen.freeze_object`.  Consumers read whatever
fields they need directly from the ``Mapping`` — no translation,
no projection.

Inherits :class:`~fido.webhook_cache.WebhookCache` for the shared
scaffolding (lock + dict, pre-inventory queue, ``apply_event``
staleness shell via per-entry ``last_applied_at``, on_change
callback, metrics base).  The staleness check at the base means
out-of-order webhook deliveries (stale edit after a newer one)
drop naturally.

INV-1 scope (this PR): shape + per-(repo, item) keying + webhook
``apply_event``.  Hydration via ``load_inventory`` and list-style
getters land in #1756 (INV-2).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import requests

from fido.frozen import freeze_object
from fido.github import GitHub
from fido.webhook_cache import WebhookCache

# Three GitHub resource kinds the cache holds, each keyed by its
# own id namespace.  ``(kind, id)`` is the storage key — review id
# 42 and review-thread-comment id 42 are different things and must
# not collide.
KIND_ISSUES = "issues"  # top-level comments on issues/PRs
KIND_PULLS = "pulls"  # inline review-thread comments
KIND_REVIEWS = "reviews"  # review submissions (Approve/Comment/Request)


@dataclass(frozen=True)
class CommentNode:
    """One cached entry: the raw GitHub object plus bookkeeping.

    ``kind`` is which endpoint the data came from (``issues`` /
    ``pulls`` / ``reviews``) — keeps the ``(kind, id)`` storage key
    derivable from the node alone (no heuristic on the raw payload).
    ``data`` is the deep-frozen raw API response (``frozendict`` +
    ``tuple``), preserving every field GitHub gave us.
    ``last_applied_at`` is the WebhookCache staleness watermark —
    set from the entry's ``updated_at`` / ``submitted_at`` on
    apply, advanced when a newer event arrives.
    """

    kind: str
    data: Mapping[str, Any]
    last_applied_at: datetime


@dataclass(frozen=True)
class CacheMetrics:
    """Per-(repo, item) cache statistics surfaced by ``fido status``."""

    repo_name: str
    item: int
    inventory_loaded_at: datetime | None
    entries_cached: int
    events_applied: int
    events_dropped_stale: int
    last_event_at: datetime | None
    last_reconcile_at: datetime | None
    last_reconcile_drift: int


# Webhook event → (cache kind, payload key for the parent item, payload key
# for the entry object).  ``pull_request_review`` events carry the entry on
# ``payload["review"]``; the other two on ``payload["comment"]``.
_EVENT_SPECS: dict[str, tuple[str, str, str]] = {
    "issue_comment": (KIND_ISSUES, "issue", "comment"),
    "pull_request_review_comment": (KIND_PULLS, "pull_request", "comment"),
    "pull_request_review": (KIND_REVIEWS, "pull_request", "review"),
}


def _entry_timestamp(entry: Mapping[str, Any]) -> datetime:
    """Pick the right ISO-8601 timestamp for staleness comparison.

    GitHub bumps ``updated_at`` on every comment edit.  Reviews
    use ``submitted_at`` instead (review bodies don't get an
    ``updated_at`` on the base object).  ``created_at`` is the
    fallback for the small handful of endpoints that emit neither.
    All three are missing only on a malformed payload — let the
    ``KeyError`` raise loud rather than masking it.
    """
    raw_ts = entry.get("updated_at") or entry.get("submitted_at") or entry["created_at"]
    return datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))


class CommentCache(WebhookCache[tuple[str, int], CommentNode, CacheMetrics]):
    """In-memory cache for one issue/PR's comments & reviews.

    Each instance is bound to a single ``(repo, item)`` pair at
    construction; one ``CommentCache`` per item, lazy-created by
    :class:`~fido.registry.WorkerRegistry`.  The repo-and-item
    binding means events for other items in the same repo route
    to a different cache instance (per-item isolation by
    construction).
    """

    def __init__(
        self,
        repo: str,
        gh: GitHub,
        item: int,
        *,
        on_change: "Callable[[CacheMetrics], None] | None" = None,
    ) -> None:
        super().__init__(f"{repo}#{item}", on_change=on_change)
        # ``_gh_repo`` is the plain ``owner/repo`` string the gh getters
        # expect; the base's ``_repo`` keeps the ``owner/repo#item``
        # identifier used for logging and metrics display.
        self._gh_repo = repo
        self._gh = gh
        self._item = item
        # The cache starts NOT loaded — WebhookCache's pre-inventory
        # queue catches events that arrive before :meth:`hydrate` runs.
        # The registry calls hydrate() immediately on creation; the gap
        # only matters when concurrent webhooks arrive on a different
        # thread while the creating thread is mid-fetch (the queue
        # drains in timestamp order once hydrate completes).

    # ── hydration ────────────────────────────────────────────────────────

    def hydrate(self, snapshot_started_at: datetime) -> None:
        """Fetch the three resource lists from GitHub and load them.

        Performs three list calls (top-level comments, review-thread
        comments, review submissions), tags each raw item with its
        ``_kind``, and hands the composed inventory to
        :meth:`WebhookCache.load_inventory`.  The base then drains the
        pre-inventory queue in timestamp order — so any webhook events
        that arrived between cache construction and hydration land
        afterwards without loss.

        *snapshot_started_at* must be the time the first list fetch
        was issued (not when it returned).  Events older than the
        snapshot are subsumed by it during the queue drain.

        For-issues (not PRs) only ``"issues"`` kind applies — the
        ``/pulls/...`` endpoints 404 there.  INV-1 already filters
        non-PR ``issue_comment`` events at the router, so in practice
        hydrate only runs against PR items, but the empty-list
        fallback below keeps it correct either way.
        """
        self.load_inventory(self._fetch_full_inventory(), snapshot_started_at)

    def refresh(self, snapshot_started_at: datetime) -> None:
        """Re-fetch the three lists and reconcile against the cache.

        Used by the periodic-reconcile watchdog (#1759) to heal drift
        from missed webhook deliveries.  Calls
        :meth:`WebhookCache.reconcile_with_inventory` so ids absent
        from the fresh snapshot are evicted (closing the codex P2
        about evict-missing on list fetches).
        """
        self.reconcile_with_inventory(self._fetch_full_inventory(), snapshot_started_at)

    def _fetch_full_inventory(self) -> list[dict[str, Any]]:
        """Fetch all three resource lists, tagging each item with its kind.

        Each endpoint is 404-tolerant independently (codex P2 on
        #1756): the ``/pulls/...`` endpoints 404 for plain issues,
        but a 404 on one of them must not throw away successful
        data from the other two.  Non-404 errors propagate.
        """
        raw_top_level = self._gh.get_issue_comments(self._gh_repo, self._item)
        raw_review_comments = self._fetch_or_empty_on_404(
            lambda: self._gh.get_pull_comments(self._gh_repo, self._item)
        )
        raw_reviews = self._fetch_or_empty_on_404(
            lambda: self._gh.get_pull_reviews(self._gh_repo, self._item)
        )
        inventory: list[dict[str, Any]] = [
            {"_kind": KIND_ISSUES, **r} for r in raw_top_level
        ]
        inventory += [{"_kind": KIND_PULLS, **r} for r in raw_review_comments]
        inventory += [{"_kind": KIND_REVIEWS, **r} for r in raw_reviews]
        return inventory

    @staticmethod
    def _fetch_or_empty_on_404(
        fetch: Callable[[], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        try:
            return fetch()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return []
            raise

    # ── public lookup API ────────────────────────────────────────────────

    def get(self, kind: str, entry_id: int) -> Mapping[str, Any] | None:
        """Return the cached raw entry, or ``None`` if not present.

        O(1) dict lookup — no GitHub round-trip.
        """
        with self._lock:
            node = self._nodes.get((kind, entry_id))
            return node.data if node is not None else None

    def list_top_level(self) -> list[Mapping[str, Any]]:
        """Snapshot of every top-level comment cached for this item.

        Returns a list of the raw frozen objects, in id order.
        Caller-side mutation is impossible (data is frozen).  Order
        is deterministic but not guaranteed to match GitHub's
        post-time order — callers that care should sort by
        ``created_at``.
        """
        return self._snapshot_kind(KIND_ISSUES)

    def list_review_comments(self) -> list[Mapping[str, Any]]:
        """Snapshot of every inline review-thread comment cached.

        Includes single-file AND multi-line range comments — they
        share the ``/pulls/{n}/comments`` endpoint and are
        distinguished by ``line`` / ``start_line`` on the raw object.
        """
        return self._snapshot_kind(KIND_PULLS)

    def list_reviews(self) -> list[Mapping[str, Any]]:
        """Snapshot of every review submission cached.

        Each entry carries ``state`` (APPROVED / COMMENTED /
        CHANGES_REQUESTED) and an optional ``body``.
        """
        return self._snapshot_kind(KIND_REVIEWS)

    def thread(self, comment_id: int) -> list[Mapping[str, Any]]:
        """Return the review thread containing *comment_id*.

        A thread is the root review comment plus every reply to it.
        Walks the cached review-thread comments, picks the root by
        following ``in_reply_to_id``, then returns the comments
        anchored to that root in id order.  Returns ``[]`` when
        ``comment_id`` is not in any thread on the cache.
        """
        with self._lock:
            pulls = {
                int(node.data["id"]): node.data
                for (kind, _), node in self._nodes.items()
                if kind == KIND_PULLS
            }
        anchor = pulls.get(comment_id)
        if anchor is None:
            return []
        root_id = anchor.get("in_reply_to_id") or comment_id
        return sorted(
            (
                data
                for data in pulls.values()
                if (data.get("in_reply_to_id") or int(data["id"])) == root_id
            ),
            key=lambda d: int(d["id"]),
        )

    def _snapshot_kind(self, kind: str) -> list[Mapping[str, Any]]:
        # Documented contract: callers see entries in id order.  Dict
        # iteration order tracks insertion history, which can diverge
        # from id ordering after reconciles or mixed webhook/inventory
        # updates — so sort explicitly (codex P2 on #1756).
        with self._lock:
            entries = [
                node.data
                for (entry_kind, _), node in self._nodes.items()
                if entry_kind == kind
            ]
        return sorted(entries, key=lambda d: int(d["id"]))

    # ── WebhookCache hooks ───────────────────────────────────────────────

    def apply_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Enrich the payload with the cache key + timestamp, then
        delegate to the base's staleness-checked shell.

        GitHub webhook payloads don't naturally include the
        ``timestamp`` / ``_kind`` keys the WebhookCache base wants —
        we derive them from the event and inject them here.  Events
        for unrecognised types or targeting a different item drop
        silently.
        """
        spec = _EVENT_SPECS.get(event_type)
        if spec is None:
            return
        kind, parent_key, entry_key = spec
        parent = payload.get(parent_key)
        if not isinstance(parent, dict):
            return
        item_number = parent.get("number")
        if not isinstance(item_number, int) or item_number != self._item:
            # Wrong item — webhook router shouldn't have sent it here.
            return
        entry = payload.get(entry_key)
        if not isinstance(entry, dict):
            return
        entry_id = entry.get("id")
        if not isinstance(entry_id, int):
            return
        enriched = {
            **payload,
            "_kind": kind,
            "_entry_id": entry_id,
            "_action": payload.get("action"),
            "_entry_frozen": freeze_object(entry),
            "timestamp": _entry_timestamp(entry),
        }
        super().apply_event(event_type, enriched)

    def _node_key(self, node: CommentNode) -> tuple[str, int]:
        # data["id"] is always present on GitHub comment/review objects.
        return (node.kind, int(node.data["id"]))

    def _node_key_from_payload(self, payload: dict[str, Any]) -> tuple[str, int]:
        return (payload["_kind"], payload["_entry_id"])

    def _node_from_inventory(
        self, raw: dict[str, Any], snapshot_started_at: datetime
    ) -> CommentNode:
        """Parse one raw inventory item into a CommentNode.

        Inventory items are dicts with ``_kind`` pinned to the
        endpoint they came from (``KIND_ISSUES`` / ``KIND_PULLS`` /
        ``KIND_REVIEWS``), and the rest of the keys forming the raw
        GitHub object.  INV-3 will compose these from the three
        list-fetch endpoints.
        """
        kind = raw["_kind"]
        rest = {k: v for k, v in raw.items() if k != "_kind"}
        return CommentNode(
            kind=kind,
            data=freeze_object(rest),
            last_applied_at=snapshot_started_at,
        )

    def _node_last_applied_at(self, node: CommentNode) -> datetime:
        return node.last_applied_at

    def _node_with_last_applied_at(
        self, node: CommentNode, ts: datetime
    ) -> CommentNode:
        return replace(node, last_applied_at=ts)

    def _nodes_equal(self, a: CommentNode, b: CommentNode) -> bool:
        # Frozen mappings compare by value; ignore last_applied_at
        # since that's cache-internal bookkeeping.
        return a.data == b.data

    def _dispatch_event(self, event_type: str, payload: dict[str, Any]) -> bool:
        kind = payload["_kind"]
        entry_id = payload["_entry_id"]
        key = (kind, entry_id)
        action = payload["_action"]
        if action == "deleted":
            self._nodes.pop(key, None)
            return True
        # All other actions (created/edited for comments,
        # submitted/edited/dismissed for reviews) upsert.
        self._nodes[key] = CommentNode(
            kind=kind,
            data=payload["_entry_frozen"],
            last_applied_at=payload["timestamp"],
        )
        return True

    def metrics(self) -> CacheMetrics:
        with self._lock:
            base = self._base_metric_fields()
        return CacheMetrics(
            repo_name=base["repo_name"],
            item=self._item,
            inventory_loaded_at=base["inventory_loaded_at"],
            entries_cached=base["node_count"],
            events_applied=base["events_applied"],
            events_dropped_stale=base["events_dropped_stale"],
            last_event_at=base["last_event_at"],
            last_reconcile_at=base["last_reconcile_at"],
            last_reconcile_drift=base["last_reconcile_drift"],
        )


__all__ = [
    "KIND_ISSUES",
    "KIND_PULLS",
    "KIND_REVIEWS",
    "CacheMetrics",
    "CommentCache",
    "CommentNode",
]
