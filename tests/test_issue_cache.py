"""Tests for fido.issue_cache.IssueTreeCache (closes #812)."""

import threading
from datetime import datetime, timedelta, timezone

from fido.issue_cache import (
    CacheMetrics,
    IssueTreeCache,
)

_T0 = datetime(2026, 4, 18, 22, 0, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float = 0) -> datetime:
    return _T0 + timedelta(seconds=offset_seconds)


def _inv(
    number: int,
    *,
    title: str = "",
    assignees: list[str] | None = None,
    parent: int | None = None,
    sub_issues: list[int] | None = None,
    milestone: str | None = None,
    created_at: str = "2026-04-15T00:00:00Z",
) -> dict[str, object]:
    """Build an inventory-shape dict (matching ``find_all_open_issues``)."""
    return {
        "number": number,
        "title": title,
        "createdAt": created_at,
        "assignees": {"nodes": [{"login": a} for a in (assignees or [])]},
        "parent": {"number": parent} if parent is not None else None,
        "subIssues": {
            "nodes": [{"number": n, "state": "OPEN"} for n in (sub_issues or [])]
        },
        "milestone": {"title": milestone} if milestone else None,
    }


def _evt(
    issue_number: int,
    *,
    timestamp: datetime | None = None,
    **extras: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "issue_number": issue_number,
        "timestamp": timestamp or _ts(10),
    }
    payload.update(extras)
    return payload


# ── inventory load + queue drain ────────────────────────────────────────────


class TestLoadInventory:
    def test_seeds_nodes(self) -> None:
        cache = IssueTreeCache("owner/repo")
        cache.load_inventory(
            [
                _inv(1, title="root", sub_issues=[2, 3]),
                _inv(2, title="child", parent=1, assignees=["fido"]),
                _inv(3, title="other", parent=1, milestone="v1"),
            ],
            snapshot_started_at=_T0,
        )
        n1 = cache.get(1)
        assert n1 is not None
        assert n1.title == "root"
        assert n1.sub_issues == [2, 3]
        n2 = cache.get(2)
        assert n2 is not None
        assert n2.assignees == {"fido"}
        assert n2.parent == 1
        n3 = cache.get(3)
        assert n3 is not None
        assert n3.milestone == "v1"

    def test_marks_loaded(self) -> None:
        cache = IssueTreeCache("owner/repo")
        assert cache.is_loaded is False
        cache.load_inventory([_inv(1)], snapshot_started_at=_T0)
        assert cache.is_loaded is True

    def test_drains_pre_inventory_queue_in_timestamp_order(self) -> None:
        cache = IssueTreeCache("owner/repo")
        # Queue events while not loaded — their 'login' adds should run in
        # timestamp order, not arrival order.
        cache.apply_event("assigned", _evt(1, timestamp=_ts(20), login="alice"))
        cache.apply_event("assigned", _evt(1, timestamp=_ts(10), login="bob"))
        # Both timestamps are AFTER snapshot, so both should apply.
        cache.load_inventory(
            [_inv(1, assignees=["seed"])],
            snapshot_started_at=_T0,
        )
        node = cache.get(1)
        assert node is not None
        # All three logins present.
        assert node.assignees == {"seed", "alice", "bob"}
        # last_applied_at advanced to the latest event timestamp.
        assert node.last_applied_at == _ts(20)

    def test_pre_inventory_event_predating_snapshot_is_dropped(self) -> None:
        cache = IssueTreeCache("owner/repo")
        cache.apply_event(
            "assigned", _evt(1, timestamp=_T0 - timedelta(hours=1), login="ghost")
        )
        cache.load_inventory([_inv(1, assignees=["seed"])], snapshot_started_at=_T0)
        node = cache.get(1)
        assert node is not None
        assert node.assignees == {"seed"}
        m = cache.metrics()
        assert m.events_dropped_stale == 1


# ── apply_event idempotency ─────────────────────────────────────────────────


class TestEventIdempotency:
    def _loaded(self) -> IssueTreeCache:
        cache = IssueTreeCache("o/r")
        cache.load_inventory(
            [
                _inv(1, title="parent", sub_issues=[2]),
                _inv(2, title="child", parent=1, assignees=["fido"]),
            ],
            snapshot_started_at=_T0,
        )
        return cache

    def test_assigned_is_idempotent(self) -> None:
        cache = self._loaded()
        cache.apply_event("assigned", _evt(2, timestamp=_ts(10), login="alice"))
        cache.apply_event("assigned", _evt(2, timestamp=_ts(20), login="alice"))
        node = cache.get(2)
        assert node is not None
        assert node.assignees == {"fido", "alice"}

    def test_unassigned_idempotent_when_login_absent(self) -> None:
        cache = self._loaded()
        cache.apply_event("unassigned", _evt(2, timestamp=_ts(10), login="never-was"))
        node = cache.get(2)
        assert node is not None
        assert node.assignees == {"fido"}

    def test_opened_is_idempotent_for_existing_node(self) -> None:
        cache = self._loaded()
        cache.apply_event(
            "opened",
            _evt(2, timestamp=_ts(10), title="changed", assignees=["other"]),
        )
        node = cache.get(2)
        assert node is not None
        # Existing node's title/assignees not overwritten by re-open.
        assert node.title == "child"
        assert node.assignees == {"fido"}

    def test_closed_is_idempotent_for_absent_node(self) -> None:
        cache = self._loaded()
        cache.apply_event("closed", _evt(999, timestamp=_ts(10)))  # never seen
        # No exception, no change in cached open count.
        assert cache.metrics().open_issue_count == 2

    def test_sub_issue_added_is_idempotent_for_present_link(self) -> None:
        cache = self._loaded()
        cache.apply_event(
            "sub_issue_added",
            _evt(1, timestamp=_ts(10), child=2),  # already linked
        )
        node = cache.get(1)
        assert node is not None
        assert node.sub_issues == [2]

    def test_sub_issue_removed_is_idempotent_for_absent_link(self) -> None:
        cache = self._loaded()
        cache.apply_event(
            "sub_issue_removed",
            _evt(1, timestamp=_ts(10), child=999),
        )
        node = cache.get(1)
        assert node is not None
        assert node.sub_issues == [2]


# ── apply_event mutations actually mutate ───────────────────────────────────


class TestEventMutations:
    def _loaded(self) -> IssueTreeCache:
        cache = IssueTreeCache("o/r")
        cache.load_inventory(
            [
                _inv(1, title="parent", sub_issues=[2, 3]),
                _inv(2, title="child", parent=1, assignees=["fido"]),
                _inv(3, title="sibling", parent=1),
            ],
            snapshot_started_at=_T0,
        )
        return cache

    def test_assigned_adds_login(self) -> None:
        cache = self._loaded()
        cache.apply_event("assigned", _evt(3, timestamp=_ts(10), login="alice"))
        node = cache.get(3)
        assert node is not None
        assert node.assignees == {"alice"}

    def test_unassigned_removes_login(self) -> None:
        cache = self._loaded()
        cache.apply_event("unassigned", _evt(2, timestamp=_ts(10), login="fido"))
        node = cache.get(2)
        assert node is not None
        assert node.assignees == set()

    def test_closed_removes_node(self) -> None:
        cache = self._loaded()
        cache.apply_event("closed", _evt(2, timestamp=_ts(10)))
        assert cache.get(2) is None
        # Parent's sub_issues list still contains the now-closed number,
        # preserving creation order for downstream picker walks.
        n1 = cache.get(1)
        assert n1 is not None
        assert n1.sub_issues == [2, 3]

    def test_opened_adds_node(self) -> None:
        cache = self._loaded()
        cache.apply_event(
            "opened",
            _evt(
                4,
                timestamp=_ts(10),
                title="new",
                assignees=["bob"],
                parent=1,
                sub_issues=[],
                milestone="v1",
                created_at=_ts(10),
            ),
        )
        node = cache.get(4)
        assert node is not None
        assert node.title == "new"
        assert node.assignees == {"bob"}
        assert node.parent == 1
        assert node.milestone == "v1"

    def test_reopened_adds_when_absent(self) -> None:
        cache = self._loaded()
        cache.apply_event(
            "reopened",
            _evt(99, timestamp=_ts(10), title="returns", created_at=_ts(10)),
        )
        node = cache.get(99)
        assert node is not None
        assert node.title == "returns"

    def test_milestoned_updates_milestone(self) -> None:
        cache = self._loaded()
        cache.apply_event("milestoned", _evt(2, timestamp=_ts(10), milestone="v2"))
        node = cache.get(2)
        assert node is not None
        assert node.milestone == "v2"

    def test_milestoned_to_none_clears_milestone(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory([_inv(1, milestone="v1")], snapshot_started_at=_T0)
        cache.apply_event("milestoned", _evt(1, timestamp=_ts(10)))
        node = cache.get(1)
        assert node is not None
        assert node.milestone is None

    def test_edited_title(self) -> None:
        cache = self._loaded()
        cache.apply_event("edited_title", _evt(2, timestamp=_ts(10), title="renamed"))
        node = cache.get(2)
        assert node is not None
        assert node.title == "renamed"

    def test_sub_issue_added_appends_and_links_parent(self) -> None:
        cache = self._loaded()
        cache.apply_event(
            "opened", _evt(4, timestamp=_ts(5), title="loose", created_at=_ts(5))
        )
        cache.apply_event("sub_issue_added", _evt(1, timestamp=_ts(10), child=4))
        n1 = cache.get(1)
        n4 = cache.get(4)
        assert n1 is not None and n4 is not None
        assert n1.sub_issues == [2, 3, 4]
        assert n4.parent == 1

    def test_sub_issue_removed_unlinks(self) -> None:
        cache = self._loaded()
        cache.apply_event("sub_issue_removed", _evt(1, timestamp=_ts(10), child=2))
        n1 = cache.get(1)
        n2 = cache.get(2)
        assert n1 is not None and n2 is not None
        assert n1.sub_issues == [3]
        assert n2.parent is None

    def test_sub_issue_added_for_unknown_parent_is_noop(self) -> None:
        cache = self._loaded()
        cache.apply_event("sub_issue_added", _evt(99, timestamp=_ts(10), child=2))
        # No raise, parent not in cache means nothing added.
        assert cache.get(99) is None

    def test_sub_issue_removed_for_unknown_parent_is_noop(self) -> None:
        cache = self._loaded()
        cache.apply_event("sub_issue_removed", _evt(99, timestamp=_ts(10), child=2))
        n2 = cache.get(2)
        assert n2 is not None
        assert n2.parent == 1  # untouched


# ── stale event rejection ───────────────────────────────────────────────────


class TestHandlersAgainstMissingNode:
    """Every node-mutation handler must no-op cleanly when the target
    node is absent from the cache (idempotent on missing)."""

    def _loaded(self) -> IssueTreeCache:
        cache = IssueTreeCache("o/r")
        cache.load_inventory([_inv(1)], snapshot_started_at=_T0)
        return cache

    def test_assigned_missing_node(self) -> None:
        cache = self._loaded()
        cache.apply_event("assigned", _evt(999, timestamp=_ts(10), login="x"))
        assert cache.get(999) is None

    def test_unassigned_missing_node(self) -> None:
        cache = self._loaded()
        cache.apply_event("unassigned", _evt(999, timestamp=_ts(10), login="x"))
        assert cache.get(999) is None

    def test_milestoned_missing_node(self) -> None:
        cache = self._loaded()
        cache.apply_event("milestoned", _evt(999, timestamp=_ts(10), milestone="v2"))
        assert cache.get(999) is None

    def test_edited_title_missing_node(self) -> None:
        cache = self._loaded()
        cache.apply_event("edited_title", _evt(999, timestamp=_ts(10), title="ghost"))
        assert cache.get(999) is None


class TestStalenessRejection:
    def test_event_older_than_last_applied_at_dropped(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory(
            [_inv(1, assignees=["fido"])], snapshot_started_at=_ts(100)
        )
        # Event timestamped BEFORE snapshot — stale.
        cache.apply_event("unassigned", _evt(1, timestamp=_ts(50), login="fido"))
        node = cache.get(1)
        assert node is not None
        assert node.assignees == {"fido"}
        assert cache.metrics().events_dropped_stale == 1

    def test_event_after_advances_last_applied_at(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory([_inv(1)], snapshot_started_at=_T0)
        cache.apply_event("assigned", _evt(1, timestamp=_ts(60), login="x"))
        cache.apply_event("assigned", _evt(1, timestamp=_ts(30), login="y"))
        node = cache.get(1)
        assert node is not None
        # 'y' was older than 60s — dropped as stale.
        assert node.assignees == {"x"}
        assert cache.metrics().events_dropped_stale == 1

    def test_unknown_event_type_is_ignored(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory([_inv(1)], snapshot_started_at=_T0)
        cache.apply_event(
            "transferred",
            _evt(1, timestamp=_ts(10)),  # not a type we handle
        )
        # No exception, no counter increment.
        m = cache.metrics()
        assert m.events_applied == 0
        assert m.events_dropped_stale == 0


# ── reconcile ───────────────────────────────────────────────────────────────


class TestReconcile:
    def test_reports_zero_drift_when_in_sync(self) -> None:
        cache = IssueTreeCache("o/r")
        snap = [_inv(1, title="x", sub_issues=[]), _inv(2, parent=1)]
        cache.load_inventory(snap, snapshot_started_at=_T0)
        drift = cache.reconcile_with_inventory(snap, snapshot_started_at=_ts(60))
        assert drift == 0
        m = cache.metrics()
        assert m.last_reconcile_drift == 0
        assert m.last_reconcile_at is not None

    def test_removes_nodes_absent_from_snapshot(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory([_inv(1), _inv(2)], snapshot_started_at=_T0)
        drift = cache.reconcile_with_inventory([_inv(1)], snapshot_started_at=_ts(60))
        assert drift == 1
        assert cache.get(2) is None

    def test_adds_nodes_present_only_in_snapshot(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory([_inv(1)], snapshot_started_at=_T0)
        drift = cache.reconcile_with_inventory(
            [_inv(1), _inv(2, title="new")], snapshot_started_at=_ts(60)
        )
        assert drift == 1
        n2 = cache.get(2)
        assert n2 is not None
        assert n2.title == "new"

    def test_replaces_diverged_node(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory([_inv(1, assignees=["fido"])], snapshot_started_at=_T0)
        drift = cache.reconcile_with_inventory(
            [_inv(1, assignees=["alice"])], snapshot_started_at=_ts(60)
        )
        assert drift == 1
        n = cache.get(1)
        assert n is not None
        assert n.assignees == {"alice"}


# ── thread safety smoke ─────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_assigned_unassigned_does_not_corrupt(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory([_inv(1)], snapshot_started_at=_T0)

        def writer(login: str, n_iters: int) -> None:
            for i in range(n_iters):
                ts = _T0 + timedelta(seconds=i + 1)
                cache.apply_event("assigned", _evt(1, timestamp=ts, login=login))
                cache.apply_event("unassigned", _evt(1, timestamp=ts, login=login))

        threads = [
            threading.Thread(target=writer, args=(name, 50))
            for name in ("a", "b", "c", "d")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Assignees set ends up empty (every add paired with same-timestamp
        # remove); cache is internally consistent (no exceptions).
        node = cache.get(1)
        assert node is not None


# ── query API ───────────────────────────────────────────────────────────────


class TestQueryAPI:
    def _loaded(self) -> IssueTreeCache:
        cache = IssueTreeCache("o/r")
        cache.load_inventory(
            [
                _inv(
                    1,
                    title="root",
                    sub_issues=[2, 3],
                    created_at="2026-04-10T00:00:00Z",
                ),
                _inv(
                    2,
                    title="a",
                    parent=1,
                    assignees=["fido"],
                    created_at="2026-04-15T00:00:00Z",
                ),
                _inv(
                    3,
                    title="b",
                    parent=1,
                    assignees=["fido", "alice"],
                    created_at="2026-04-12T00:00:00Z",
                ),
                _inv(
                    4,
                    title="other",
                    assignees=["alice"],
                    created_at="2026-04-11T00:00:00Z",
                ),
            ],
            snapshot_started_at=_T0,
        )
        return cache

    def test_get_returns_defensive_copy(self) -> None:
        cache = self._loaded()
        node = cache.get(2)
        assert node is not None
        node.assignees.add("hacker")
        # Cache is untouched.
        n_again = cache.get(2)
        assert n_again is not None
        assert n_again.assignees == {"fido"}

    def test_get_unknown_returns_none(self) -> None:
        cache = self._loaded()
        assert cache.get(999) is None

    def test_all_open_returns_copies(self) -> None:
        cache = self._loaded()
        snap = cache.all_open()
        assert set(snap.keys()) == {1, 2, 3, 4}
        snap[2].assignees.clear()
        assert cache.get(2) is not None
        assert cache.get(2).assignees == {"fido"}  # type: ignore[union-attr]

    def test_assigned_to_returns_oldest_first(self) -> None:
        cache = self._loaded()
        results = cache.assigned_to("fido")
        # #3 created 2026-04-12, #2 created 2026-04-15 — #3 is oldest.
        assert [n.number for n in results] == [3, 2]

    def test_assigned_to_empty_when_no_match(self) -> None:
        cache = self._loaded()
        assert cache.assigned_to("nobody") == []


# ── metrics ─────────────────────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_before_inventory(self) -> None:
        cache = IssueTreeCache("o/r")
        m = cache.metrics()
        assert isinstance(m, CacheMetrics)
        assert m.repo_name == "o/r"
        assert m.inventory_loaded_at is None
        assert m.open_issue_count == 0
        assert m.events_applied == 0
        assert m.events_dropped_stale == 0
        assert m.last_event_at is None
        assert m.last_reconcile_at is None

    def test_metrics_after_load_and_event(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory([_inv(1), _inv(2)], snapshot_started_at=_T0)
        cache.apply_event("assigned", _evt(1, timestamp=_ts(10), login="x"))
        m = cache.metrics()
        assert m.inventory_loaded_at is not None
        assert m.open_issue_count == 2
        assert m.events_applied == 1
        assert m.last_event_at == _ts(10)


class TestNodeShape:
    def test_inventory_with_missing_created_at_falls_back_to_min(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory(
            [
                {
                    "number": 1,
                    "title": "no date",
                    # createdAt omitted entirely
                    "assignees": {"nodes": []},
                    "parent": None,
                    "subIssues": {"nodes": []},
                    "milestone": None,
                }
            ],
            snapshot_started_at=_T0,
        )
        node = cache.get(1)
        assert node is not None
        assert node.created_at == datetime.min.replace(tzinfo=timezone.utc)

    def test_inventory_with_no_assignees(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory(
            [
                {
                    "number": 1,
                    "title": "x",
                    "createdAt": "2026-04-15T00:00:00Z",
                    "assignees": None,  # GitHub sometimes returns null
                    "parent": None,
                    "subIssues": None,
                    "milestone": None,
                }
            ],
            snapshot_started_at=_T0,
        )
        node = cache.get(1)
        assert node is not None
        assert node.assignees == set()
        assert node.parent is None
        assert node.sub_issues == []
        assert node.milestone is None

    def test_inventory_with_login_blank_skipped(self) -> None:
        cache = IssueTreeCache("o/r")
        cache.load_inventory(
            [
                {
                    "number": 1,
                    "title": "x",
                    "createdAt": "2026-04-15T00:00:00Z",
                    "assignees": {"nodes": [{"login": ""}, {"login": "fido"}]},
                    "parent": None,
                    "subIssues": {"nodes": []},
                    "milestone": None,
                }
            ],
            snapshot_started_at=_T0,
        )
        node = cache.get(1)
        assert node is not None
        assert node.assignees == {"fido"}
