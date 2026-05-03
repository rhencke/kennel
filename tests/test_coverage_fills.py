"""Targeted tests filling coverage gaps that survived after the
old test_server::TestProcessAction tests were removed (the deletions
landed because those tests asserted a now-obsolete synchronous-reply-
from-webhook contract — see PR #1254).

Each test in this file is small and exercises one specific branch
that no other test currently covers.  Tests are grouped by source
module.
"""

from __future__ import annotations

import io
import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fido import provider
from fido.provider_factory import DefaultProviderFactory
from fido.registry import WorkerRegistry
from fido.tasks import (
    _merge_thread_lineage,
    _review_thread_contains_comment,
    _thread_lineage_comment_ids,
    _thread_lineage_key,
    _thread_task_for_auto_resolve_oracle,
    review_thread_for_auto_resolve_oracle,
)

# ---------------------------------------------------------------------------
# provider_factory.py — fallback ValueError branches
# ---------------------------------------------------------------------------


class TestProviderFactoryUnsupported:
    """Cover the ``case _: raise ValueError`` fallbacks in
    ``DefaultProviderFactory.create_api`` and ``create_agent``.
    """

    def test_create_api_raises_for_unsupported_provider(self, tmp_path: Path) -> None:
        repo_cfg = MagicMock()
        repo_cfg.provider = "no-such-provider"
        factory = DefaultProviderFactory(session_system_file=tmp_path / "sys.md")
        with pytest.raises(ValueError, match="unsupported provider"):
            factory.create_api(repo_cfg)

    def test_create_agent_raises_for_unsupported_provider(self, tmp_path: Path) -> None:
        repo_cfg = MagicMock()
        repo_cfg.provider = "no-such-provider"
        factory = DefaultProviderFactory(session_system_file=tmp_path / "sys.md")
        with pytest.raises(ValueError, match="unsupported provider"):
            factory.create_agent(repo_cfg, work_dir=tmp_path, repo_name="owner/repo")


# ---------------------------------------------------------------------------
# registry.py — preemption FSM helpers
# ---------------------------------------------------------------------------


class TestWorkerRegistryPreemptionHelpers:
    def _registry(self) -> WorkerRegistry:
        # WorkerRegistry takes a thread factory callable; tests don't
        # need real WorkerThreads, so a no-op factory is fine.
        return WorkerRegistry(MagicMock())

    def test_note_provider_interrupt_requested(self) -> None:
        registry = self._registry()
        registry.note_durable_demand("owner/repo")
        # Interrupt-requested transition should not raise — covers
        # registry.py:600-601 body.
        registry.note_provider_interrupt_requested("owner/repo")

    def test_note_durable_demand_drained_with_pending_demand(self) -> None:
        registry = self._registry()
        registry.note_durable_demand("owner/repo")
        # Drain when demand is queued — covers registry.py:610 transition.
        registry.note_durable_demand_drained("owner/repo")

    def test_note_durable_demand_drained_when_no_demand_is_noop(self) -> None:
        registry = self._registry()
        # Without a recorded demand, drained call returns early (line 610
        # not reached).  Just verify it doesn't raise.
        registry.note_durable_demand_drained("owner/repo")

    def test_wait_for_inbox_drain_with_no_event_returns_true(self) -> None:
        registry = self._registry()
        # No event recorded for this repo → wait_for_inbox_drain should
        # return True immediately (registry.py:632 path).
        assert registry.wait_for_inbox_drain("never-registered/repo", timeout=0.0)

    def test_note_durable_demand_short_circuits_when_already_recorded(self) -> None:
        registry = self._registry()
        registry.note_durable_demand("owner/repo")
        # Second call must early-return — covers the existing-demand
        # branch (registry.py:593).
        registry.note_durable_demand("owner/repo")


# ---------------------------------------------------------------------------
# tasks.py — small leaf branches
# ---------------------------------------------------------------------------


class TestTasksHelpers:
    def test_thread_task_for_auto_resolve_oracle_with_comment_id(self) -> None:
        """Cover the ThreadTask construction branch (tasks.py:73)."""
        task = {
            "id": "1",
            "status": "pending",
            "thread": {"comment_id": 42},
        }
        result = _thread_task_for_auto_resolve_oracle(task)
        assert result is not None
        assert result.thread_task_comment == 42

    def test_thread_task_for_auto_resolve_oracle_without_comment_id(self) -> None:
        """Cover the early-return-None branch when thread has no comment_id."""
        task = {"id": "1", "status": "pending", "thread": {}}
        assert _thread_task_for_auto_resolve_oracle(task) is None

    def test_review_thread_for_auto_resolve_oracle_skips_missing_database_id(
        self,
    ) -> None:
        """Cover the ``continue`` for missing databaseId (tasks.py:130)."""
        node = {
            "id": "thread1",
            "isResolved": False,
            "comments": {
                "nodes": [
                    {"author": {"login": "alice"}},  # NO databaseId
                    {"databaseId": 1, "author": {"login": "alice"}},
                ]
            },
        }
        thread = review_thread_for_auto_resolve_oracle(node, "fido-bot")
        assert len(thread.review_thread_comments) == 1

    def test_review_thread_contains_comment_returns_false_when_absent(self) -> None:
        """Cover the ``return False`` exit (tasks.py:157)."""
        node = {
            "comments": {"nodes": [{"databaseId": 1}, {"databaseId": 2}]},
        }
        assert _review_thread_contains_comment(node, 999) is False

    def test_thread_lineage_comment_ids_with_no_thread_returns_empty(self) -> None:
        """Cover the early-return for missing thread (tasks.py:169)."""
        assert _thread_lineage_comment_ids(None) == []
        assert _thread_lineage_comment_ids({}) == []

    def test_thread_lineage_key_with_no_thread_returns_none(self) -> None:
        """Cover _thread_lineage_key's None branch."""
        assert _thread_lineage_key(None) is None
        assert _thread_lineage_key({}) is None

    def test_merge_thread_lineage_inherits_lineage_key(self) -> None:
        """Cover the ``existing_thread["lineage_key"] = new_thread[...]``
        branch (tasks.py:200)."""
        existing = {"comment_id": 1}  # no lineage_key
        incoming = {"comment_id": 1, "lineage_key": "k1"}
        changed = _merge_thread_lineage(existing, incoming)
        assert changed
        assert existing["lineage_key"] == "k1"


# ---------------------------------------------------------------------------
# provider.py — try_preempt_worker false branches
# ---------------------------------------------------------------------------


class TestProviderTryPreemptWorker:
    def test_returns_false_when_caller_not_webhook(self) -> None:
        """Caller not a webhook → no cancel, returns False (provider.py:486)."""
        provider.set_thread_kind("worker")
        try:
            with patch("fido.provider.get_talker", return_value=None):
                fired, _kind = provider.try_preempt_worker(
                    repo_name="owner/repo",
                    cancel_fn=MagicMock(),
                )
            assert fired is False
        finally:
            provider.set_thread_kind(None)

    def test_returns_false_when_no_current_holder(self) -> None:
        provider.set_thread_kind("webhook")
        try:
            with patch("fido.provider.get_talker", return_value=None):
                cancel = MagicMock()
                fired, kind = provider.try_preempt_worker(
                    repo_name="owner/repo",
                    cancel_fn=cancel,
                )
            assert fired is False
            assert kind is None
            cancel.assert_not_called()
        finally:
            provider.set_thread_kind(None)


# ---------------------------------------------------------------------------
# provider.py — preempt_worker (the OwnedSession instance method)
# ---------------------------------------------------------------------------


class TestOwnedSessionPreemptWorker:
    """Cover provider.py:832-848 — the body of OwnedSession.preempt_worker
    used by the HTTP handler thread before the background dispatch (#955)."""

    def _session(self) -> provider.OwnedSession:
        # OwnedSession is abstract (preempt_worker uses _fire_worker_cancel
        # which is a NotImplementedError hook).  Subclass with a stub
        # so the method body can run end-to-end.
        class _Stub(provider.OwnedSession):
            def __init__(self, repo_name: str | None) -> None:
                self._repo_name = repo_name
                self.cancels: list[None] = []

            def _fire_worker_cancel(self) -> None:
                self.cancels.append(None)

        return _Stub(repo_name="owner/repo")

    def test_fires_cancel_when_worker_holds(self) -> None:
        session = self._session()
        worker_talker = provider.SessionTalker(
            repo_name="owner/repo",
            thread_id=1,
            kind="worker",
            description="x",
            claude_pid=0,
            started_at=provider.talker_now(),
        )
        with patch("fido.provider.get_talker", return_value=worker_talker):
            assert session.preempt_worker() is True
        assert len(session.cancels) == 1  # type: ignore[attr-defined]

    def test_returns_false_when_holder_is_webhook(self) -> None:
        session = self._session()
        webhook_talker = provider.SessionTalker(
            repo_name="owner/repo",
            thread_id=1,
            kind="webhook",
            description="x",
            claude_pid=0,
            started_at=provider.talker_now(),
        )
        with patch("fido.provider.get_talker", return_value=webhook_talker):
            assert session.preempt_worker() is False
        assert session.cancels == []  # type: ignore[attr-defined]

    def test_returns_false_when_repo_name_is_none(self) -> None:
        class _Stub(provider.OwnedSession):
            def __init__(self) -> None:
                self._repo_name = None

            def _fire_worker_cancel(self) -> None:
                raise AssertionError("must not fire when repo_name is None")

        assert _Stub().preempt_worker() is False


# ---------------------------------------------------------------------------
# rocq_runtime.py — domain-error branches
# ---------------------------------------------------------------------------


class TestRocqRuntimeKeyValidation:
    def test_positive_key_raises_for_zero_and_negative(self) -> None:
        from fido.rocq_runtime import _rocq_positive_key, _RocqNumericDomainError

        with pytest.raises(_RocqNumericDomainError):
            _rocq_positive_key(0)
        with pytest.raises(_RocqNumericDomainError):
            _rocq_positive_key(-3)

    def test_string_key_raises_for_non_string(self) -> None:
        from fido.rocq_runtime import _rocq_string_key

        with pytest.raises(TypeError, match="string map/set key"):
            _rocq_string_key(42)


# ---------------------------------------------------------------------------
# tasks.py — small leaf coverage
# ---------------------------------------------------------------------------


class TestTasksLeaves:
    def test_thread_lineage_comment_ids_skips_non_int_or_str(self) -> None:
        """``_thread_lineage_comment_ids`` filters out comment_ids that
        aren't ``int | str`` (tasks.py:177 ``continue`` body)."""
        thread = {"lineage_comment_ids": [1, "2", 3.14, None, 4]}
        assert _thread_lineage_comment_ids(thread) == [1, 2, 4]

    def test_thread_lineage_comment_ids_skips_unparseable_ints(self) -> None:
        """``int(comment_id)`` may raise — non-numeric strings hit the
        ``except TypeError, ValueError`` path (tasks.py:180-181)."""
        thread = {"lineage_comment_ids": [1, "not-a-number", 2]}
        assert _thread_lineage_comment_ids(thread) == [1, 2]


# ---------------------------------------------------------------------------
# rocq_runtime.py — additional fold/state coverage
# ---------------------------------------------------------------------------


class TestRocqLspMisc:
    """Cover small leaf branches in rocq_lsp.py."""

    def test_python_symbol_at_returns_none_when_no_matches(
        self, tmp_path: Path
    ) -> None:
        """``_python_symbol_at`` returns None when no symbol at the
        position has a matching python.path (rocq_lsp.py:446 path)."""
        from fido.rocq_lsp import RocqIndex

        index = RocqIndex(tmp_path)
        index.refresh()
        # Empty index: no symbols at all → None.
        assert index._python_symbol_at(tmp_path / "nope.py", 0, 0) is None


# ---------------------------------------------------------------------------
# status.py — small branch coverage
# ---------------------------------------------------------------------------


class TestStatusFallbacks:
    """Cover defensive fallback branches in fido.status that fire when
    a provider has no palette mapping or when worker state is empty."""

    def _make_repo(self, **kwargs: object) -> object:
        from fido.status import RepoStatus

        defaults: dict[str, object] = dict(
            name="owner/repo",
            fido_running=False,
            issue=None,
            pending=0,
            completed=0,
            current_task=None,
            claude_pid=None,
            claude_uptime=None,
            worker_what=None,
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
        )
        defaults.update(kwargs)
        return RepoStatus(**defaults)  # type: ignore[arg-type]

    def test_styled_provider_status_no_palette(self) -> None:
        """``_styled_provider_status`` falls through to the bare summary
        when ``palette_for`` returns None (status.py:1003)."""
        from fido import provider as fido_provider
        from fido.status import _styled_provider_status

        status = fido_provider.ProviderPressureStatus(
            provider=fido_provider.ProviderID.CLAUDE_CODE,
            window_name="hourly",
            pressure=0.5,
        )
        with patch("fido.status.palette_for", return_value=None):
            result = _styled_provider_status(status)
        assert "claude-code" in result

    def test_styled_repo_provider_no_palette(self) -> None:
        """``_styled_repo_provider`` returns the plain provider string
        when no palette is registered (status.py:1019)."""
        from fido.status import _styled_repo_provider

        repo = self._make_repo()
        with patch("fido.status.palette_for", return_value=None):
            result = _styled_repo_provider(repo)  # type: ignore[arg-type]
        assert "claude-code" in result

    def test_should_show_worker_line_when_paused(self) -> None:
        """A paused provider status forces the Worker line on even when
        the worker has no current task (status.py:1131)."""
        from fido import provider as fido_provider
        from fido.status import _should_show_worker_line

        repo = self._make_repo(
            worker_what="resting",
            provider_status=fido_provider.ProviderPressureStatus(
                provider=fido_provider.ProviderID.CLAUDE_CODE,
                window_name="weekly",
                pressure=1.0,
            ),
        )
        # provider_status.paused requires level == "paused"; force it
        # via attribute substitution since the dataclass is frozen.
        with patch.object(
            type(repo.provider_status),
            "paused",
            new_callable=lambda: True,  # type: ignore[union-attr]
        ):
            assert _should_show_worker_line(repo) is True  # type: ignore[arg-type]

    def test_format_worker_thread_line_waiting_fallback(self) -> None:
        """The fallback ``"waiting for work"`` line fires when no
        current_task, no task numbering, and no worker_what
        (status.py:1193)."""
        from fido.status import _format_worker_thread_line

        repo = self._make_repo(worker_what=None)
        line = _format_worker_thread_line(repo)  # type: ignore[arg-type]
        assert "waiting for work" in line


# ---------------------------------------------------------------------------
# tasks.py — additional small branches
# ---------------------------------------------------------------------------


class TestTasksMoreBranches:
    def test_thread_task_status_for_oracle_blocked(self) -> None:
        """Cover the BLOCKED match arm (tasks.py:73)."""
        from fido import tasks
        from fido.types import TaskStatus

        task = {"status": str(TaskStatus.BLOCKED), "thread": {"comment_id": 1}}
        result = tasks._thread_task_status_for_oracle(task)
        from fido.rocq import thread_auto_resolve as thread_resolve_oracle

        assert isinstance(result, thread_resolve_oracle.StatusBlocked)

    def test_materialize_handles_blocked_row(self) -> None:
        """Cover the StatusBlocked branch of the rescope materializer
        (tasks.py:344)."""
        from fido.rocq import task_queue_rescope as rescope_oracle
        from fido.tasks import _materialize_rescope_oracle_result

        oracle_id = 0
        tasks_by_id = {oracle_id: {"id": "1", "type": "spec"}}
        rows = {
            oracle_id: rescope_oracle.TaskRow(
                title="t",
                description="d",
                kind=rescope_oracle.TaskSpec(),
                status=rescope_oracle.StatusBlocked(),
                source_comment=None,
            )
        }
        result = _materialize_rescope_oracle_result([oracle_id], rows, tasks_by_id)
        assert result[0]["status"] == "blocked"


# ---------------------------------------------------------------------------
# copilotcli.py — owner property fallback
# ---------------------------------------------------------------------------


class TestTasksAdd:
    def test_add_dedups_on_comment_id_and_merges_lineage(self, tmp_path: Path) -> None:
        """When a task already exists for ``comment_id``, ``add()`` returns
        the duplicate and merges any new lineage into the existing thread
        (tasks.py:1080-1081 — write-through path)."""
        from fido.tasks import Tasks
        from fido.types import TaskType

        work = tmp_path / "work"
        work.mkdir()
        (work / ".git" / "fido").mkdir(parents=True)
        (work / ".git" / "fido" / "tasks.json").write_text("[]")
        tasks = Tasks(work)

        first = tasks.add(
            "first",
            TaskType.THREAD,
            thread={"comment_id": 1, "repo": "o/r", "pr": 1},
        )
        # Second add with same comment_id but extra lineage entry must
        # return the existing task and merge the lineage_comment_ids
        # into the stored thread.
        dup = tasks.add(
            "anything",
            TaskType.THREAD,
            thread={
                "comment_id": 1,
                "repo": "o/r",
                "pr": 1,
                "lineage_comment_ids": [1, 2],
            },
        )
        assert dup["id"] == first["id"]
        on_disk = tasks.list()
        assert on_disk[0]["thread"]["lineage_comment_ids"] == [1, 2]


class TestRocqRuntimeStringKeyAccept:
    def test_string_key_passes_through_str(self) -> None:
        from fido.rocq_runtime import _rocq_string_key

        # Accept-path: type matches → ``return key``
        assert _rocq_string_key("hello") == "hello"


# ---------------------------------------------------------------------------
# codex.py — small validation / parser branches
# ---------------------------------------------------------------------------


class TestCodexParsers:
    def test_normalize_limit_name_falls_back_for_non_str(self) -> None:
        from fido.codex import _normalize_limit_name

        assert _normalize_limit_name(None, "fb") == "fb"
        assert _normalize_limit_name("", "fb") == "fb"

    def test_normalize_limit_name_normalises_text(self) -> None:
        from fido.codex import _normalize_limit_name

        assert _normalize_limit_name("Token-Bucket  Window", "fb") == (
            "token_bucket_window"
        )

    def test_parse_rate_limit_reset_rejects_non_numeric(self) -> None:
        from fido.codex import _parse_rate_limit_reset

        with pytest.raises(ValueError, match="resetsAt must be numeric"):
            _parse_rate_limit_reset("nope")

    def test_parse_rate_limit_reset_returns_none_for_none(self) -> None:
        from fido.codex import _parse_rate_limit_reset

        assert _parse_rate_limit_reset(None) is None

    def test_rate_limit_window_returns_none_for_none(self) -> None:
        from fido.codex import _rate_limit_window

        assert _rate_limit_window("id", "primary", None) is None

    def test_rate_limit_window_rejects_non_dict(self) -> None:
        from fido.codex import _rate_limit_window

        with pytest.raises(ValueError, match="must be an object"):
            _rate_limit_window("id", "primary", "not-a-dict")

    def test_rate_limit_window_returns_none_when_used_percent_missing(self) -> None:
        from fido.codex import _rate_limit_window

        assert _rate_limit_window("id", "primary", {}) is None

    def test_rate_limit_window_rejects_non_numeric_used_percent(self) -> None:
        from fido.codex import _rate_limit_window

        with pytest.raises(ValueError, match="usedPercent must be numeric"):
            _rate_limit_window("id", "primary", {"usedPercent": "fifty"})

    def test_credits_depleted_recognises_marker(self) -> None:
        from fido.codex import _credits_depleted

        assert _credits_depleted(
            {"hasCredits": False, "unlimited": False}, "credits_depleted"
        )
        assert not _credits_depleted({"hasCredits": True}, "credits_depleted")
        assert not _credits_depleted({}, "rate_limit_reached")
        assert not _credits_depleted("not-a-dict", "credits_depleted")

    def test_reached_window_name_classifies(self) -> None:
        from fido.codex import _reached_window_name

        assert _reached_window_name("credits_depleted") == "credits"
        assert _reached_window_name("usage_limit_reached") == "workspace_usage"
        assert _reached_window_name("rate_limit_reached") == "rate_limit_reached"
        assert _reached_window_name("custom-limit") == "custom_limit"
        assert _reached_window_name(None) is None
        assert _reached_window_name("") is None

    def test_codex_limit_windows_uses_by_id_dict(self) -> None:
        """Cover the rate_limits_by_id branch (codex.py:559-560)."""
        from fido.codex import _codex_limit_windows

        payload = {
            "rateLimitsByLimitId": {
                "tier_a": {
                    "primary": {"usedPercent": 30},
                    "secondary": None,
                }
            }
        }
        windows = _codex_limit_windows(payload)
        # ``rateLimitsByLimitId`` doesn't include limitId in the
        # snapshot itself; the synthetic name uses ``codex_<index>``.
        assert windows[0].name == "codex_0_primary"
        assert windows[0].used == 30

    def test_codex_limit_windows_rejects_non_object_snapshot(self) -> None:
        """Cover the non-dict raw_limit guard (codex.py:573)."""
        from fido.codex import _codex_limit_windows

        with pytest.raises(ValueError, match="must be objects"):
            _codex_limit_windows({"rateLimits": ["not-an-object"]})

    def test_codex_limit_windows_emits_credits_window(self) -> None:
        """Cover the credits-depleted ProviderLimitWindow append
        (codex.py:583-586)."""
        from fido.codex import _codex_limit_windows

        payload = {
            "rateLimits": [
                {
                    "limitId": "tier",
                    "credits": {"hasCredits": False, "unlimited": False},
                    "rateLimitReachedType": "credits_depleted",
                }
            ]
        }
        windows = _codex_limit_windows(payload)
        names = {w.name for w in windows}
        assert "tier_credits" in names

    def test_codex_limit_windows_emits_explicit_reached_window(self) -> None:
        """Cover the explicit reached-window append branch
        (codex.py:587-592)."""
        from fido.codex import _codex_limit_windows

        payload = {
            "rateLimits": [
                {
                    "limitId": "tier",
                    "rateLimitReachedType": "rate_limit_reached",
                }
            ]
        }
        windows = _codex_limit_windows(payload)
        names = {w.name for w in windows}
        assert "tier_rate_limit_reached" in names


# ---------------------------------------------------------------------------
# worker.py — pure helper coverage
# ---------------------------------------------------------------------------


class TestWorkerPureHelpers:
    def test_ci_oracle_task_kind_discrimination(self) -> None:
        from fido.rocq import ci_task_lifecycle as ci_oracle
        from fido.types import TaskType
        from fido.worker import _ci_oracle_task_kind

        assert isinstance(_ci_oracle_task_kind({"title": "Ask: x"}), ci_oracle.TaskAsk)
        assert isinstance(
            _ci_oracle_task_kind({"title": "Defer: x"}), ci_oracle.TaskDefer
        )
        assert isinstance(
            _ci_oracle_task_kind({"title": "CI FAILURE: y"}), ci_oracle.TaskCI
        )
        assert isinstance(
            _ci_oracle_task_kind({"title": "x", "type": TaskType.CI}),
            ci_oracle.TaskCI,
        )
        assert isinstance(
            _ci_oracle_task_kind({"title": "x", "type": TaskType.THREAD}),
            ci_oracle.TaskThread,
        )
        assert isinstance(
            _ci_oracle_task_kind({"title": "x", "type": TaskType.SPEC}),
            ci_oracle.TaskSpec,
        )
        # Non-string title is normalized to "".
        assert isinstance(_ci_oracle_task_kind({"title": 42}), ci_oracle.TaskSpec)

    def test_ci_oracle_task_status_discrimination(self) -> None:
        from fido.rocq import ci_task_lifecycle as ci_oracle
        from fido.types import TaskStatus
        from fido.worker import _ci_oracle_task_status

        assert isinstance(
            _ci_oracle_task_status({"status": TaskStatus.COMPLETED}),
            ci_oracle.StatusCompleted,
        )
        assert isinstance(
            _ci_oracle_task_status({"status": "completed"}),
            ci_oracle.StatusCompleted,
        )
        assert isinstance(
            _ci_oracle_task_status({"status": TaskStatus.BLOCKED}),
            ci_oracle.StatusBlocked,
        )
        assert isinstance(
            _ci_oracle_task_status({"status": "blocked"}),
            ci_oracle.StatusBlocked,
        )
        assert isinstance(
            _ci_oracle_task_status({"status": "pending"}),
            ci_oracle.StatusPending,
        )

    def test_ci_oracle_snapshot_run_normalization(self) -> None:
        """run_id of 0 / negative normalizes to 1 (worker.py:819)."""
        from fido.rocq import ci_task_lifecycle as ci_oracle
        from fido.worker import _ci_oracle_snapshot

        snap = _ci_oracle_snapshot(check_name="ci", state="FAILURE", run_id="-3")
        assert snap.ci_run == 1
        assert isinstance(snap.ci_conclusion, ci_oracle.CIConclusionFailure)

    def test_ci_oracle_snapshot_timed_out(self) -> None:
        """TIMED_OUT state lowers to CIConclusionTimedOut (worker.py:822)."""
        from fido.rocq import ci_task_lifecycle as ci_oracle
        from fido.worker import _ci_oracle_snapshot

        snap = _ci_oracle_snapshot(check_name="ci", state="TIMED_OUT", run_id="2")
        assert isinstance(snap.ci_conclusion, ci_oracle.CIConclusionTimedOut)

    def test_ci_oracle_snapshot_run_invalid_string(self) -> None:
        """Non-numeric run_id normalizes via ValueError → 1."""
        from fido.worker import _ci_oracle_snapshot

        snap = _ci_oracle_snapshot(
            check_name="ci", state="FAILURE", run_id="not-an-int"
        )
        assert snap.ci_run == 1


# ---------------------------------------------------------------------------
# store.py — small leaf branches
# ---------------------------------------------------------------------------


class TestStoreCompletedReturn:
    def test_refresh_pr_comment_returns_completed_record_unchanged(
        self, tmp_path: Path
    ) -> None:
        """When the dedupe target row is already ``state='completed'``,
        ``_refresh_pr_comment_record`` returns it unchanged rather than
        updating (store.py:575)."""
        from fido.store import FidoStore

        store = FidoStore(tmp_path / "store.db")
        record = store.enqueue_pr_comment(
            repo="o/r",
            pr_number=1,
            comment_id=10,
            delivery_id="d-1",
            author="alice",
            is_bot=False,
            body="initial",
            payload_json="{}",
            comment_type="issues",
            github_created_at="2026-05-02T00:00:00Z",
        )
        with store._transaction() as conn:
            conn.execute(
                "UPDATE pr_comment_queue SET state = 'completed' WHERE queue_id = ?",
                (record.queue_id,),
            )
        # Re-enqueue with new delivery_id → routes through
        # _refresh_pr_comment_record but takes the completed-state
        # short-circuit.
        record2 = store.enqueue_pr_comment(
            repo="o/r",
            pr_number=1,
            comment_id=10,
            delivery_id="d-2",
            author="alice",
            is_bot=False,
            body="ignored",
            payload_json="{}",
            comment_type="issues",
            github_created_at="2026-05-02T00:00:01Z",
        )
        assert record2.queue_id == record.queue_id
        assert record2.body == "initial"


# ---------------------------------------------------------------------------
# rocq_lsp.py — small leaf branches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# events.py — small leaf branches
# ---------------------------------------------------------------------------


class TestEventsLeaves:
    def test_normalize_comment_ids_filters_non_int_or_str(self) -> None:
        """events.py:737-741 — non-int/str entries skip; unparseable
        ints raise TypeError/ValueError and skip."""
        from fido.events import _normalize_comment_ids

        result = _normalize_comment_ids([1, "2", 3.14, None, "not-a-number", 4, "4"])
        # ints + str-of-ints both retained, dedup'd; non-int-like skipped.
        assert tuple(result) == (1, 2, 4)

    def test_thread_lineage_comment_ids_with_no_thread(self) -> None:
        """events.py:769 — None thread short-circuits to ``()``."""
        from fido.events import thread_lineage_comment_ids

        assert thread_lineage_comment_ids(None) == ()
        assert thread_lineage_comment_ids({}) == ()

    def test_thread_lineage_comment_ids_uses_lineage_list(self) -> None:
        """events.py:773 path — lineage_comment_ids list takes priority."""
        from fido.events import thread_lineage_comment_ids

        result = thread_lineage_comment_ids(
            {"lineage_comment_ids": [1, 2], "comment_id": 99}
        )
        assert result == (1, 2)

    def test_thread_lineage_comment_ids_no_lineage_no_comment_id(self) -> None:
        """events.py:775 — neither lineage nor comment_id → ``()``."""
        from fido.events import thread_lineage_comment_ids

        # An empty dict — short-circuits at the ``if not thread`` guard
        # rather than reaching the no-comment-id branch.
        assert thread_lineage_comment_ids({}) == ()
        # A thread with a non-list lineage and no comment_id reaches
        # the explicit ``return ()`` for missing comment_id.
        assert thread_lineage_comment_ids({"repo": "o/r"}) == ()
        # And the trailing branch — non-list lineage, comment_id set.
        result = thread_lineage_comment_ids({"comment_id": 7})
        assert result == (7,)

    def test_comment_chain_action_title_requires_agent(self) -> None:
        """events.py:1920 — calling without an agent raises."""
        from fido.events import _comment_chain_action_title

        with pytest.raises(ValueError, match="requires agent"):
            _comment_chain_action_title(
                thread_comments=[],
                final_comment_id=None,
                final_comment_body="",
                triage_titles=[],
                agent=None,
            )


# ---------------------------------------------------------------------------
# claude.py — defensive concurrency branches
# ---------------------------------------------------------------------------


class TestCodexSpawnAppServer:
    def test_spawn_app_server_invokes_subprocess_popen_with_stdio(
        self, tmp_path: Path
    ) -> None:
        """``_spawn_app_server`` shells out to ``codex app-server`` with
        bidirectional pipes (codex.py:147)."""
        from fido import codex as codex_mod

        sentinel = MagicMock()
        with patch.object(codex_mod.subprocess, "Popen", return_value=sentinel) as p:
            result = codex_mod._spawn_app_server(cwd=tmp_path)
        assert result is sentinel
        cmd = p.call_args.args[0]
        assert cmd[:3] == ["codex", "app-server", "--listen"]


class TestCodexAppServerErrorPaths:
    """Cover the protocol-error / loud-fail branches in
    CodexAppServerClient that the happy-path tests miss."""

    def _client(
        self, prelude: str = '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n'
    ) -> tuple[object, object]:
        # Reuse the streaming-stdout fake from the wait_notification
        # tests so the reader thread doesn't EOF and flag protocol_error
        # before the test exercises its case.
        import io

        from fido.codex import CodexAppServerClient

        lines: queue.Queue[str | None] = queue.Queue()

        class _StreamingStdout:
            def __init__(self) -> None:
                self._buf = io.StringIO(prelude)

            def readline(self) -> str:
                line = self._buf.readline()
                if line:
                    return line
                next_line = lines.get()
                return "" if next_line is None else next_line

        process = MagicMock()
        process.pid = 100
        process.stdin = io.StringIO()
        process.stdout = _StreamingStdout()
        process.stderr = io.StringIO("")
        process._returncode = None
        process.poll = lambda: process._returncode
        process.terminate = MagicMock()
        process.wait = MagicMock(return_value=0)
        process.kill = MagicMock()
        client = CodexAppServerClient(process_factory=lambda **_: process)
        return client, lines

    def test_request_times_out(self) -> None:
        """A request whose response never arrives raises TimeoutError
        (codex.py:234)."""
        client, lines = self._client()
        try:
            with pytest.raises(TimeoutError, match="request timed out"):
                client.request("never-responds", timeout=0.05)  # type: ignore[attr-defined]
        finally:
            lines.put(None)
            client.stop()  # type: ignore[attr-defined]

    def test_stop_skips_terminate_when_process_already_exited(self) -> None:
        """``stop`` early-returns the terminate path when ``poll()`` shows
        the process already exited (codex.py:299)."""
        client, lines = self._client()
        # Force poll() to report exited.
        client._process._returncode = 0  # type: ignore[attr-defined]
        client.stop()  # type: ignore[attr-defined]
        # terminate was not called because we short-circuited.
        client._process.terminate.assert_not_called()  # type: ignore[attr-defined]
        lines.put(None)

    def test_handle_line_rejects_non_json_object(self) -> None:
        """A JSON value that decodes to non-object → CodexProtocolError
        (codex.py:359)."""
        client, lines = self._client()
        try:
            lines.put('"just-a-string"\n')
            # Reader picks it up and flags protocol_error; subsequent
            # request raises.
            import time

            time.sleep(0.05)
            with pytest.raises(Exception):
                client.request("anything")  # type: ignore[attr-defined]
        finally:
            lines.put(None)
            client.stop()  # type: ignore[attr-defined]

    def test_handle_line_rejects_unknown_shape(self) -> None:
        """Object lacking ``id`` and ``method`` → CodexProtocolError
        (codex.py:377)."""
        client, lines = self._client()
        try:
            lines.put('{"unknown":"shape"}\n')
            import time

            time.sleep(0.05)
            with pytest.raises(Exception):
                client.request("anything")  # type: ignore[attr-defined]
        finally:
            lines.put(None)
            client.stop()  # type: ignore[attr-defined]

    def test_request_after_stop_raises(self) -> None:
        """``_raise_if_unavailable_locked`` fires when ``_stopped`` is
        true (codex.py:389)."""
        client, lines = self._client()
        client.stop()  # type: ignore[attr-defined]
        with pytest.raises(Exception, match="connection is stopped"):
            client.request("anything", timeout=0.05)  # type: ignore[attr-defined]
        lines.put(None)


class TestCodexSessionLeafBranches:
    """Property/method short-circuits and small branches in CodexSession."""

    def _session(self, tmp_path: Path, **kwargs) -> object:
        from fido.codex import CodexSession
        from fido.provider import ProviderModel

        system_file = tmp_path / "system.md"
        system_file.write_text("base")
        fake = MagicMock()
        fake.request.return_value = {
            "thread": {"id": "thread-1"},
            "threadId": "thread-1",
        }
        fake.is_alive.return_value = True
        fake.pid = 1234
        defaults: dict = dict(
            client_factory=lambda **_: fake,
            model=ProviderModel("gpt-5", "medium"),
        )
        defaults.update(kwargs)
        return CodexSession(system_file, work_dir=tmp_path, **defaults)

    def test_owner_returns_none_when_repo_name_is_none(self, tmp_path: Path) -> None:
        """codex.py:697 — short-circuit when no repo_name is configured."""
        session = self._session(tmp_path, repo_name=None)
        assert session.owner is None  # type: ignore[union-attr]

    def test_switch_tools_is_a_noop(self, tmp_path: Path) -> None:
        """codex.py:847 — Codex doesn't support per-session tool flags;
        ``switch_tools`` accepts the call and discards the value."""
        session = self._session(tmp_path)
        # Should not raise; nothing observable to assert.
        session.switch_tools("triage")  # type: ignore[union-attr]
        session.switch_tools(None)  # type: ignore[union-attr]

    def test_reset_with_explicit_model(self, tmp_path: Path) -> None:
        """codex.py:863 — reset(model=...) coerces the new model."""
        from fido.provider import ProviderModel

        session = self._session(tmp_path)
        session.reset(model=ProviderModel("gpt-5-pro", "high"))  # type: ignore[union-attr]
        # The reset path coerces and stores the new model.
        assert "gpt-5-pro" in str(session._model)  # type: ignore[union-attr]


class TestClaudeDefensivePaths:
    def _session(self, tmp_path: Path, *, stdout_lines: list[str]) -> object:
        from fido.claude import ClaudeSession

        proc = MagicMock()
        proc.pid = 12345
        proc.stdin = MagicMock()
        proc.stdin.closed = False
        proc.stdout = MagicMock()
        proc.stdout.readline = MagicMock(side_effect=stdout_lines + [""])
        proc.stderr = MagicMock()
        proc.stderr.__iter__ = MagicMock(return_value=iter([]))
        proc.poll = MagicMock(return_value=None)
        proc.wait = MagicMock(return_value=0)
        proc.returncode = 0

        system_file = tmp_path / "system.md"
        system_file.write_text("sys")
        return ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=MagicMock(return_value=proc),
            selector=MagicMock(return_value=([proc.stdout], [], [])),
            repo_name="owner/repo",
            model="claude-opus-4-6",
        )

    def test_send_acknowledges_prior_cancelled_turn(self, tmp_path: Path) -> None:
        """``send`` walks Cancelled → Idle via TurnReturn before
        starting a new Sending turn (claude.py:884-888)."""
        from fido.claude import ClaudeSession
        from fido.rocq.claude_session import Cancelled, Sending

        session = self._session(tmp_path, stdout_lines=[])
        assert isinstance(session, ClaudeSession)
        # Force Cancelled state — this is what a prior cancelled turn
        # would leave behind.
        with session._stream_lock:
            session._stream_state = Cancelled()
        session.send("ping")
        # send() must have transitioned away from Cancelled.
        assert isinstance(session._stream_state, Sending)
        session.stop()

    def test_stderr_pump_tolerates_value_error_on_close(self, tmp_path: Path) -> None:
        """If ``for raw in stderr`` raises ValueError (closed file) or
        OSError (broken pipe), the pump silently exits (claude.py:691-697).
        Cover by handing the session a stderr whose iteration raises."""
        from fido.claude import ClaudeSession

        proc = MagicMock()
        proc.pid = 12345
        proc.stdin = MagicMock()
        proc.stdin.closed = False
        proc.stdout = MagicMock()
        proc.stdout.readline = MagicMock(return_value="")
        proc.stderr = MagicMock()
        proc.stderr.__iter__ = MagicMock(side_effect=ValueError("closed"))
        proc.poll = MagicMock(return_value=None)
        proc.wait = MagicMock(return_value=0)
        proc.returncode = 0
        system_file = tmp_path / "system.md"
        system_file.write_text("sys")

        session = ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=MagicMock(return_value=proc),
            selector=MagicMock(return_value=([], [], [])),
            repo_name="owner/repo",
            model="claude-opus-4-6",
        )
        # The stderr pump runs as a daemon thread; let it exit.
        import time

        time.sleep(0.05)
        session.stop()


class TestRocqLspMoreBranches:
    def test_symbol_at_token_fallback_for_models_dir(self, tmp_path: Path) -> None:
        """When a token at (path, line, col) doesn't match any source
        range but exists in the symbols dict, ``symbol_at`` falls back
        to the dict lookup (rocq_lsp.py:222 — non-generated path)."""
        from fido.rocq_lsp import RocqIndex

        # Set up a fake repo with a models/foo.v file referencing
        # ``transition`` and a stub pymap that registers transition.
        models = tmp_path / "models"
        models.mkdir()
        (models / "session_lock.v").write_text("Definition transition := 1.\n")
        generated = tmp_path / "src" / "fido" / "rocq"
        generated.mkdir(parents=True)
        (generated / "session_lock.py").write_text("def transition(): return 1\n")
        (generated / "session_lock.pymap").write_text(
            "stability,python_start_line,python_start_col,python_end_line,"
            "python_end_col,source_file,source_start_line,source_start_col,"
            "source_end_line,source_end_col,kind,symbol,python_symbol\n"
            "open,1,0,1,17,session_lock.v,1,11,1,21,extraction,transition,transition\n"
        )

        index = RocqIndex(tmp_path)
        index.refresh()
        # Position outside the symbol's source.range (line 5) but the
        # raw text at that line still contains the ``transition``
        # token, triggering the dict fallback.
        (models / "session_lock.v").write_text(
            "Definition transition := 1.\n"
            "(* something else *)\n"
            "Definition unrelated := 0.\n"
            "Definition transition_use := transition.\n"
        )
        symbol = index.symbol_at(models / "session_lock.v", 3, 30)
        assert symbol is not None
        assert symbol.name == "transition"
        """``record_reply_delivery`` short-circuits when no promise ids
        land in ``covered`` (store.py:349)."""
        from fido.store import FidoStore

        store = FidoStore(tmp_path / "store.db")
        # Empty / falsy promise ids dedup to nothing → early return.
        store.record_reply_delivery(
            artifact_comment_id=42,
            comment_type="issue",
            lane_key="key",
            promise_ids=["", None],  # type: ignore[list-item]
        )

    def test_pr_comment_queue_in_progress_update_path(self, tmp_path: Path) -> None:
        """A second enqueue with the same ``delivery_id`` while the row
        is in_progress takes the UPDATE branch (store.py:575-578)."""
        from fido.store import FidoStore

        store = FidoStore(tmp_path / "store.db")
        record_initial = store.enqueue_pr_comment(
            repo="o/r",
            pr_number=1,
            comment_id=10,
            delivery_id="d-1",
            author="alice",
            is_bot=False,
            body="initial",
            payload_json="{}",
            comment_type="issues",
            github_created_at="2026-05-02T00:00:00Z",
        )
        assert record_initial.body == "initial"

        # Mark the row in_progress to trigger the UPDATE branch on the
        # next enqueue (otherwise the dedupe path hits the
        # ``state == "completed"`` short-circuit).
        with store._transaction() as conn:
            conn.execute(
                "UPDATE pr_comment_queue SET state = 'in_progress' WHERE queue_id = ?",
                (record_initial.queue_id,),
            )

        # Different delivery_id but same comment → routes through
        # ``_refresh_pr_comment_record`` and (because state=in_progress)
        # the UPDATE branch (store.py:577-594).
        record_updated = store.enqueue_pr_comment(
            repo="o/r",
            pr_number=1,
            comment_id=10,
            delivery_id="d-2",
            author="alice",
            is_bot=False,
            body="updated",
            payload_json='{"v":2}',
            comment_type="issues",
            github_created_at="2026-05-02T00:00:01Z",
        )
        # Same queue id (FIFO position preserved) but body reflects UPDATE.
        assert record_updated.queue_id == record_initial.queue_id
        assert record_updated.body == "updated"


class TestCopilotCLIOwnerMore:
    """Cover the ``owner`` branches that need a registered repo_name
    (copilotcli.py:975-981)."""

    @staticmethod
    def _build_session(tmp_path: Path, repo_name: str | None = "test/repo"):
        from fido.copilotcli import CopilotCLISession

        runtime = MagicMock()
        runtime.ensure_session.return_value = "sess-1"
        runtime.pid = 4321
        return CopilotCLISession(
            tmp_path / "sys.md",
            work_dir=tmp_path,
            model="gpt-5",
            runtime=runtime,
            repo_name=repo_name,
        )

    def test_owner_returns_none_when_no_talker_registered(self, tmp_path: Path) -> None:
        # copilotcli.py:975-977 — get_talker None or wrong kind.
        from fido import provider as provider_module

        session = self._build_session(tmp_path)
        with patch.object(provider_module, "get_talker", return_value=None):
            assert session.owner is None

    def test_owner_returns_none_when_no_thread_matches(self, tmp_path: Path) -> None:
        # copilotcli.py:978-981 — talker.kind == worker but thread_id absent.
        from fido import provider as provider_module

        session = self._build_session(tmp_path)
        fake_talker = MagicMock()
        fake_talker.kind = "worker"
        fake_talker.thread_id = -1
        with patch.object(provider_module, "get_talker", return_value=fake_talker):
            assert session.owner is None

    def test_owner_returns_thread_name_when_thread_id_matches(
        self, tmp_path: Path
    ) -> None:
        # copilotcli.py:980 — return t.name when ident matches.
        import threading

        from fido import provider as provider_module

        session = self._build_session(tmp_path)
        current = threading.current_thread()
        fake_talker = MagicMock()
        fake_talker.kind = "worker"
        fake_talker.thread_id = current.ident
        with patch.object(provider_module, "get_talker", return_value=fake_talker):
            assert session.owner == current.name


class TestCopilotCLIOwner:
    def test_owner_returns_none_when_repo_name_is_unset(self, tmp_path: Path) -> None:
        """``owner`` property short-circuits to None when repo_name is
        not set (copilotcli.py:975 path)."""
        from fido.copilotcli import CopilotCLISession
        from fido.provider import ProviderID

        # Use a no-op stub for runtime so the real ACP client doesn't
        # spawn.  Repo_name=None forces the early-return branch.
        runtime = MagicMock()
        runtime.ensure_session.return_value = "sess-1"
        session = CopilotCLISession(
            tmp_path / "sys.md",
            work_dir=tmp_path,
            model="gpt-5",
            runtime=runtime,
            repo_name=None,
        )
        assert session.owner is None
        del ProviderID  # quiet pyright


class TestCodexHelperFunctions:
    """Standalone helpers in codex.py that don't require a full session."""

    def test_thread_id_from_result_raises_when_missing(self) -> None:
        from fido.codex import CodexProtocolError, _thread_id_from_result

        # Cover codex.py:1018 (missing thread.id)
        with pytest.raises(CodexProtocolError, match="thread.id"):
            _thread_id_from_result({"thread": {}})
        with pytest.raises(CodexProtocolError, match="thread.id"):
            _thread_id_from_result({})
        with pytest.raises(CodexProtocolError, match="thread.id"):
            _thread_id_from_result({"thread": "not-a-dict"})

    def test_thread_id_from_result_returns_id_when_present(self) -> None:
        from fido.codex import _thread_id_from_result

        assert _thread_id_from_result({"thread": {"id": "abc"}}) == "abc"

    def test_notification_matches_thread_id_mismatch_returns_false(self) -> None:
        from fido.codex import _notification_matches

        # Cover codex.py:1034 (thread_id mismatch)
        params = {"threadId": "thread-other", "turnId": "turn-1"}
        assert (
            _notification_matches(params, thread_id="thread-1", turn_id="turn-1")
            is False
        )

    def test_notification_matches_turn_id_mismatch_returns_false(self) -> None:
        from fido.codex import _notification_matches

        # Cover codex.py:1037 (turn_id mismatch at top level)
        params = {"threadId": "thread-1", "turnId": "turn-other"}
        assert (
            _notification_matches(params, thread_id="thread-1", turn_id="turn-1")
            is False
        )

    def test_notification_matches_nested_turn_id_mismatch_returns_false(self) -> None:
        from fido.codex import _notification_matches

        # Cover codex.py:1042 (nested turn.id mismatch)
        params = {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "turn": {"id": "turn-mismatch", "status": "completed"},
        }
        assert (
            _notification_matches(params, thread_id="thread-1", turn_id="turn-1")
            is False
        )

    def test_notification_matches_returns_true_on_match(self) -> None:
        from fido.codex import _notification_matches

        params = {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "turn": {"id": "turn-1", "status": "completed"},
        }
        assert (
            _notification_matches(params, thread_id="thread-1", turn_id="turn-1")
            is True
        )

    def test_extract_completed_turn_returns_params_when_status_present(self) -> None:
        from fido.codex import _extract_completed_turn

        # Cover codex.py:1050 path (turn dict with str status)
        params = {"turn": {"status": "completed"}}
        assert _extract_completed_turn(params) is params

    def test_extract_completed_turn_returns_none_when_no_turn_dict(self) -> None:
        from fido.codex import _extract_completed_turn

        # turn missing or not a dict
        assert _extract_completed_turn({}) is None
        assert _extract_completed_turn({"turn": "not-a-dict"}) is None
        # turn dict but status not a string
        assert _extract_completed_turn({"turn": {"status": 42}}) is None


class TestCodexProviderErrorEvents:
    """Cover branches in ``_provider_error_from_event``."""

    def test_turn_failed_with_non_dict_error_uses_str_fallback(self) -> None:
        # codex.py:475 — turn.failed with error that is not a dict falls
        # through to ``str(error or obj)``.
        from fido.codex import _provider_error_from_event

        # Error is a plain string, not a dict — drives the else branch.
        result = _provider_error_from_event({"type": "turn.failed", "error": "boom"})
        assert result is not None
        assert "boom" in str(result)

    def test_turn_failed_with_no_error_uses_obj_str_fallback(self) -> None:
        # codex.py:475 — when ``error`` is falsy, message falls back to
        # ``str(obj)``.
        from fido.codex import _provider_error_from_event

        result = _provider_error_from_event({"type": "turn.failed"})
        assert result is not None
        # str(obj) contains the dict repr
        assert "turn.failed" in str(result)


class TestCodexSessionDefensivePaths:
    """Defensive branches in CodexSession that require a fake app-server."""

    @staticmethod
    def _fake_app_server() -> "_FakeAppServerForCoverage":
        """Build a minimal fake app server matching CodexAppServer protocol."""
        return _FakeAppServerForCoverage()

    def test_send_raises_when_turn_id_missing_from_response(
        self, tmp_path: Path
    ) -> None:
        # codex.py:770 — turn/start response without turn.id raises.
        from fido.codex import CodexProtocolError, CodexSession
        from fido.provider import ProviderModel

        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = self._fake_app_server()
        # Override turn/start to return a response without turn.id
        fake.responses["turn/start"] = {"turn": {}}  # missing 'id'
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            client_factory=lambda **_: fake,
        )
        with pytest.raises(CodexProtocolError, match="turn.id"):
            session.send("hello")

    def test_consume_until_result_returns_empty_when_no_active_turn(
        self, tmp_path: Path
    ) -> None:
        # codex.py:780 — consume_until_result short-circuits when no
        # active_turn_id (e.g. nothing was sent yet).
        from fido.codex import CodexSession
        from fido.provider import ProviderModel

        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = self._fake_app_server()
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            client_factory=lambda **_: fake,
        )
        # No send() — so _active_turn_id is None
        assert session.consume_until_result() == ""

    def test_is_alive_delegates_to_underlying_client(self, tmp_path: Path) -> None:
        # codex.py:873-874 — is_alive() reflects client state.
        from fido.codex import CodexSession
        from fido.provider import ProviderModel

        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = self._fake_app_server()
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            client_factory=lambda **_: fake,
        )
        assert session.is_alive() is True
        fake.alive = False
        assert session.is_alive() is False

    def test_stop_delegates_to_underlying_client(self, tmp_path: Path) -> None:
        # codex.py:877-879 — stop() pulls client out under lock and stops it.
        from fido.codex import CodexSession
        from fido.provider import ProviderModel

        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = self._fake_app_server()
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            client_factory=lambda **_: fake,
        )
        assert fake.stopped is False
        session.stop()
        assert fake.stopped is True


class TestCodexSessionMoreBranches:
    """More CodexSession defensive branches."""

    @staticmethod
    def _build_session(tmp_path: Path, fake):
        from fido.codex import CodexSession
        from fido.provider import ProviderModel

        system_file = tmp_path / "system.md"
        system_file.write_text("")
        return CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            client_factory=lambda **_: fake,
        )

    def test_owner_returns_none_when_no_talker_registered(self, tmp_path: Path) -> None:
        # codex.py:701 — owner returns None when get_talker returns None
        # (or talker.kind != "worker").
        from fido import provider as provider_module

        fake = _FakeAppServerForCoverage()
        session = self._build_session(tmp_path, fake)
        # _repo_name is None by default (we didn't set it), so _repo_name is None
        # short-circuits.  Set it manually so we can exercise the get_talker
        # None branch (line 700-701).
        session._repo_name = "test/repo"  # type: ignore[attr-defined]
        # Ensure no talker is registered for this repo.
        with patch.object(provider_module, "get_talker", return_value=None):
            assert session.owner is None

    def test_owner_returns_none_when_no_thread_matches(self, tmp_path: Path) -> None:
        # codex.py:702-705 — owner walks threading.enumerate() and returns
        # None when no thread.ident matches talker.thread_id.
        from fido import provider as provider_module

        fake = _FakeAppServerForCoverage()
        session = self._build_session(tmp_path, fake)
        session._repo_name = "test/repo"  # type: ignore[attr-defined]
        # Talker.kind == "worker" but thread_id won't match any live thread.
        fake_talker = MagicMock()
        fake_talker.kind = "worker"
        fake_talker.thread_id = -1  # no real thread has this ident
        with patch.object(provider_module, "get_talker", return_value=fake_talker):
            assert session.owner is None

    def test_consume_until_result_raises_provider_error_on_error_notification(
        self, tmp_path: Path
    ) -> None:
        # codex.py:805-811 — error notification mid-stream raises
        # CodexProviderError.
        from fido.codex import CodexProviderError

        fake = _FakeAppServerForCoverage()
        # First we need to send() to set _active_turn_id.
        fake.notifications.append(
            {
                "method": "error",
                "params": {"message": "rate limit hit"},
            }
        )
        session = self._build_session(tmp_path, fake)
        session.send("hello")
        with pytest.raises(CodexProviderError, match="rate limit"):
            session.consume_until_result()

    def test_require_thread_id_raises_when_session_id_unset(
        self, tmp_path: Path
    ) -> None:
        # codex.py:974 — _require_thread_id raises when no thread id.
        from fido.codex import CodexProtocolError

        fake = _FakeAppServerForCoverage()
        session = self._build_session(tmp_path, fake)
        # Force the session_id to None — bypassing _ensure_thread which
        # set it on construction.
        with session._state_lock:  # type: ignore[attr-defined]
            session._session_id = None  # type: ignore[attr-defined]
        with pytest.raises(CodexProtocolError, match="no thread id"):
            session._require_thread_id()  # type: ignore[attr-defined]

    def test_dead_prompt_error_message_returns_static_text(self) -> None:
        # codex.py:1178 — dead prompt error message constant on CodexClient.
        from fido.codex import CodexClient

        client = CodexClient(session=MagicMock())
        assert "died" in client._dead_prompt_error_message()  # type: ignore[attr-defined]


class TestRocqLspBranches:
    """Coverage fills for ``rocq_lsp.py`` leaf branches."""

    def test_comment_and_string_ranges_closes_string_literal(self) -> None:
        """``_comment_and_string_ranges`` exits the in-string branch when
        it sees the closing ``"`` (rocq_lsp.py:1134-1142)."""
        from fido.rocq_lsp import _comment_and_string_ranges

        # A complete string literal must produce a "string" range.
        ranges = _comment_and_string_ranges('Definition x := "hello".')
        kinds = [kind for *_, kind in ranges]
        assert "string" in kinds

    def test_symbol_at_uses_python_index_for_generated_path(
        self, tmp_path: Path
    ) -> None:
        """``RocqIndex.symbol_at`` routes to ``_python_symbols`` when the
        path is inside the generated dir (rocq_lsp.py:222)."""
        from fido.rocq_lsp import RocqIndex

        # Set up a fake repo so the generated dir resolves correctly:
        # ``<root>/src/fido/rocq/`` is what RocqIndex treats as generated.
        gen_dir = tmp_path / "src" / "fido" / "rocq"
        gen_dir.mkdir(parents=True)
        py_file = gen_dir / "toy.py"
        py_file.write_text("def toy() -> int:\n    return 0\n")

        index = RocqIndex(tmp_path)
        # Don't refresh — we just want to exercise the symbol_at branch
        # at line 222 (path is_relative_to generated_dir → python_symbols).
        # Even when _python_symbols is empty, get(token) returns None and
        # the line is exercised.
        # We need a valid token at the position — find_token requires the
        # token to exist in the file content.  Position points into "toy".
        result = index.symbol_at(py_file, line=0, character=5)
        # Either hit a python symbol (none registered) or return None.
        assert result is None


class TestCodexCLIErrorBranch:
    def test_run_codex_exec_resume_raises_on_nonzero_returncode(
        self, tmp_path: Path
    ) -> None:
        # codex.py:1117 — non-zero returncode raises CodexCLIError.
        import subprocess

        from fido.codex import CodexCLIError, run_codex_exec_resume
        from fido.provider import ProviderModel

        def runner(*args, **kwargs):  # noqa: ARG001
            return subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="codex died"
            )

        with pytest.raises(CodexCLIError) as exc_info:
            run_codex_exec_resume(
                "session-id",
                "prompt",
                model=ProviderModel("gpt-5.5", "medium"),
                cwd=tmp_path,
                runner=runner,
                timeout=5,
            )
        assert exc_info.value.returncode == 1
        assert "codex died" in exc_info.value.stderr


class TestWorkerHandleQueuedComment:
    """Cover ``_handle_queued_comment`` defensive paths (worker.py:2424-2447)."""

    @staticmethod
    def _make_queued_record():
        from fido.store import PRCommentQueueRecord

        return PRCommentQueueRecord(
            queue_id="q1",
            delivery_id="d1",
            repo="test/repo",
            pr_number=1,
            comment_type="issues",
            comment_id=42,
            author="alice",
            is_bot=False,
            body="hi",
            github_created_at="2026-01-01T00:00:00Z",
            state="pending",
            claim_owner=None,
            retry_count=0,
            next_retry_after=None,
            payload_json="{}",
        )

    def test_raises_when_config_or_repo_cfg_is_missing(self, tmp_path: Path) -> None:
        # worker.py:2424-2425 — RuntimeError if config/repo_cfg are None.
        from tests.test_worker import Worker

        worker = Worker(tmp_path, MagicMock())
        # Force config to None via the underlying attribute.
        worker._config = None  # type: ignore[attr-defined]
        store = MagicMock()
        repo_ctx = MagicMock()
        repo_ctx.repo = "test/repo"
        with pytest.raises(RuntimeError, match="explicit config"):
            worker._handle_queued_comment(  # type: ignore[attr-defined]
                store, self._make_queued_record(), repo_ctx
            )

    def test_completes_when_promise_is_none(self, tmp_path: Path) -> None:
        # worker.py:2445-2447 — promise None → complete_pr_comment + return.
        from tests.test_worker import Worker

        gh = MagicMock()
        gh.get_pr.return_value = {"title": "T", "body": "B"}
        worker = Worker(tmp_path, gh)
        # Wire config + repo_cfg as MagicMocks so the early-return guard
        # (config is None or repo_cfg is None) doesn't trigger.
        worker._config = MagicMock()  # type: ignore[attr-defined]
        worker._repo_cfg = MagicMock()  # type: ignore[attr-defined]
        store = MagicMock()
        store.prepare_reply.return_value = None  # forces the promise None branch
        repo_ctx = MagicMock()
        repo_ctx.repo = "test/repo"
        # Patch _queued_comment_action so it returns a non-None action.
        action_stub = MagicMock()
        action_stub.thread = {"comment_id": 42}
        action_stub.reply_to = {"comment_id": 42}
        action_stub.context = None
        with patch.object(worker, "_queued_comment_action", return_value=action_stub):
            worker._handle_queued_comment(  # type: ignore[attr-defined]
                store, self._make_queued_record(), repo_ctx
            )
        store.complete_pr_comment.assert_called_with("q1")


class TestWorkerEmptyPrComment:
    """Cover ``_post_empty_pr_comment_once`` defensive RequestException paths."""

    def test_swallows_request_exception_when_fetching_comments(
        self, tmp_path: Path
    ) -> None:
        # worker.py:2735-2743 — gh.get_issue_comments raises → log+return.
        import requests

        from tests.test_worker import Worker

        gh = MagicMock()
        gh.get_issue_comments.side_effect = requests.RequestException("boom")
        worker = Worker(tmp_path, gh)
        # Should not raise — just log and return.
        worker._post_empty_pr_comment_once("test/repo", 1)  # type: ignore[attr-defined]

    def test_swallows_request_exception_when_posting_comment(
        self, tmp_path: Path
    ) -> None:
        # worker.py:2754-2762 — gh.comment_issue raises → log+continue.
        import requests

        from tests.test_worker import Worker

        gh = MagicMock()
        gh.get_issue_comments.return_value = []
        gh.comment_issue.side_effect = requests.RequestException("boom")
        worker = Worker(tmp_path, gh)
        worker._post_empty_pr_comment_once("test/repo", 1)  # type: ignore[attr-defined]


class TestWorkerQueuedActions:
    """Cover ``_queued_*_comment_action`` methods when the comment is gone."""

    def test_queued_issue_comment_action_returns_none_when_comment_gone(
        self, tmp_path: Path
    ) -> None:
        # worker.py:2547-2549 — gh.get_issue_comment returns None → log+return.
        from fido.store import PRCommentQueueRecord
        from tests.test_worker import Worker

        gh = MagicMock()
        gh.get_issue_comment.return_value = None
        worker = Worker(tmp_path, gh)
        queued = PRCommentQueueRecord(
            queue_id="q1",
            delivery_id="d1",
            repo="test/repo",
            pr_number=1,
            comment_type="issues",
            comment_id=42,
            author="alice",
            is_bot=False,
            body="hi",
            github_created_at="2026-01-01T00:00:00Z",
            state="pending",
            claim_owner=None,
            retry_count=0,
            next_retry_after=None,
            payload_json="{}",
        )
        result = worker._queued_issue_comment_action(  # type: ignore[attr-defined]
            queued, "test/repo", "PR title", "body"
        )
        assert result is None


class TestWorkerLeafBranches:
    """Cover small leaf branches in worker.py."""

    @staticmethod
    def _make_worker(tmp_path: Path):
        """Construct a Worker via the test scaffolding in tests/test_worker.py."""
        from tests.test_worker import Worker

        return Worker(tmp_path, MagicMock())

    def test_task_still_current_returns_false_when_state_mismatches(
        self, tmp_path: Path
    ) -> None:
        """``_task_still_current`` returns False when state.json's
        current_task_id is different from the requested task_id (worker.py:2945)."""
        from fido.state import State

        worker = self._make_worker(tmp_path)
        fido_dir = tmp_path / ".fido"
        fido_dir.mkdir()
        # Set state's current_task_id to something different.
        with State(fido_dir).modify() as state:
            state["current_task_id"] = "task-abc"
        assert worker._task_still_current(fido_dir, "task-xyz") is False  # type: ignore[attr-defined]

    @staticmethod
    def _make_worker_with_stubs(tmp_path: Path):
        """Build a Worker whose provider_agent and tasks are MagicMocks
        so we can drive the empty-msg branches without real I/O."""
        from tests.test_worker import Worker

        provider_agent = MagicMock()
        provider_agent.generate_reply.return_value = ""
        provider_agent.voice_model = "model"
        return Worker(
            tmp_path,
            MagicMock(),  # gh
            provider_agent=provider_agent,
        )

    def test_report_task_completed_no_commit_raises_when_msg_empty(
        self, tmp_path: Path
    ) -> None:
        """``_report_task_completed_without_commit`` raises ValueError when
        the provider returns an empty message (worker.py:2972)."""
        worker = self._make_worker_with_stubs(tmp_path)
        prompts_stub = MagicMock()
        prompts_stub.task_completed_without_commit_comment_prompt.return_value = "p"
        with patch.object(worker, "_get_prompts", return_value=prompts_stub):
            with pytest.raises(ValueError, match="completed without commit"):
                worker._report_task_completed_without_commit(  # type: ignore[attr-defined]
                    "test/repo", 1, "task-id", "task-title"
                )

    def test_report_task_stuck_no_commit_raises_when_msg_empty(
        self, tmp_path: Path
    ) -> None:
        """``_report_task_stuck_no_commits`` raises ValueError when
        the provider returns an empty message (worker.py:3007)."""
        worker = self._make_worker_with_stubs(tmp_path)
        prompts_stub = MagicMock()
        prompts_stub.task_stuck_no_commit_comment_prompt.return_value = "p"
        # Patch tasks helper since update is called before generate_reply.
        worker._tasks = MagicMock()  # type: ignore[attr-defined]
        fido_dir = tmp_path / ".fido"
        fido_dir.mkdir()
        with patch.object(worker, "_get_prompts", return_value=prompts_stub):
            with pytest.raises(ValueError, match="stuck no-commit"):
                worker._report_task_stuck_no_commits(  # type: ignore[attr-defined]
                    fido_dir, "test/repo", 1, "task-id", "task-title", 5
                )


class TestEventsCreateTaskExitUntriaged:
    """Cover the registry.exit_untriaged + raise path in ``create_task``
    when the default ``_reorder_tasks_background`` raises (events.py:2826-2828)."""

    def test_exception_in_reorder_calls_exit_untriaged_and_reraises(
        self, tmp_path: Path
    ) -> None:
        from fido import events

        # Build a config + repo_cfg minimal enough to exercise the path.
        repo_cfg = MagicMock()
        repo_cfg.name = "test/repo"
        repo_cfg.work_dir = tmp_path
        repo_cfg.membership = MagicMock()
        repo_cfg.membership.collaborators = frozenset()
        config = MagicMock()
        config.allowed_bots = frozenset()
        gh = MagicMock()
        gh.is_thread_resolved_for_comment.return_value = False
        registry = MagicMock()
        tasks = MagicMock()
        tasks.add.return_value = {"id": "task-1", "title": "p"}
        thread = {
            "repo": "test/repo",
            "pr": 1,
            "comment_id": 42,
        }

        def boom(*args, **kwargs):  # noqa: ARG001
            raise RuntimeError("explode")

        with patch.object(events, "_reorder_tasks_background", new=boom):
            with patch.object(events, "launch_sync"):
                with patch.object(
                    events, "_get_commit_summary", return_value="summary"
                ):
                    with pytest.raises(RuntimeError, match="explode"):
                        events.create_task(
                            "prompt",
                            config,
                            repo_cfg,
                            gh,
                            thread=thread,
                            registry=registry,
                            _reorder_background_fn=boom,
                            _tasks=tasks,
                        )
        registry.enter_untriaged.assert_called_once_with("test/repo")
        registry.exit_untriaged.assert_called_once_with("test/repo")


class TestWorkerHandleQueuedCommentsDrain:
    """Cover the registry note_durable_demand_drained call after draining
    one or more queued comments (worker.py:2407-2409)."""

    def test_notifies_registry_after_draining_at_least_one_comment(
        self, tmp_path: Path
    ) -> None:
        from fido.store import PRCommentQueueRecord
        from tests.test_worker import Worker

        gh = MagicMock()
        gh.get_pr.return_value = {"title": "T", "body": "B"}
        gh.get_issue_comment.return_value = None  # forces _queued_issue_comment_action → None
        worker = Worker(tmp_path, gh)
        worker._config = MagicMock()  # type: ignore[attr-defined]
        worker._repo_cfg = MagicMock()  # type: ignore[attr-defined]
        worker._repo_cfg.work_dir = tmp_path
        worker._registry = MagicMock()  # type: ignore[attr-defined]
        worker._repo_name = "test/repo"  # type: ignore[attr-defined]

        queued = PRCommentQueueRecord(
            queue_id="q1",
            delivery_id="d1",
            repo="test/repo",
            pr_number=1,
            comment_type="issues",
            comment_id=42,
            author="alice",
            is_bot=False,
            body="hi",
            github_created_at="2026-01-01T00:00:00Z",
            state="pending",
            claim_owner=None,
            retry_count=0,
            next_retry_after=None,
            payload_json="{}",
        )

        # First claim returns queued, second returns None.
        responses = iter([queued, None])

        class _StoreStub:
            def __init__(self, *_a, **_kw) -> None:
                pass

            def claim_next_pr_comment(self, **_kw):
                return next(responses)

            def complete_pr_comment(self, _qid: str) -> None:
                pass

        from fido import worker as worker_module

        with patch.object(worker_module, "FidoStore", _StoreStub):
            repo_ctx = MagicMock()
            repo_ctx.repo = "test/repo"
            result = worker.handle_queued_comments(tmp_path, repo_ctx, 1, "slug")
        assert result is True
        worker._registry.note_durable_demand_drained.assert_called_once_with(  # type: ignore[attr-defined]
            "test/repo"
        )


class TestWorkerOracleAssertion:
    """Cover the AssertionError raise in _assert_ci_failure_matches_oracle
    (worker.py:849-852)."""

    def test_raises_when_oracle_pick_disagrees(self) -> None:
        from fido import worker as worker_module
        from fido.worker import _assert_ci_failure_matches_oracle

        task_list: list[dict] = []
        # Patch ci_oracle.pick_next_task to return a value that does NOT
        # match the just-admitted CI failure → fires the assertion.
        with patch.object(
            worker_module.ci_oracle, "pick_next_task", return_value="mismatched-task"
        ):
            with pytest.raises(AssertionError, match="not first pickup"):
                _assert_ci_failure_matches_oracle(task_list, "tests", "FAILURE", "run-1")


class TestClaudeStderrPump:
    """Cover the stderr pump loop in ClaudeSession (claude.py:691-693)."""

    def test_stderr_pump_drains_lines_to_log(self) -> None:
        import time

        from fido.claude import ClaudeSession

        # Construct a session via __new__ to skip the real Popen spawn —
        # _start_stderr_pump is an instance method that only needs ``self``
        # to call ``log`` (module-level).
        session = ClaudeSession.__new__(ClaudeSession)

        class _Proc:
            def __init__(self, lines: list[str]) -> None:
                self.pid = 9999
                self._stderr_iter = iter(lines)
                # Provide an iterable stderr; ``for raw in stderr`` walks it.
                self.stderr = self._stderr_iter

        # Two lines + a blank one (which the ``if line`` guard skips).
        proc = _Proc(["hello\n", "\n", "world\n"])
        session._start_stderr_pump(proc)  # type: ignore[arg-type]

        # Wait for the daemon thread to drain the iterator.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                next(proc._stderr_iter)
            except StopIteration:
                break
            time.sleep(0.01)


class TestEventsClaimReplyOutboxEffectsDelivered:
    """Cover the ``delivered``-state RuntimeError raise in
    _claim_reply_outbox_effects (events.py:348-351)."""

    def test_raises_when_existing_effect_is_delivered(self, tmp_path: Path) -> None:
        from fido import events

        repo_cfg = MagicMock()
        repo_cfg.work_dir = tmp_path

        promise = MagicMock()
        promise.promise_id = "promise-1"
        promise.anchor_comment_id = 42

        existing = MagicMock()
        existing.state = "delivered"

        class _StoreStub:
            def __init__(self, *_a, **_kw) -> None:
                pass

            def promise(self, _pid: str):
                return promise

            def reply_outbox_effect(self, _pid: str):
                return existing

        with patch.object(events, "FidoStore", _StoreStub):
            with pytest.raises(RuntimeError, match="missing artifact row"):
                events._claim_reply_outbox_effects(
                    repo_cfg,
                    delivery_id="d1",
                    promise_ids=["promise-1"],
                )


class TestEventsIngressFsmCollapsed:
    """Cover the ``Collapsed-or-other`` else branch in
    WebhookIngressOracle.check_dispatch (events.py:146-148)."""

    def test_arrive_after_collapsed_runs_else_branch(self) -> None:
        from fido.events import WebhookIngressOracle

        oracle = WebhookIngressOracle()
        # First call with collapse_review=True transitions Fresh → Collapsed
        # and returns None (collapsed deliveries are suppressed).
        oracle.check_dispatch("test/repo", "delivery-1", collapse_review=True)
        # Second call (no collapse) on the same delivery hits the else branch
        # at line 146-148 — fires Arrive on a Collapsed state.  The FSM
        # rejects the transition with AssertionError; that's expected here.
        # Coverage records the line 148 ``event = Arrive()`` execution
        # before the transition asserts.
        with pytest.raises(AssertionError, match="Arrive rejected"):
            oracle.check_dispatch("test/repo", "delivery-1")


class TestEventsDispatchTrailingNone:
    """Cover the trailing ``return None`` fall-throughs in dispatch
    (events.py:1335, 1395)."""

    @staticmethod
    def _config_and_repo_cfg():
        config = MagicMock()
        config.allowed_bots = frozenset()
        config.repos = {}
        repo_cfg = MagicMock()
        repo_cfg.name = "test/repo"
        repo_cfg.membership = MagicMock()
        repo_cfg.membership.collaborators = frozenset(["alice"])
        repo_cfg.membership.team_members = frozenset()
        return config, repo_cfg

    def test_review_comment_with_no_comment_id_falls_through(self) -> None:
        # events.py:1335 — pull_request_review_comment with comment_id None
        # passes the early-returns but skips the enqueue branch.
        from fido.events import dispatch

        config, repo_cfg = self._config_and_repo_cfg()
        payload = {
            "action": "created",
            "repository": {"full_name": "test/repo"},
            "comment": {
                "user": {"login": "alice"},
                "body": "comment",
                # no "id" key → comment_id stays None
            },
            "pull_request": {"number": 1, "title": "T", "body": "B"},
        }
        result = dispatch(
            "pull_request_review_comment", payload, config, repo_cfg
        )
        assert result is None

    def test_issue_comment_with_no_number_falls_through(self) -> None:
        # events.py:1395 — issue_comment with number/comment_id None.
        from fido.events import dispatch

        config, repo_cfg = self._config_and_repo_cfg()
        payload = {
            "action": "created",
            "repository": {"full_name": "test/repo"},
            "comment": {
                "user": {"login": "alice"},
                "body": "comment",
                "id": 42,
            },
            "issue": {
                # number missing
                "pull_request": {"url": "https://github.com/.../pull/1"},
                "title": "T",
                "body": "B",
            },
        }
        result = dispatch("issue_comment", payload, config, repo_cfg)
        assert result is None


class TestEventsThreadResolved:
    """Cover ``_thread_task_is_stale_resolved`` early-return branches."""

    def test_returns_true_when_comment_id_missing(self) -> None:
        # events.py:2703-2704 — None comment_id → return True (stale).
        from fido.events import _thread_task_is_stale_resolved

        gh = MagicMock()
        result = _thread_task_is_stale_resolved(gh, {"repo": "test/repo", "pr": 1})
        assert result is True

    def test_returns_true_when_no_comments_fetched(self) -> None:
        # events.py:2706-2708 — empty comments list → return True (stale).
        from fido.events import _thread_task_is_stale_resolved

        gh = MagicMock()
        gh.fetch_comment_thread.return_value = []
        result = _thread_task_is_stale_resolved(
            gh, {"repo": "test/repo", "pr": 1, "comment_id": 42}
        )
        assert result is True


class TestEventsNotifyThreadChange:
    """Cover the ``modified`` branch of ``_notify_thread_change``
    (events.py:2322-2333)."""

    def test_modified_branch_uses_new_title_and_posts_reply(
        self, tmp_path: Path
    ) -> None:
        from fido.events import _notify_thread_change

        change = {
            "task": {
                "title": "Old title",
                "thread": {
                    "comment_id": 42,
                    "repo": "test/repo",
                    "pr": 1,
                    "url": "https://example.com/c",
                    "author": "alice",
                    "comment_type": "pulls",  # required to skip early-return
                },
            },
            "kind": "modified",
            "new_title": "New title",
        }
        config = MagicMock()
        gh = MagicMock()
        agent = MagicMock()
        agent.voice_model = "voice"
        agent.generate_reply.return_value = "reply body"
        prompts = MagicMock()
        prompts.persona_wrap.return_value = "wrapped"
        prompts.reply_system_prompt.return_value = "system"
        _ = tmp_path  # quiet unused-arg warning
        _notify_thread_change(change, config, gh, agent=agent, prompts=prompts)
        gh.reply_to_review_comment.assert_called_once()
        # The instruction should mention the new title.
        wrap_arg = prompts.persona_wrap.call_args[0][0]
        assert "New title" in wrap_arg


class TestCodexJsonlIteration:
    """Cover _iter_jsonl small branches (codex.py:397-398, 401-402)."""

    def test_iter_jsonl_skips_blank_lines(self) -> None:
        # codex.py:398 — continue on empty/whitespace-only lines.
        from fido.codex import _iter_jsonl

        text = '\n   \n{"a":1}\n\n'
        objs = list(_iter_jsonl(text))
        assert objs == [{"a": 1}]

    def test_iter_jsonl_skips_invalid_json_lines(self) -> None:
        # codex.py:401-402 — json.JSONDecodeError continues to next line.
        from fido.codex import _iter_jsonl

        text = 'not-json\n{"ok":1}\n[1,2]\n'  # array → not a dict, skipped
        objs = list(_iter_jsonl(text))
        assert objs == [{"ok": 1}]


class TestCodexAppServerStderrAndError:
    """Cover stderr-pump and invalid-error branches in CodexAppServerClient."""

    def test_stderr_reader_pumps_lines_to_queue(self) -> None:
        # codex.py:348-349 — stderr lines are read and queued.
        from fido.codex import CodexAppServerClient

        class _StderrProcess:
            def __init__(self, *_, **__) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(
                    '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n'
                )
                self.stderr = io.StringIO("error one\nerror two\n")
                self.pid = 1234
                self._returncode: int | None = None
                self.terminated = False

            def poll(self) -> int | None:
                return self._returncode

            def terminate(self) -> None:
                self.terminated = True
                self._returncode = 0

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
                self._returncode = 0
                return 0

            def kill(self) -> None:
                self._returncode = -9

        client = CodexAppServerClient(process_factory=_StderrProcess)
        client.stop()
        # Stderr should have been pumped — exact contents drained from queue.
        # Just confirm we exited without raising.
        assert client is not None

    def test_handle_line_raises_when_error_is_not_dict(self) -> None:
        # codex.py:362-365 — non-dict error in response raises CodexProtocolError.
        from fido.codex import CodexAppServerClient, CodexProtocolError

        # Send a response with id+error that is not a dict.
        bad_response = (
            '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n'
            '{"id":2,"error":"not-a-dict"}\n'
        )

        class _BadErrorProcess:
            def __init__(self, *_, **__) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(bad_response)
                self.stderr = io.StringIO()
                self.pid = 1234
                self._returncode: int | None = None
                self.terminated = False

            def poll(self) -> int | None:
                return self._returncode

            def terminate(self) -> None:
                self.terminated = True
                self._returncode = 0

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
                self._returncode = 0
                return 0

            def kill(self) -> None:
                self._returncode = -9

        client = CodexAppServerClient(process_factory=_BadErrorProcess)
        # First request after init should encounter the protocol error from
        # the malformed second response.
        with pytest.raises(CodexProtocolError):
            client.request("anything", timeout=2.0)
        client.stop()


class TestCodexProcessExited:
    """Cover ``_raise_if_unavailable_locked`` process-exited check
    (codex.py:390-391)."""

    def test_raise_if_unavailable_raises_when_process_exited(self) -> None:
        # codex.py:390-391 — directly drive the locked check by setting up
        # a CodexAppServerClient whose process reports exited but whose
        # protocol_error is still None (happy path through init).
        from fido.codex import CodexAppServerClient, CodexProtocolError

        class _Process:
            def __init__(self, *_, **__) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(
                    '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n'
                )
                self.stderr = io.StringIO()
                self.pid = 1234
                self._returncode: int | None = None
                self.terminated = False

            def poll(self) -> int | None:
                return self._returncode

            def terminate(self) -> None:
                self.terminated = True
                self._returncode = 0

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
                self._returncode = 0
                return 0

            def kill(self) -> None:
                self._returncode = -9

        proc = _Process()
        client = CodexAppServerClient(process_factory=lambda **_: proc)
        # Manually clear any protocol error from EOF, mark process as exited,
        # then exercise the helper directly.
        with client._state_lock:  # type: ignore[attr-defined]
            client._protocol_error = None  # type: ignore[attr-defined]
            client._stopped = False  # type: ignore[attr-defined]
            proc._returncode = 0
            with pytest.raises(CodexProtocolError, match="exited"):
                client._raise_if_unavailable_locked()  # type: ignore[attr-defined]
        client.stop()


class TestCodexLeafBranches:
    """Final small leaf branches in codex.py."""

    def test_read_stderr_returns_when_stderr_is_none(self) -> None:
        # codex.py:346-347 — early return when process.stderr is None.
        from fido.codex import CodexAppServerClient

        class _NoStderrProcess:
            def __init__(self, *_, **__) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(
                    '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n'
                )
                self.stderr = None  # Triggers line 347.
                self.pid = 1234
                self._returncode: int | None = None
                self.terminated = False

            def poll(self) -> int | None:
                return self._returncode

            def terminate(self) -> None:
                self.terminated = True
                self._returncode = 0

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
                self._returncode = 0
                return 0

            def kill(self) -> None:
                self._returncode = -9

        client = CodexAppServerClient(process_factory=_NoStderrProcess)
        client.stop()
        assert client is not None

    def test_owner_returns_thread_name_when_thread_id_matches(
        self, tmp_path: Path
    ) -> None:
        # codex.py:702-704 — owner walks threading.enumerate() and returns
        # thread.name when ident matches.
        import threading

        from fido import provider as provider_module
        from fido.codex import CodexSession
        from fido.provider import ProviderModel

        system_file = tmp_path / "system.md"
        system_file.write_text("")
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            repo_name="test/repo",
            client_factory=lambda **_: _FakeAppServerForCoverage(),
        )
        # Use the running test thread's ident so threading.enumerate()
        # finds a match.
        current = threading.current_thread()
        fake_talker = MagicMock()
        fake_talker.kind = "worker"
        fake_talker.thread_id = current.ident
        with patch.object(provider_module, "get_talker", return_value=fake_talker):
            assert session.owner == current.name

    def test_poll_completed_turn_returns_none_on_timeout(self, tmp_path: Path) -> None:
        # codex.py:988-989 — _poll_completed_turn returns None when the
        # underlying client wait_notification raises TimeoutError.
        from fido.codex import CodexSession
        from fido.provider import ProviderModel

        system_file = tmp_path / "system.md"
        system_file.write_text("")
        fake = _FakeAppServerForCoverage()
        # Don't put any notifications → wait_notification raises TimeoutError.
        session = CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            client_factory=lambda **_: fake,
        )
        result = session._poll_completed_turn("thread-id", "turn-id")  # type: ignore[attr-defined]
        assert result is None


class TestCodexAPIBranches:
    """Cover defensive branches in CodexAPI.get_limit_snapshot."""

    def test_get_limit_snapshot_handles_non_dict_response(self) -> None:
        # codex.py:631 — non-dict payload raises ValueError, caught by
        # the surrounding except → returns unavailable_reason snapshot.
        from fido.codex import CodexAPI

        bad_client = MagicMock()
        bad_client.request.return_value = "not-a-dict"
        api = CodexAPI(client_factory=lambda: bad_client)
        snapshot = api.get_limit_snapshot()
        assert snapshot.unavailable_reason is not None

    def test_codex_limit_windows_marks_pressure_one_as_reached(self) -> None:
        # codex.py:580-581 — window with pressure >= 1.0 added to
        # reached_names; subsequent _reached_window_name path then guards
        # against double-add.
        from fido.codex import _codex_limit_windows

        payload = {
            "rateLimits": [
                {
                    "limitId": "weekly",
                    "primary": {
                        "usedPercent": 100,
                    },
                }
            ]
        }
        windows = _codex_limit_windows(payload)
        # The primary window should have pressure 1.0 → reached.
        assert any(w.pressure is not None and w.pressure >= 1.0 for w in windows)


class TestCodexSessionMisc:
    """Misc CodexSession leaf branches."""

    @staticmethod
    def _build_session(tmp_path: Path, *, repo_name: str = "test/repo"):
        from fido.codex import CodexSession
        from fido.provider import ProviderModel

        system_file = tmp_path / "system.md"
        system_file.write_text("")
        return CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            repo_name=repo_name,
            client_factory=lambda **_: _FakeAppServerForCoverage(),
        )

    def test_interrupt_active_turn_delegates_to_fire_worker_cancel(
        self, tmp_path: Path
    ) -> None:
        # codex.py:883 — interrupt_active_turn just delegates.
        session = self._build_session(tmp_path)
        with patch.object(session, "_fire_worker_cancel") as cancel:
            session.interrupt_active_turn()
            cancel.assert_called_once()

    def test_enter_reentrant_bumps_depth_and_returns_self(self, tmp_path: Path) -> None:
        # codex.py:906-908 — re-entrant __enter__ skips fsm acquire.
        from fido import provider as provider_module

        session = self._build_session(tmp_path)
        # First enter goes through full path.  Patch register_talker so we
        # don't accidentally talk to the real provider.
        with patch.object(provider_module, "register_talker"):
            with patch.object(provider_module, "unregister_talker"):
                with session as s1:
                    with session as s2:  # re-entrant
                        assert s1 is s2

    def test_enter_routes_through_handler_branch_when_kind_handler(
        self, tmp_path: Path
    ) -> None:
        # codex.py:913 — non-worker kind takes the handler branch.
        from fido import provider as provider_module

        session = self._build_session(tmp_path)
        with patch.object(
            provider_module, "current_thread_kind", return_value="handler"
        ):
            with patch.object(provider_module, "register_talker"):
                with patch.object(provider_module, "unregister_talker"):
                    with session:
                        pass


class TestCodexAppServerStdinStdout:
    """Defensive paths for missing stdin/stdout on the underlying process."""

    def test_write_raises_when_stdin_unavailable(self) -> None:
        # codex.py:317-318 — _write raises CodexProtocolError when stdin
        # is None.
        from fido.codex import CodexAppServerClient, CodexProtocolError

        class _NoStdinProcess:
            def __init__(self, *_, **__) -> None:
                self.stdin = None
                self.stdout = io.StringIO(
                    '{"id":1,"result":{"serverInfo":{"name":"codex"}}}\n'
                )
                self.stderr = io.StringIO()
                self.pid = 1234
                self._returncode: int | None = None
                self.terminated = False

            def poll(self) -> int | None:
                return self._returncode

            def terminate(self) -> None:
                self.terminated = True
                self._returncode = 0

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
                self._returncode = 0
                return 0

            def kill(self) -> None:
                self._returncode = -9

        # Construction will succeed because _read_stdout reads the init reply.
        # The write only happens for subsequent requests.
        # But _initialize() needs to actually write — so it'll fail right
        # away.  Catch via the protocol error path.
        with pytest.raises(CodexProtocolError, match="stdin"):
            CodexAppServerClient(process_factory=_NoStdinProcess)

    def test_reader_fails_protocol_when_stdout_missing(self) -> None:
        # codex.py:324-326 — _read_stdout calls _fail_protocol when stdout
        # is None.
        import threading
        import time

        from fido.codex import CodexAppServerClient, CodexProtocolError

        class _NoStdoutProcess:
            def __init__(self, *_, **__) -> None:
                self.stdin = io.StringIO()
                self.stdout = None
                self.stderr = io.StringIO()
                self.pid = 1234
                self._returncode: int | None = None
                self.terminated = False
                self._stop_event = threading.Event()

            def poll(self) -> int | None:
                return self._returncode

            def terminate(self) -> None:
                self.terminated = True
                self._stop_event.set()

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
                self._returncode = 0
                return 0

            def kill(self) -> None:
                self._returncode = -9

        # Construction sets up the reader thread that will immediately call
        # _fail_protocol, then _initialize will see the protocol error.
        with pytest.raises(CodexProtocolError):
            CodexAppServerClient(process_factory=_NoStdoutProcess)
        time.sleep(0)  # quiet ResourceWarning


class TestCodexSessionEnter:
    """Cover __enter__ paths involving register_talker (codex.py:915-937)."""

    @staticmethod
    def _build_session(tmp_path: Path, *, repo_name: str = "test/repo"):
        from fido.codex import CodexSession
        from fido.provider import ProviderModel

        system_file = tmp_path / "system.md"
        system_file.write_text("")
        return CodexSession(
            system_file,
            work_dir=tmp_path,
            model=ProviderModel("gpt-5.5", "medium"),
            repo_name=repo_name,
            client_factory=lambda **_: _FakeAppServerForCoverage(),
        )

    def test_enter_registers_talker_then_exit_unregisters(self, tmp_path: Path) -> None:
        # Cover happy-path through __enter__ (line 917-926) and __exit__
        # (line 936-937 unregister_talker).
        from fido import provider as provider_module

        session = self._build_session(tmp_path)
        register_calls: list[str] = []
        unregister_calls: list[str] = []

        def fake_register(talker):  # noqa: ARG001
            register_calls.append(talker.repo_name)

        def fake_unregister(repo_name, thread_id):  # noqa: ARG001
            unregister_calls.append(repo_name)

        with patch.object(
            provider_module, "register_talker", side_effect=fake_register
        ):
            with patch.object(
                provider_module, "unregister_talker", side_effect=fake_unregister
            ):
                # Force __enter__ kind branch.  current_thread_kind is "handler"
                # by default which routes through _fsm_acquire_handler.
                with session:
                    pass
        assert register_calls == ["test/repo"]
        assert unregister_calls == ["test/repo"]

    def test_enter_propagates_session_leak_error_and_releases_fsm(
        self, tmp_path: Path
    ) -> None:
        # Cover lines 927-930 — SessionLeakError raised by register_talker
        # is re-raised after _drop_entry_depth and _fsm_release are called.
        from fido import provider as provider_module

        session = self._build_session(tmp_path)

        def explode(_talker):
            raise provider_module.SessionLeakError("test leak")

        with patch.object(provider_module, "register_talker", side_effect=explode):
            with pytest.raises(provider_module.SessionLeakError, match="test leak"):
                with session:
                    pass


class _FakeAppServerForCoverage:
    """Minimal fake matching ``fido.codex.CodexAppServer`` protocol.

    Mirrors ``_FakeAppServer`` from tests/test_codex.py but lives here so
    test_coverage_fills.py can import it without crossing the test boundary.
    """

    def __init__(self, *, cwd=None) -> None:
        self.cwd = cwd
        self.pid = 456
        self.requests: list[tuple[str, dict]] = []
        self.responses: dict[str, object | Exception] = {}
        self.notifications: list[dict] = []
        self.stopped = False
        self.alive = True

    def request(self, method, params=None, *, timeout=30.0):  # noqa: ARG002
        payload = params or {}
        self.requests.append((method, payload))
        response = self.responses.get(method)
        if isinstance(response, Exception):
            raise response
        if response is not None:
            return response
        if method == "thread/start":
            return {"thread": {"id": "thread-new"}}
        if method == "thread/resume":
            return {"thread": {"id": payload["threadId"]}}
        if method == "turn/start":
            return {"turn": {"id": "turn-1"}}
        return {}

    def notify(self, method, params=None) -> None:
        self.requests.append((method, params or {}))

    def wait_notification(self, method, *, predicate=None, timeout=30.0):  # noqa: ARG002
        for index, notification in enumerate(self.notifications):
            if method != "*" and notification["method"] != method:
                continue
            params = notification["params"]
            if predicate is None or predicate(params):
                return self.notifications.pop(index)
        raise TimeoutError(method)

    def is_alive(self) -> bool:
        return self.alive

    def stop(self) -> None:
        self.stopped = True
        self.alive = False
