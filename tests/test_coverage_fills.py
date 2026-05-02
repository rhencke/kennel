"""Targeted tests filling coverage gaps that survived after the
old test_server::TestProcessAction tests were removed (the deletions
landed because those tests asserted a now-obsolete synchronous-reply-
from-webhook contract — see PR #1254).

Each test in this file is small and exercises one specific branch
that no other test currently covers.  Tests are grouped by source
module.
"""

from __future__ import annotations

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
        from fido.rocq_runtime import _RocqNumericDomainError, _rocq_positive_key

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
            type(repo.provider_status), "paused", new_callable=lambda: True  # type: ignore[union-attr]
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
        result = _materialize_rescope_oracle_result(
            [oracle_id], rows, tasks_by_id
        )
        assert result[0]["status"] == "blocked"


# ---------------------------------------------------------------------------
# copilotcli.py — owner property fallback
# ---------------------------------------------------------------------------


class TestTasksAdd:
    def test_add_dedups_on_comment_id_and_merges_lineage(
        self, tmp_path: Path
    ) -> None:
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
                "UPDATE pr_comment_queue SET state = 'completed'"
                " WHERE queue_id = ?",
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

    def test_pr_comment_queue_in_progress_update_path(
        self, tmp_path: Path
    ) -> None:
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
                "UPDATE pr_comment_queue SET state = 'in_progress'"
                " WHERE queue_id = ?",
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


class TestCopilotCLIOwner:
    def test_owner_returns_none_when_repo_name_is_unset(
        self, tmp_path: Path
    ) -> None:
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
