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
