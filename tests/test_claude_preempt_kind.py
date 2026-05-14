"""Tests for thread-kind helpers in :mod:`fido.provider`.

The kind-aware preempt path that used to live in ``ClaudeSession.prompt``
(#637) was removed in #955 — preemption now fires synchronously in the HTTP
handler via :meth:`~fido.provider.OwnedSession.preempt_worker` before the
background thread spawns, and :meth:`~fido.provider.OwnedSession.hold_for_handler`
provides a secondary safety net before lock acquisition.  Tests for those
paths live in ``test_claude_hold_for_handler.py`` and ``test_server.py``.
"""

from fido import provider
from fido.provider import ThreadKind


def test_set_thread_kind_roundtrip() -> None:
    provider.set_thread_kind(None)
    assert provider.current_thread_kind() == "worker"
    provider.set_thread_kind(ThreadKind.WEBHOOK)
    assert provider.current_thread_kind() == "webhook"
    provider.set_thread_kind(ThreadKind.WORKER)
    assert provider.current_thread_kind() == "worker"
    provider.set_thread_kind(ThreadKind.BACKGROUND)
    assert provider.current_thread_kind() == "background"
    provider.set_thread_kind(None)
    assert provider.current_thread_kind() == "worker"


def test_try_preempt_worker_webhook_preempts_worker() -> None:
    """A real webhook caller preempts a running worker."""
    from datetime import datetime, timezone

    repo = "owner/repo"
    talker = provider.SessionTalker(
        repo_name=repo,
        thread_id=42,
        kind=ThreadKind.WORKER,
        description="long worker turn",
        subprocess_pid=1,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    try:
        provider.register_talker(talker)
        provider.set_thread_kind(ThreadKind.WEBHOOK)
        cancelled: list[bool] = []
        preempted, current_kind = provider.try_preempt_worker(
            repo, lambda: cancelled.append(True)
        )
        assert preempted is True
        assert current_kind == "worker"
        assert cancelled == [True]
    finally:
        provider.unregister_talker(repo, 42)
        provider.set_thread_kind(None)


def test_try_preempt_worker_background_does_not_preempt() -> None:
    """Background callers (e.g. the rescope thread, #1711) do not
    preempt a running worker — that's the whole point of the third
    kind.  Without this, every rescope iteration would cancel the
    worker mid-flight and livelock long worker turns."""
    from datetime import datetime, timezone

    repo = "owner/repo"
    talker = provider.SessionTalker(
        repo_name=repo,
        thread_id=42,
        kind=ThreadKind.WORKER,
        description="long worker turn",
        subprocess_pid=1,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    try:
        provider.register_talker(talker)
        provider.set_thread_kind(ThreadKind.BACKGROUND)
        cancelled: list[bool] = []
        preempted, current_kind = provider.try_preempt_worker(
            repo, lambda: cancelled.append(True)
        )
        assert preempted is False
        assert current_kind == "worker"
        assert cancelled == []
    finally:
        provider.unregister_talker(repo, 42)
        provider.set_thread_kind(None)


def test_try_preempt_worker_background_holder_not_preempted_by_webhook() -> None:
    """A real webhook caller does not preempt a background holder
    (e.g. an in-flight rescope iteration).  Same shape as the
    no-preempt-on-webhook protection from #955 — kind != \"worker\"
    means \"don't preempt me\"."""
    from datetime import datetime, timezone

    repo = "owner/repo"
    talker = provider.SessionTalker(
        repo_name=repo,
        thread_id=42,
        kind=ThreadKind.BACKGROUND,
        description="rescope iteration in flight",
        subprocess_pid=1,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    try:
        provider.register_talker(talker)
        provider.set_thread_kind(ThreadKind.WEBHOOK)
        cancelled: list[bool] = []
        preempted, current_kind = provider.try_preempt_worker(
            repo, lambda: cancelled.append(True)
        )
        assert preempted is False
        assert current_kind == "background"
        assert cancelled == []
    finally:
        provider.unregister_talker(repo, 42)
        provider.set_thread_kind(None)
