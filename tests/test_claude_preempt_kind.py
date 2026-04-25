"""Tests for thread-kind helpers in :mod:`fido.provider`.

The kind-aware preempt path that used to live in ``ClaudeSession.prompt``
(#637) was removed in #955 — preemption now fires synchronously in the HTTP
handler via :meth:`~fido.provider.OwnedSession.preempt_worker` before the
background thread spawns, and :meth:`~fido.provider.OwnedSession.hold_for_handler`
provides a secondary safety net before lock acquisition.  Tests for those
paths live in ``test_claude_hold_for_handler.py`` and ``test_server.py``.
"""

from fido import provider


def test_set_thread_kind_roundtrip() -> None:
    provider.set_thread_kind(None)
    assert provider.current_thread_kind() == "worker"
    provider.set_thread_kind("webhook")
    assert provider.current_thread_kind() == "webhook"
    provider.set_thread_kind("worker")
    assert provider.current_thread_kind() == "worker"
    provider.set_thread_kind(None)
    assert provider.current_thread_kind() == "worker"
