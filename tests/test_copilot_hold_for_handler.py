"""Reentrance + hold_for_handler tests for CopilotCLISession (#658)."""

from pathlib import Path

import pytest

from fido import provider
from fido.copilotcli import CopilotCLIClient, CopilotCLISession


class FakeRuntime:
    def __init__(self) -> None:
        self._session_id = "sess-created"
        self.cancel_calls: list[str] = []
        self.dropped_session_count = 0
        self.pid = None

    def ensure_session(self, session_id, model):  # noqa: ARG002
        return self._session_id

    def recover_session(self, session_id, model):  # noqa: ARG002
        return self._session_id

    def reset_session(self, model):  # noqa: ARG002
        return self._session_id

    def cancel(self, session_id: str) -> None:
        self.cancel_calls.append(session_id)

    def is_alive(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def prompt(self, session_id, content, model):  # noqa: ARG002
        return "ok", "completed", session_id


def _session(tmp_path: Path) -> CopilotCLISession:
    sys_file = tmp_path / "persona.md"
    sys_file.write_text("")
    return CopilotCLISession(
        sys_file,
        work_dir=tmp_path,
        model=CopilotCLIClient.work_model,
        runtime=FakeRuntime(),
        repo_name="owner/repo",
    )


def test_hold_for_handler_allows_nested_with(tmp_path: Path) -> None:
    """Copilot's hold_for_handler + inner ``with session:`` work as a
    single talker registration across both entries (mirrors the claude
    behaviour in #658)."""
    session = _session(tmp_path)
    provider.set_thread_kind("webhook")
    try:
        with session.hold_for_handler():
            assert provider.get_talker("owner/repo") is not None
            with session:
                # Reentrant — still the single outer talker.
                assert provider.get_talker("owner/repo") is not None
        assert provider.get_talker("owner/repo") is None
    finally:
        provider.set_thread_kind(None)


def test_hold_for_handler_does_not_fire_runtime_cancel_when_free(
    tmp_path: Path,
) -> None:
    """No holder + no ``preempt_worker=True`` → no runtime cancel."""
    session = _session(tmp_path)
    assert isinstance(session._runtime, FakeRuntime)
    provider.set_thread_kind("webhook")
    try:
        with session.hold_for_handler():
            pass
    finally:
        provider.set_thread_kind(None)
    assert session._runtime.cancel_calls == []


def test_hold_for_handler_preempt_fires_runtime_cancel_on_worker_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Webhook caller + worker currently holding (per talker registry)
    → runtime cancel fires once via preempt-always semantics in
    ``__enter__`` (#637)."""
    session = _session(tmp_path)
    assert isinstance(session._runtime, FakeRuntime)

    def fake_talker(_repo: str) -> provider.SessionTalker:
        return provider.SessionTalker(
            repo_name="owner/repo",
            thread_id=999_999,
            kind="worker",
            description="fake-worker",
            claude_pid=0,
            started_at=provider.talker_now(),
        )

    monkeypatch.setattr(provider, "get_talker", fake_talker)
    provider.set_thread_kind("webhook")
    try:
        with session.hold_for_handler():
            pass
    finally:
        provider.set_thread_kind(None)
    assert session._runtime.cancel_calls == ["sess-created"]
