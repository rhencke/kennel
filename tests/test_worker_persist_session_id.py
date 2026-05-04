"""Tests for session-id persistence in WorkerThread (#649)."""

import json
import subprocess
from pathlib import Path
from typing import Never
from unittest.mock import MagicMock

import pytest

from fido.github import GitHub
from fido.issue_cache import IssueTreeCache
from fido.worker import WorkerThread
from tests.fakes import _FakeDispatcher


def _make_thread(tmp_path: Path, **kwargs: object) -> WorkerThread:
    gh = MagicMock(spec=GitHub)
    kwargs.setdefault("issue_cache", IssueTreeCache("owner/repo"))
    kwargs.setdefault("dispatcher", _FakeDispatcher())
    return WorkerThread(
        tmp_path,
        "owner/repo",
        gh=gh,
        **kwargs,
    )


def _init_git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    fido_dir = tmp_path / ".git" / "fido"
    fido_dir.mkdir(parents=True, exist_ok=True)
    return fido_dir


def test_load_persisted_session_id_returns_value(tmp_path: Path) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(json.dumps({"session_id": "abc-123"}))
    thread = _make_thread(tmp_path)
    assert thread._load_persisted_session_id() == "abc-123"


def test_load_persisted_session_id_none_when_absent(tmp_path: Path) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(json.dumps({"issue": 5}))
    thread = _make_thread(tmp_path)
    assert thread._load_persisted_session_id() is None


def test_load_persisted_session_id_none_when_not_a_git_repo(tmp_path: Path) -> None:
    """tmp_path without `git init` — persistence is unavailable but must
    not crash callers.  Same shape as test fixtures that construct a worker
    against a bare directory."""
    thread = _make_thread(tmp_path)
    assert thread._resolve_fido_dir() is None
    assert thread._load_persisted_session_id() is None


def test_load_persisted_session_id_handles_state_load_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(json.dumps({"session_id": "abc"}))
    thread = _make_thread(tmp_path)
    from fido import state as state_mod

    def boom(self: object) -> Never:
        raise OSError("permission denied")

    monkeypatch.setattr(state_mod.State, "load", boom)
    assert thread._load_persisted_session_id() is None


def test_persist_session_id_writes_new_value(tmp_path: Path) -> None:
    fido_dir = _init_git_repo(tmp_path)
    session = MagicMock()
    session.session_id = "new-sid-456"
    agent = MagicMock()
    agent.session = session
    provider = MagicMock()
    provider.agent = agent
    thread = _make_thread(tmp_path, provider=provider)
    thread._persist_session_id()
    persisted = json.loads((fido_dir / "state.json").read_text())
    assert persisted["session_id"] == "new-sid-456"


def test_persist_session_id_skips_when_unchanged(tmp_path: Path) -> None:
    """Avoid rewriting state.json when the id hasn't changed — keeps the
    file's mtime stable and reduces lock contention under steady state."""
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(
        json.dumps({"session_id": "same-sid", "issue": 1})
    )
    mtime_before = (fido_dir / "state.json").stat().st_mtime_ns
    session = MagicMock()
    session.session_id = "same-sid"
    agent = MagicMock()
    agent.session = session
    provider = MagicMock()
    provider.agent = agent
    thread = _make_thread(tmp_path, provider=provider)
    thread._persist_session_id()
    # mtime should not be bumped beyond what State.modify does on read-only
    # operations — the file content must stay the same.
    persisted = json.loads((fido_dir / "state.json").read_text())
    assert persisted == {"session_id": "same-sid", "issue": 1}
    assert (fido_dir / "state.json").stat().st_mtime_ns >= mtime_before


def test_persist_session_id_noop_when_no_provider(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    thread = _make_thread(tmp_path)
    thread._provider = None  # pyright: ignore[reportPrivateUsage]
    thread._persist_session_id()  # must not raise


def test_persist_session_id_noop_when_no_session(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    provider = MagicMock()
    provider.agent.session = None
    thread = _make_thread(tmp_path, provider=provider)
    thread._persist_session_id()  # must not raise


def test_persist_session_id_noop_when_session_has_empty_id(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    session = MagicMock()
    session.session_id = ""
    provider = MagicMock()
    provider.agent.session = session
    thread = _make_thread(tmp_path, provider=provider)
    thread._persist_session_id()  # must not raise and must not write


def test_persist_session_id_noop_when_fido_dir_unresolvable(tmp_path: Path) -> None:
    """Not a git repo → no fido_dir → persistence silently skipped."""
    session = MagicMock()
    session.session_id = "some-sid"
    provider = MagicMock()
    provider.agent.session = session
    thread = _make_thread(tmp_path, provider=provider)
    # No git init — resolving fails
    thread._persist_session_id()  # must not raise


def test_persist_session_id_swallows_state_modify_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    _init_git_repo(tmp_path)
    session = MagicMock()
    session.session_id = "sid"
    provider = MagicMock()
    provider.agent.session = session
    thread = _make_thread(tmp_path, provider=provider)
    from contextlib import contextmanager

    from fido import state as state_mod

    @contextmanager
    def boom(self: object) -> object:
        raise OSError("state.json locked by another process")
        yield  # unreachable

    monkeypatch.setattr(state_mod.State, "modify", boom)
    with caplog.at_level(logging.WARNING, logger="fido"):
        thread._persist_session_id()
    assert "failed to persist session_id" in caplog.text


# ── _retire_poisoned_session ──────────────────────────────────────────────────


def test_retire_poisoned_session_clears_session_id_from_state(
    tmp_path: Path,
) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(json.dumps({"session_id": "poisoned-sid"}))
    thread = _make_thread(tmp_path)
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    persisted = json.loads((fido_dir / "state.json").read_text())
    assert "session_id" not in persisted


def test_retire_poisoned_session_preserves_other_state_keys(tmp_path: Path) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(
        json.dumps({"session_id": "poisoned-sid", "issue": 42})
    )
    thread = _make_thread(tmp_path)
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    persisted = json.loads((fido_dir / "state.json").read_text())
    assert persisted == {"issue": 42}


def test_retire_poisoned_session_calls_session_reset(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    session = MagicMock()
    agent = MagicMock()
    agent.session = session
    provider = MagicMock()
    provider.agent = agent
    thread = _make_thread(tmp_path, provider=provider)
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    session.reset.assert_called_once()


def test_retire_poisoned_session_clears_session_issue(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    thread = _make_thread(tmp_path)
    thread._session_issue = 99  # pyright: ignore[reportPrivateUsage]
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    assert thread._session_issue is None  # pyright: ignore[reportPrivateUsage]


def test_retire_poisoned_session_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    _init_git_repo(tmp_path)
    thread = _make_thread(tmp_path)
    with caplog.at_level(logging.WARNING, logger="fido"):
        thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    assert "context overflow" in caplog.text


def test_retire_poisoned_session_noop_when_no_git_repo(tmp_path: Path) -> None:
    """Not a git repo — no fido_dir — must not raise."""
    thread = _make_thread(tmp_path)
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]


def test_retire_poisoned_session_noop_when_no_provider(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    thread = _make_thread(tmp_path)
    thread._provider = None  # pyright: ignore[reportPrivateUsage]
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]  # must not raise


def test_retire_poisoned_session_noop_when_no_session(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    provider = MagicMock()
    provider.agent.session = None
    thread = _make_thread(tmp_path, provider=provider)
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]  # must not raise


def test_retire_poisoned_session_swallows_state_modify_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging
    from contextlib import contextmanager

    from fido import state as state_mod

    _init_git_repo(tmp_path)
    thread = _make_thread(tmp_path)

    @contextmanager
    def boom(self: object) -> object:
        raise OSError("state.json locked")
        yield  # unreachable

    monkeypatch.setattr(state_mod.State, "modify", boom)
    with caplog.at_level(logging.WARNING, logger="fido"):
        thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    assert "failed to clear session_id after context overflow" in caplog.text
