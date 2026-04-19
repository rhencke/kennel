"""Tests for session-id persistence in WorkerThread (#649)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from kennel.github import GitHub
from kennel.issue_cache import IssueTreeCache
from kennel.worker import WorkerThread


def _make_thread(tmp_path: Path, **kwargs) -> WorkerThread:
    gh = MagicMock(spec=GitHub)
    kwargs.setdefault("issue_cache", IssueTreeCache("owner/repo"))
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
    tmp_path: Path, monkeypatch
) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(json.dumps({"session_id": "abc"}))
    thread = _make_thread(tmp_path)
    from kennel import state as state_mod

    def boom(self):
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
    tmp_path: Path, monkeypatch, caplog
) -> None:
    import logging

    _init_git_repo(tmp_path)
    session = MagicMock()
    session.session_id = "sid"
    provider = MagicMock()
    provider.agent.session = session
    thread = _make_thread(tmp_path, provider=provider)
    from contextlib import contextmanager

    from kennel import state as state_mod

    @contextmanager
    def boom(self):
        raise OSError("state.json locked by another process")
        yield  # unreachable

    monkeypatch.setattr(state_mod.State, "modify", boom)
    with caplog.at_level(logging.WARNING, logger="kennel"):
        thread._persist_session_id()
    assert "failed to persist session_id" in caplog.text
