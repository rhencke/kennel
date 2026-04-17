"""Tests for kennel.worker — WorkerContext, lock acquisition, git context."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock, call, patch

import pytest

import kennel.worker as worker_module
from kennel.claude import ClaudeClient
from kennel.config import RepoConfig, RepoMembership
from kennel.prompts import Prompts
from kennel.provider import (
    ProviderID,
    ProviderLimitSnapshot,
    ProviderLimitWindow,
    ProviderModel,
    TurnSessionMode,
)
from kennel.state import (
    State,
    _resolve_git_dir,
)
from kennel.tasks import (
    _apply_queue_to_body,
    _auto_complete_ask_tasks,
    _format_work_queue,
    sync_tasks,
    sync_tasks_background,
)
from kennel.worker import (
    LockHeld,
    RepoContext,
    RepoContextFilter,
    RepoNameFilter,
    WorkerContext,
    _pick_next_task,
    _sanitize_slug,
    _sanitize_status_text,
    _thread_repo,
    _write_pr_description,
    acquire_lock,
    build_prompt,
    ci_ready_for_review,
    create_compact_script,
    latest_decisive_review,
    provider_run,
    provider_start,
    run,
    should_rerequest_review,
)
from kennel.worker import (
    Worker as _WorkerBase,
)
from kennel.worker import (
    WorkerThread as _WorkerThreadBase,
)

_MISSING = object()


def _default_repo_cfg(
    work_dir: Path,
    *,
    repo_name: str = "",
    membership: RepoMembership | None = None,
) -> RepoConfig:
    return RepoConfig(
        name=repo_name or "owner/repo",
        work_dir=work_dir,
        provider=ProviderID.CLAUDE_CODE,
        membership=membership if membership is not None else RepoMembership(),
    )


class Worker(_WorkerBase):
    def __init__(self, work_dir: Path, gh, *args, **kwargs) -> None:
        repo_cfg = kwargs.get("repo_cfg", _MISSING)
        if (
            repo_cfg is _MISSING
            and kwargs.get("provider") is None
            and kwargs.get("provider_agent") is None
        ):
            kwargs["repo_cfg"] = _default_repo_cfg(
                work_dir,
                repo_name=kwargs.get("repo_name", ""),
                membership=kwargs.get("membership"),
            )
        super().__init__(work_dir, gh, *args, **kwargs)


class WorkerThread(_WorkerThreadBase):
    def __init__(self, work_dir: Path, repo_name: str, gh, *args, **kwargs) -> None:
        repo_cfg = kwargs.get("repo_cfg", _MISSING)
        if repo_cfg is _MISSING and kwargs.get("provider") is None:
            kwargs["repo_cfg"] = _default_repo_cfg(
                work_dir,
                repo_name=repo_name,
                membership=kwargs.get("membership"),
            )
        super().__init__(work_dir, repo_name, gh, *args, **kwargs)


@pytest.fixture(autouse=True)
def _patch_worker_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module, "Worker", Worker)
    monkeypatch.setattr(worker_module, "WorkerThread", WorkerThread)


@pytest.fixture(autouse=True)
def _no_claude_session_spawn(monkeypatch):
    """Patch ClaudeSession for every test in this module.

    Worker.run() now creates a ClaudeSession on entry.  Without this fixture
    every test that calls worker.run() would try to spawn a real claude
    subprocess.  The mock is a MagicMock so all attribute and method accesses
    (stop, send, iter_events, …) work without raising AttributeError.
    """
    from kennel import claude

    monkeypatch.setattr(claude, "ClaudeSession", MagicMock(return_value=MagicMock()))


def _client(return_value: str = "", *, side_effect=None) -> MagicMock:
    """Build a mock ClaudeClient with run_turn configured."""
    client = MagicMock(spec=ClaudeClient)
    client.provider_id = ProviderID.CLAUDE_CODE
    client.voice_model = "claude-opus-4-6"
    client.work_model = "claude-sonnet-4-6"
    client.brief_model = "claude-haiku-4-5"
    client.session = None
    client.session_owner = None
    client.session_alive = False
    client.session_pid = None
    client.session_id = None
    client.extract_session_id.return_value = ""
    if side_effect is not None:
        client.run_turn.side_effect = side_effect
    else:
        client.run_turn.return_value = return_value
    return client


class TestRepoContextFilter:
    def test_filter_injects_repo_name_from_thread_local(self) -> None:
        _thread_repo.repo_name = "confusio"
        record = logging.LogRecord("", logging.INFO, "", 0, "", (), None)
        try:
            assert RepoContextFilter().filter(record) is True
            assert record.repo_name == "confusio"  # type: ignore[attr-defined]
        finally:
            del _thread_repo.repo_name

    def test_filter_defaults_to_dash_when_no_context(self) -> None:
        # Ensure the thread-local is not set for this test.
        _thread_repo.__dict__.pop("repo_name", None)
        record = logging.LogRecord("", logging.INFO, "", 0, "", (), None)
        RepoContextFilter().filter(record)
        assert record.repo_name == "-"  # type: ignore[attr-defined]

    def test_filter_always_returns_true(self) -> None:
        _thread_repo.__dict__.pop("repo_name", None)
        record = logging.LogRecord("", logging.WARNING, "", 0, "", (), None)
        assert RepoContextFilter().filter(record) is True

    def test_filter_preserves_explicit_repo_name(self) -> None:
        _thread_repo.repo_name = "confusio"
        record = logging.LogRecord("", logging.INFO, "", 0, "", (), None)
        record.repo_name = "orly"  # type: ignore[attr-defined]
        try:
            assert RepoContextFilter().filter(record) is True
            assert record.repo_name == "orly"  # type: ignore[attr-defined]
        finally:
            del _thread_repo.repo_name


class TestRepoNameFilter:
    def _record_with_repo(self, repo_name: str) -> logging.LogRecord:
        record = logging.LogRecord("", logging.INFO, "", 0, "", (), None)
        record.repo_name = repo_name  # type: ignore[attr-defined]
        return record

    def test_passes_matching_repo(self) -> None:
        f = RepoNameFilter("kennel")
        assert f.filter(self._record_with_repo("kennel")) is True

    def test_blocks_other_repo(self) -> None:
        f = RepoNameFilter("kennel")
        assert f.filter(self._record_with_repo("confusio")) is False

    def test_blocks_default_dash(self) -> None:
        f = RepoNameFilter("kennel")
        assert f.filter(self._record_with_repo("-")) is False

    def test_blocks_record_without_repo_name(self) -> None:
        f = RepoNameFilter("kennel")
        record = logging.LogRecord("", logging.INFO, "", 0, "", (), None)
        assert f.filter(record) is False


class TestResolveGitDir:
    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def test_returns_path(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="/some/repo/.git\n"))
        result = self._make_worker(tmp_path).resolve_git_dir(_run=mock_run)
        assert result == Path("/some/repo/.git")

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="  /a/b/.git  \n"))
        result = self._make_worker(tmp_path).resolve_git_dir(_run=mock_run)
        assert result == Path("/a/b/.git")

    def test_calls_correct_command(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="/a/.git"))
        self._make_worker(tmp_path).resolve_git_dir(_run=mock_run)
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_propagates_called_process_error(self, tmp_path: Path) -> None:
        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(128, "git"))
        with pytest.raises(subprocess.CalledProcessError):
            self._make_worker(tmp_path).resolve_git_dir(_run=mock_run)


class TestAcquireLock:
    def test_returns_open_fd(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd = acquire_lock(fido_dir)
        assert not fd.closed
        fd.close()

    def test_creates_fido_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "nested" / "fido"
        fd = acquire_lock(fido_dir)
        assert fido_dir.is_dir()
        fd.close()

    def test_creates_lock_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd = acquire_lock(fido_dir)
        assert (fido_dir / "lock").exists()
        fd.close()

    def test_raises_lock_held_when_already_locked(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = acquire_lock(fido_dir)
        try:
            with pytest.raises(LockHeld):
                acquire_lock(fido_dir)
        finally:
            fd1.close()

    def test_lock_held_message(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = acquire_lock(fido_dir)
        try:
            with pytest.raises(LockHeld, match="another fido"):
                acquire_lock(fido_dir)
        finally:
            fd1.close()

    def test_reacquirable_after_release(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = acquire_lock(fido_dir)
        fd1.close()
        fd2 = acquire_lock(fido_dir)
        assert not fd2.closed
        fd2.close()


class TestWorkerContextManager:
    def test_closes_lock_fd_on_exit(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd = acquire_lock(fido_dir)
        ctx = WorkerContext(
            work_dir=tmp_path, git_dir=tmp_path / ".git", fido_dir=fido_dir, lock_fd=fd
        )
        with ctx:
            assert not fd.closed
        assert fd.closed

    def test_closes_lock_fd_on_exception(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd = acquire_lock(fido_dir)
        ctx = WorkerContext(
            work_dir=tmp_path, git_dir=tmp_path / ".git", fido_dir=fido_dir, lock_fd=fd
        )
        with pytest.raises(RuntimeError):
            with ctx:
                raise RuntimeError("boom")
        assert fd.closed


class TestCreateContext:
    def _mock_run(self, git_dir: Path) -> MagicMock:
        return MagicMock(return_value=MagicMock(stdout=str(git_dir) + "\n"))

    def test_returns_worker_context(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        ctx = Worker(tmp_path, MagicMock()).create_context(_run=self._mock_run(git_dir))
        assert isinstance(ctx, WorkerContext)
        assert ctx.work_dir == tmp_path
        assert ctx.git_dir == git_dir
        assert ctx.fido_dir == git_dir / "fido"
        ctx.lock_fd.close()

    def test_creates_fido_dir(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        ctx = Worker(tmp_path, MagicMock()).create_context(_run=self._mock_run(git_dir))
        assert ctx.fido_dir.is_dir()
        ctx.lock_fd.close()

    def test_propagates_lock_held(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        fido_dir = git_dir / "fido"
        fd1 = acquire_lock(fido_dir)
        try:
            with pytest.raises(LockHeld):
                Worker(tmp_path, MagicMock()).create_context(
                    _run=self._mock_run(git_dir)
                )
        finally:
            fd1.close()


class TestRepoContext:
    def test_fields(self) -> None:
        ctx = RepoContext(
            repo="alice/myrepo",
            owner="alice",
            repo_name="myrepo",
            gh_user="bot",
            default_branch="main",
            membership=RepoMembership(collaborators=frozenset({"alice", "carol"})),
        )
        assert ctx.repo == "alice/myrepo"
        assert ctx.owner == "alice"
        assert ctx.repo_name == "myrepo"
        assert ctx.gh_user == "bot"
        assert ctx.default_branch == "main"
        assert ctx.collaborators == frozenset({"alice", "carol"})

    def test_collaborators_default_empty(self) -> None:
        ctx = RepoContext(
            repo="alice/repo",
            owner="alice",
            repo_name="repo",
            gh_user="bot",
            default_branch="main",
        )
        assert ctx.collaborators == frozenset()


class TestWorker:
    def _make_gh(
        self,
        repo: str = "owner/myrepo",
        user: str = "fido-bot",
        branch: str = "main",
    ) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = repo
        gh.get_user.return_value = user
        gh.get_default_branch.return_value = branch
        gh.get_pr.return_value = {"body": ""}
        return gh

    # --- constructor / config injection ---

    def test_config_stored_when_passed(self, tmp_path: Path) -> None:
        from kennel.config import Config, RepoConfig

        cfg = RepoConfig(
            name="owner/repo", work_dir=tmp_path, provider=ProviderID.CLAUDE_CODE
        )
        config = Config(
            port=9000,
            secret=b"s",
            repos={"owner/repo": cfg},
            allowed_bots=frozenset(),
            log_level="DEBUG",
            sub_dir=tmp_path,
        )
        worker = Worker(tmp_path, MagicMock(), config=config, repo_cfg=None)
        assert worker._config is config

    def test_repo_cfg_stored_when_passed(self, tmp_path: Path) -> None:
        from kennel.config import RepoConfig

        cfg = RepoConfig(
            name="owner/repo", work_dir=tmp_path, provider=ProviderID.CLAUDE_CODE
        )
        worker = Worker(tmp_path, MagicMock(), repo_cfg=cfg)
        assert worker._repo_cfg is cfg

    def test_repo_cfg_provider_selects_copilot_provider(self, tmp_path: Path) -> None:
        from kennel.config import RepoConfig

        worker = Worker(
            tmp_path,
            MagicMock(),
            repo_cfg=RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.COPILOT_CLI,
            ),
        )
        assert worker._provider.provider_id == ProviderID.COPILOT_CLI  # pyright: ignore[reportPrivateUsage]

    def test_config_defaults_to_none(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock())
        assert worker._config is None

    def test_repo_cfg_defaults_to_none(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock(), repo_cfg=None, provider=MagicMock())
        assert worker._repo_cfg is None

    # --- discover_repo_context ---

    def test_discover_returns_repo_context(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        result = Worker(tmp_path, gh).discover_repo_context()
        assert isinstance(result, RepoContext)

    def test_discover_repo_field(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="alice/proj")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.repo == "alice/proj"

    def test_discover_owner_parsed(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="alice/proj")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.owner == "alice"

    def test_discover_repo_name_parsed(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="alice/proj")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.repo_name == "proj"

    def test_discover_gh_user(self, tmp_path: Path) -> None:
        gh = self._make_gh(user="fido")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.gh_user == "fido"

    def test_discover_default_branch(self, tmp_path: Path) -> None:
        gh = self._make_gh(branch="develop")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.default_branch == "develop"

    def test_discover_passes_cwd_to_get_repo_info(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        Worker(tmp_path, gh).discover_repo_context()
        gh.get_repo_info.assert_called_once_with(cwd=tmp_path)

    def test_discover_passes_cwd_to_get_default_branch(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        Worker(tmp_path, gh).discover_repo_context()
        gh.get_default_branch.assert_called_once_with(cwd=tmp_path)

    def test_discover_splits_on_first_slash_only(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="org/repo")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.owner == "org"
        assert result.repo_name == "repo"

    def test_discover_uses_injected_membership(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        membership = RepoMembership(collaborators=frozenset({"alice", "bob"}))
        result = Worker(tmp_path, gh, membership=membership).discover_repo_context()
        assert result.membership is membership
        assert result.collaborators == frozenset({"alice", "bob"})

    def test_discover_default_membership_empty(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.collaborators == frozenset()

    def test_discover_does_not_call_get_collaborators(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        Worker(tmp_path, gh).discover_repo_context()
        gh.get_collaborators.assert_not_called()

    # --- set_status ---

    def _session(
        self,
        *,
        status: str = "ok",
        emoji: str = "🐕",
        raw: str | None = None,
    ) -> MagicMock:
        """Build a mock :class:`ClaudeSession` whose ``prompt`` returns a JSON nudge."""
        session = MagicMock()
        if raw is None:
            raw = f'{{"status": "{status}", "emoji": "{emoji}"}}'
        session.prompt.return_value = raw
        return session

    def test_set_status_calls_set_user_status_on_success(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        (tmp_path / "persona.md").write_text("I am Fido.")
        Worker(tmp_path, gh, session=self._session(status="writing tests")).set_status(
            "writing tests", _sub_dir_fn=lambda: tmp_path
        )
        gh.set_user_status.assert_called_once_with("writing tests", "🐕", busy=True)

    def test_set_status_busy_false_forwarded(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        (tmp_path / "persona.md").write_text("I am Fido.")
        Worker(
            tmp_path, gh, session=self._session(status="napping", emoji="😴")
        ).set_status("napping", busy=False, _sub_dir_fn=lambda: tmp_path)
        gh.set_user_status.assert_called_once_with("napping", "😴", busy=False)

    def test_set_status_uses_what_as_fallback_when_status_empty(
        self, tmp_path: Path
    ) -> None:
        gh = self._make_gh()
        (tmp_path / "persona.md").write_text("I am Fido.")
        Worker(
            tmp_path, gh, session=self._session(status="", emoji=":dog:")
        ).set_status("idle", _sub_dir_fn=lambda: tmp_path)
        assert gh.set_user_status.call_args[0][0] == "idle"

    def test_set_status_emoji_fallback_when_empty(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        (tmp_path / "persona.md").write_text("I am Fido.")
        Worker(
            tmp_path, gh, session=self._session(status="Sniffing endpoints", emoji="")
        ).set_status("idle", _sub_dir_fn=lambda: tmp_path)
        gh.set_user_status.assert_called_once_with(
            "Sniffing endpoints", ":dog:", busy=True
        )

    def test_set_status_text_truncated_to_80_chars(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        long_text = "x" * 100
        (tmp_path / "persona.md").write_text("I am Fido.")
        Worker(tmp_path, gh, session=self._session(status=long_text)).set_status(
            "something", _sub_dir_fn=lambda: tmp_path
        )
        called_text = gh.set_user_status.call_args[0][0]
        assert len(called_text) == 80

    def test_set_status_skipped_when_no_session(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = self._make_gh()
        (tmp_path / "persona.md").write_text("I am Fido.")
        with caplog.at_level(logging.INFO, logger="kennel"):
            Worker(tmp_path, gh, session=None).set_status(
                "idle", _sub_dir_fn=lambda: tmp_path
            )
        gh.set_user_status.assert_not_called()
        assert "no session available" in caplog.text

    def test_set_status_logs_warning_on_empty_status(
        self, tmp_path: Path, caplog
    ) -> None:
        import logging

        gh = self._make_gh()
        (tmp_path / "persona.md").write_text("I am Fido.")
        with caplog.at_level(logging.WARNING, logger="kennel"):
            Worker(
                tmp_path, gh, session=self._session(status="", emoji=":dog:")
            ).set_status("idle", _sub_dir_fn=lambda: tmp_path)
        assert "falling back" in caplog.text

    def test_set_status_logs_info_on_success(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = self._make_gh()
        (tmp_path / "persona.md").write_text("I am Fido.")
        with caplog.at_level(logging.INFO, logger="kennel"):
            Worker(tmp_path, gh, session=self._session(status="fetching")).set_status(
                "fetching", _sub_dir_fn=lambda: tmp_path
            )
        assert "set_status" in caplog.text

    def test_set_status_falls_back_to_empty_persona_on_oserror(
        self, tmp_path: Path
    ) -> None:
        gh = self._make_gh()
        missing_dir = tmp_path / "no_such_dir"
        session = self._session(status="working")
        Worker(tmp_path, gh, session=session).set_status(
            "working", _sub_dir_fn=lambda: missing_dir
        )
        prompt_arg = session.prompt.call_args[0][0]
        assert "working (busy)" in prompt_arg

    def test_set_status_passes_system_prompt_to_session(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        (tmp_path / "persona.md").write_text("I am Fido.")
        session = self._session(status="working")
        Worker(tmp_path, gh, session=session).set_status(
            "working", _sub_dir_fn=lambda: tmp_path
        )
        assert session.prompt.call_args[1]["system_prompt"] is not None

    def test_set_status_reports_activity_to_registry(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        registry = MagicMock()
        registry.get_all_activities.return_value = []
        (tmp_path / "persona.md").write_text("I am Fido.")
        Worker(
            tmp_path,
            gh,
            repo_name="owner/myrepo",
            registry=registry,
            session=self._session(status="working"),
        ).set_status("working", busy=True, _sub_dir_fn=lambda: tmp_path)
        registry.report_activity.assert_called_once_with(
            "owner/myrepo", "working", True
        )

    def test_set_status_uses_full_registry_snapshot_for_prompt(
        self, tmp_path: Path
    ) -> None:
        gh = self._make_gh()
        registry = MagicMock()
        activity_a = MagicMock()
        activity_a.repo_name = "owner/repo-a"
        activity_a.what = "fixing bug"
        activity_a.busy = True
        activity_b = MagicMock()
        activity_b.repo_name = "owner/repo-b"
        activity_b.what = "napping"
        activity_b.busy = False
        registry.get_all_activities.return_value = [activity_a, activity_b]
        (tmp_path / "persona.md").write_text("I am Fido.")
        session = self._session(status="fixing bug")
        Worker(
            tmp_path,
            gh,
            repo_name="owner/repo-a",
            registry=registry,
            session=session,
        ).set_status("fixing bug", _sub_dir_fn=lambda: tmp_path)
        prompt_arg = session.prompt.call_args[0][0]
        assert "owner/repo-a" in prompt_arg
        assert "owner/repo-b" in prompt_arg

    def test_set_status_acquires_status_lock_when_registry_present(
        self, tmp_path: Path
    ) -> None:
        gh = self._make_gh()
        registry = MagicMock()
        registry.get_all_activities.return_value = []
        (tmp_path / "persona.md").write_text("I am Fido.")
        Worker(
            tmp_path,
            gh,
            repo_name="owner/repo",
            registry=registry,
            session=self._session(status="working"),
        ).set_status("working", _sub_dir_fn=lambda: tmp_path)
        registry.status_update.assert_called_once()

    def test_set_status_serializes_concurrent_calls_via_registry_lock(
        self, tmp_path: Path
    ) -> None:
        """Concurrent set_status calls on different workers sharing a registry serialize."""
        from kennel.registry import WorkerRegistry

        registry = WorkerRegistry(MagicMock())
        inside_count = 0
        max_concurrent = 0
        counter_lock = threading.Lock()

        def slow_prompt(*args, **kwargs) -> str:
            nonlocal inside_count, max_concurrent
            with counter_lock:
                inside_count += 1
                max_concurrent = max(max_concurrent, inside_count)
            time.sleep(0.005)
            with counter_lock:
                inside_count -= 1
            return '{"status": "ok", "emoji": "🐕"}'

        def make_session() -> MagicMock:
            s = MagicMock()
            s.prompt.side_effect = slow_prompt
            return s

        workers = [
            Worker(
                tmp_path,
                self._make_gh(),
                repo_name=f"owner/repo{i}",
                registry=registry,
                session=make_session(),
            )
            for i in range(3)
        ]
        threads = [
            threading.Thread(
                target=w.set_status,
                args=("working",),
                kwargs={"_sub_dir_fn": lambda: tmp_path / "nosub"},
            )
            for w in workers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert max_concurrent == 1

    # --- get_current_issue ---

    def _make_issue_gh(self, state: str = "OPEN") -> MagicMock:
        gh = MagicMock()
        gh.view_issue.return_value = {"state": state, "title": "Test", "body": ""}
        return gh

    def test_get_issue_returns_none_when_no_state(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        gh = self._make_issue_gh()
        assert Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo") is None

    def test_get_issue_returns_issue_number_when_open(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 7})
        gh = self._make_issue_gh(state="OPEN")
        assert Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo") == 7

    def test_get_issue_returns_int_type(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 7})
        gh = self._make_issue_gh(state="OPEN")
        result = Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert isinstance(result, int)

    def test_get_issue_returns_none_when_closed(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 4})
        gh = self._make_issue_gh(state="CLOSED")
        assert Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo") is None

    def test_get_issue_clears_state_when_closed(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 4})
        gh = self._make_issue_gh(state="CLOSED")
        Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert State(fido_dir).load() == {}

    def test_get_issue_does_not_call_view_issue_when_no_state(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        gh = self._make_issue_gh()
        Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        gh.view_issue.assert_not_called()

    def test_get_issue_calls_view_issue_with_correct_args(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 12})
        gh = self._make_issue_gh(state="OPEN")
        Worker(tmp_path, gh).get_current_issue(fido_dir, "alice/proj")
        gh.view_issue.assert_called_once_with("alice/proj", 12)

    def test_get_issue_logs_info_when_closed(self, tmp_path: Path, caplog) -> None:
        import logging

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 9})
        gh = self._make_issue_gh(state="CLOSED")
        with caplog.at_level(logging.INFO, logger="kennel"):
            Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert "advancing" in caplog.text

    def test_get_issue_state_preserved_when_open(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 5})
        gh = self._make_issue_gh(state="OPEN")
        Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert State(fido_dir).load() == {"issue": 5}

    # --- run ---

    def _make_mock_ctx(self, tmp_path: Path) -> MagicMock:
        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        mock_ctx.fido_dir = tmp_path / ".git" / "fido"
        return mock_ctx

    def _make_mock_repo_ctx(self) -> MagicMock:
        repo_ctx = MagicMock(spec=RepoContext)
        repo_ctx.repo = "owner/repo"
        repo_ctx.gh_user = "fido-bot"
        repo_ctx.default_branch = "main"
        repo_ctx.membership = RepoMembership()
        return repo_ctx

    # --- create_session / stop_session ---

    def test_create_session_instantiates_claude_session(self, tmp_path: Path) -> None:
        from kennel import claude

        worker = Worker(tmp_path, MagicMock())
        worker.create_session()
        claude.ClaudeSession.assert_called_once()

    def test_create_session_passes_work_dir(self, tmp_path: Path) -> None:
        from kennel import claude

        worker = Worker(tmp_path, MagicMock())
        worker.create_session()
        _, kwargs = claude.ClaudeSession.call_args
        assert kwargs.get("work_dir") == tmp_path

    def test_create_session_switches_to_opus(self, tmp_path: Path) -> None:
        from kennel import claude

        mock_session = MagicMock()
        claude.ClaudeSession.return_value = mock_session
        worker = Worker(tmp_path, MagicMock())
        worker.create_session()
        _, kwargs = claude.ClaudeSession.call_args
        assert kwargs.get("model") == "claude-opus-4-6"
        mock_session.switch_model.assert_not_called()

    def test_create_session_stores_on_self(self, tmp_path: Path) -> None:
        from kennel import claude

        mock_session = MagicMock()
        claude.ClaudeSession.return_value = mock_session
        worker = Worker(tmp_path, MagicMock())
        worker.create_session()
        assert worker._session is mock_session

    def test_session_getter_reads_bootstrap_session_before_provider_exists(
        self,
    ) -> None:
        worker = Worker.__new__(Worker)
        worker.__dict__["_bootstrap_session"] = "boot"
        assert worker._session == "boot"

    def test_ensure_provider_creates_provider_from_repo_cfg(
        self, tmp_path: Path
    ) -> None:
        worker = Worker(
            tmp_path,
            MagicMock(),
            repo_cfg=_default_repo_cfg(tmp_path, repo_name="owner/repo"),
        )
        worker._provider = None  # pyright: ignore[reportPrivateUsage]
        agent = worker._provider_agent
        assert worker._provider is not None  # pyright: ignore[reportPrivateUsage]
        assert agent is worker._provider.agent  # pyright: ignore[reportPrivateUsage]

    def test_ensure_provider_requires_repo_cfg(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock(), repo_cfg=None)
        with pytest.raises(
            RuntimeError, match="worker provider requires explicit repo_cfg"
        ):
            worker._ensure_provider()  # pyright: ignore[reportPrivateUsage]

    def test_provider_constructor_path_attaches_session(self, tmp_path: Path) -> None:
        provider = MagicMock()
        provider.agent = MagicMock(spec=ClaudeClient)
        provider.agent.session = None
        session = MagicMock()
        provider.agent.attach_session.side_effect = lambda attached: setattr(
            provider.agent, "session", attached
        )
        worker = Worker(tmp_path, MagicMock(), provider=provider, session=session)
        provider.agent.attach_session.assert_called_once_with(session)
        assert worker._session is session

    def test_stop_session_calls_stop(self, tmp_path: Path) -> None:
        mock_session = MagicMock()
        worker = Worker(tmp_path, MagicMock())
        worker._session = mock_session
        worker.stop_session()
        mock_session.stop.assert_called_once()

    def test_stop_session_clears_session(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock())
        worker._session = MagicMock()
        worker.stop_session()
        assert worker._session is None

    def test_stop_session_is_noop_when_none(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock())
        assert worker._session is None
        worker.stop_session()  # must not raise

    def test_run_creates_session_with_fido_dir(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        mock_create = MagicMock()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "create_session", mock_create),
            patch.object(worker, "stop_session"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            worker.run()
        mock_create.assert_called_once_with()

    def test_run_recovers_reply_promises_before_normal_handlers(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "t", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        order: list[str] = []

        def mark_recover(*args, **kwargs):
            order.append("recover")
            return True

        def mark_ci(*args, **kwargs):
            order.append("ci")
            return False

        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "create_session"),
            patch.object(worker, "stop_session"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch("kennel.events.recover_reply_promises", side_effect=mark_recover),
            patch.object(worker, "handle_ci", side_effect=mark_ci),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=0),
        ):
            worker.run()
        assert order[:2] == ["recover", "ci"]

    def test_run_does_not_switch_model_for_carry_over_session(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "t", "body": "", "state": "OPEN"}
        mock_session = MagicMock()
        # same issue as last time — no boundary restart, no model switch
        worker = Worker(tmp_path, gh, session=mock_session, session_issue=7)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=0),
        ):
            worker.run()
        assert worker._next_turn_session_mode == TurnSessionMode.REUSE

    def test_run_marks_fresh_session_mode_at_issue_boundary(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "t", "body": "", "state": "OPEN"}
        mock_session = MagicMock()
        # session was working on issue 5; now issue 7 is picked — boundary
        worker = Worker(tmp_path, gh, session=mock_session, session_issue=5)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=0),
        ):
            worker.run()
        assert worker._next_turn_session_mode == TurnSessionMode.FRESH

    def test_run_sets_session_issue_after_picking_issue(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "t", "body": "", "state": "OPEN"}
        mock_session = MagicMock()
        worker = Worker(tmp_path, gh, session=mock_session, session_issue=7)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=0),
        ):
            worker.run()
        assert worker._session_issue == 7

    def test_run_does_not_create_session_when_provided(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        mock_session = MagicMock()
        worker = Worker(tmp_path, gh, session=mock_session)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "create_session") as mock_create,
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            worker.run()
        mock_create.assert_not_called()

    # --- run() ---

    def test_run_returns_2_when_lock_held(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with patch.object(worker, "create_context", side_effect=LockHeld("held")):
            assert worker.run() == 2

    def _run_patches(self, worker: Worker, mock_ctx: MagicMock) -> list:
        """Return a list of common patches needed to run Worker.run() in tests.

        Patches create_context, discover_repo_context, setup_hooks, teardown_hooks,
        get_current_issue (→ None), and find_next_issue (→ None) so the loop exits
        cleanly without hitting GitHub or Claude.
        """
        return [
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(
                worker, "setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ]

    def test_run_returns_0_on_success(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(
                worker, "setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            assert worker.run() == 0

    def test_run_logs_warning_on_lock_held(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with patch.object(worker, "create_context", side_effect=LockHeld("held")):
            with caplog.at_level(logging.WARNING, logger="kennel"):
                worker.run()
        assert "another fido" in caplog.text

    def test_run_logs_info_on_success(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(
                worker, "setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            with caplog.at_level(logging.INFO, logger="kennel"):
                worker.run()
        assert "worker started" in caplog.text

    def test_run_logs_repo_info(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(
                worker, "setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            with caplog.at_level(logging.INFO, logger="kennel"):
                worker.run()
        assert "owner/repo" in caplog.text

    def test_run_teardown_called_even_on_exception(self, tmp_path: Path) -> None:
        """teardown_hooks must be called even if the main loop raises."""
        mock_ctx = self._make_mock_ctx(tmp_path)
        mock_teardown = MagicMock()
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(
                worker, "setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch.object(worker, "teardown_hooks", mock_teardown),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            worker.run()
        mock_teardown.assert_called_once()

    def test_run_setup_hooks_called_with_fido_dir(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        mock_setup = MagicMock(return_value=("c", "s"))
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", mock_setup),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            worker.run()
        mock_setup.assert_called_once_with(mock_ctx.fido_dir)

    def test_run_calls_get_current_issue(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        mock_get_issue = MagicMock(return_value=None)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", mock_get_issue),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            worker.run()
        mock_get_issue.assert_called_once_with(mock_ctx.fido_dir, "owner/repo")

    def test_run_calls_find_next_issue_when_no_current(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        mock_find = MagicMock(return_value=None)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", mock_find),
        ):
            worker.run()
        mock_find.assert_called_once_with(mock_ctx.fido_dir, repo_ctx)

    def test_run_skips_find_next_when_current_issue_exists(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix bug", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_find = MagicMock(return_value=None)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "find_next_issue", mock_find),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(1, "my-branch", False)
            ),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
        ):
            worker.run()
        mock_find.assert_not_called()

    def test_run_returns_0_when_no_issue(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            assert worker.run() == 0

    def test_run_calls_post_pickup_comment_when_issue_found(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {
            "title": "Test issue",
            "body": "",
            "state": "OPEN",
        }
        worker = Worker(tmp_path, gh)
        mock_pickup = MagicMock()
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=5),
            patch.object(worker, "post_pickup_comment", mock_pickup),
            patch.object(
                worker, "find_or_create_pr", return_value=(1, "my-branch", False)
            ),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
        ):
            worker.run()
        mock_pickup.assert_called_once_with("owner/repo", 5, "Test issue", "fido-bot")

    def test_run_views_issue_for_title(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {
            "title": "Some title",
            "body": "",
            "state": "OPEN",
        }
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=3),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(1, "my-branch", False)
            ),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
        ):
            worker.run()
        gh.view_issue.assert_called_once_with("owner/repo", 3)

    def test_run_calls_find_or_create_pr_when_issue_found(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {
            "title": "My task",
            "body": "Issue body text",
            "state": "OPEN",
        }
        worker = Worker(tmp_path, gh)
        mock_focp = MagicMock(return_value=(42, "my-branch", False))
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=8),
            patch.object(worker, "post_pickup_comment"),
            patch.object(worker, "find_or_create_pr", mock_focp),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
        ):
            worker.run()
        mock_focp.assert_called_once_with(
            mock_ctx.fido_dir, repo_ctx, 8, "My task", "Issue body text"
        )

    def test_run_skips_ci_thread_rescope_for_fresh_pr(self, tmp_path: Path) -> None:
        """When find_or_create_pr returns is_fresh=True, skip rescope/CI/threads."""
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "New thing", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_rescope = MagicMock()
        mock_ci = MagicMock(return_value=False)
        mock_threads = MagicMock(return_value=False)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "create_session"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "new-thing", True)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "rescope_before_pick", mock_rescope),
            patch.object(worker, "handle_ci", mock_ci),
            patch.object(worker, "handle_threads", mock_threads),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=0),
        ):
            worker.run()
        mock_rescope.assert_not_called()
        mock_ci.assert_not_called()
        mock_threads.assert_not_called()

    def test_run_calls_ci_thread_rescope_for_existing_pr(self, tmp_path: Path) -> None:
        """When find_or_create_pr returns is_fresh=False, rescope/CI/threads run."""
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Old thing", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_rescope = MagicMock()
        mock_ci = MagicMock(return_value=False)
        mock_threads = MagicMock(return_value=False)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "create_session"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "old-thing", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "rescope_before_pick", mock_rescope),
            patch.object(worker, "handle_ci", mock_ci),
            patch.object(worker, "handle_threads", mock_threads),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=0),
        ):
            worker.run()
        mock_rescope.assert_called_once()
        mock_ci.assert_called_once()
        mock_threads.assert_called_once()

    def test_run_no_issue_never_reaches_checks(self, tmp_path: Path) -> None:
        """CI/thread/rescope checks are unreachable when no issue is selected."""
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        mock_rescope = MagicMock()
        mock_ci = MagicMock(return_value=False)
        mock_threads = MagicMock(return_value=False)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "create_session"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
            patch.object(worker, "rescope_before_pick", mock_rescope),
            patch.object(worker, "handle_ci", mock_ci),
            patch.object(worker, "handle_threads", mock_threads),
        ):
            result = worker.run()
        assert result == 0
        mock_rescope.assert_not_called()
        mock_ci.assert_not_called()
        mock_threads.assert_not_called()

    def test_run_second_iteration_runs_checks_after_fresh_pr(
        self, tmp_path: Path
    ) -> None:
        """After a fresh PR is created, the next iteration must run checks."""
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "t", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        iteration = 0

        def focp_side_effect(*_a: object, **_kw: object) -> tuple[int, str, bool]:
            nonlocal iteration
            iteration += 1
            # First call: fresh PR; second call: existing PR
            return (42, "fix-bug", iteration == 1)

        ci_calls: list[int] = []

        def ci_side_effect(*_a: object, **_kw: object) -> bool:
            ci_calls.append(iteration)
            return False

        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "create_session"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(worker, "find_or_create_pr", side_effect=focp_side_effect),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "rescope_before_pick"),
            patch.object(worker, "handle_ci", side_effect=ci_side_effect),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=0),
        ):
            worker.run()  # iteration 1: fresh PR — checks skipped
            worker.run()  # iteration 2: existing PR — checks run
        # handle_ci was only called on iteration 2
        assert ci_calls == [2]


class TestWorkerFindNextIssue:
    """Tests for Worker.find_next_issue."""

    def _make_worker(self, tmp_path: Path) -> tuple["Worker", MagicMock]:
        gh = MagicMock()
        provider = MagicMock()
        provider.api.get_limit_snapshot.return_value = ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE
        )
        return Worker(tmp_path, gh, provider=provider), gh

    def _make_repo_ctx(
        self,
        owner: str = "alice",
        repo_name: str = "proj",
        repo: str = "alice/proj",
        gh_user: str = "fido-bot",
    ) -> RepoContext:
        return RepoContext(
            repo=repo,
            owner=owner,
            repo_name=repo_name,
            gh_user=gh_user,
            default_branch="main",
            membership=RepoMembership(collaborators=frozenset({owner})),
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "fido"
        d.mkdir()
        return d

    def test_returns_none_when_no_issues(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_all_open_issues.return_value = []
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result is None

    def test_pauses_before_querying_when_provider_is_over_ninety_five_percent(
        self, tmp_path: Path
    ) -> None:
        gh = MagicMock()
        provider = MagicMock()
        provider.api.get_limit_snapshot.return_value = ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE,
            windows=(
                ProviderLimitWindow(
                    name="five_hour",
                    used=96,
                    limit=100,
                ),
            ),
        )
        worker = Worker(tmp_path, gh, provider=provider, repo_name="owner/repo")
        fido_dir = self._fido_dir(tmp_path)

        result = worker.find_next_issue(fido_dir, self._make_repo_ctx())

        assert result is None
        gh.find_all_open_issues.assert_not_called()
        gh.find_issues.assert_not_called()
        gh.set_user_status.assert_called_once_with(
            "Leaving the last 5% for the human until a little while.",
            ":sleeping:",
            busy=False,
        )

    def test_pause_message_uses_reset_time_when_known(self, tmp_path: Path) -> None:
        gh = MagicMock()
        provider = MagicMock()
        provider.api.get_limit_snapshot.return_value = ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE,
            windows=(
                ProviderLimitWindow(
                    name="five_hour",
                    used=97,
                    limit=100,
                    resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=timezone.utc),
                ),
            ),
        )
        worker = Worker(tmp_path, gh, provider=provider, repo_name="owner/repo")
        fido_dir = self._fido_dir(tmp_path)

        worker.find_next_issue(fido_dir, self._make_repo_ctx())

        gh.set_user_status.assert_called_once_with(
            "Leaving the last 5% for the human until 07:00 UTC.",
            ":sleeping:",
            busy=False,
        )

    def test_pause_reports_activity_to_registry(self, tmp_path: Path) -> None:
        gh = MagicMock()
        provider = MagicMock()
        provider.api.get_limit_snapshot.return_value = ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE,
            windows=(ProviderLimitWindow(name="five_hour", used=95, limit=100),),
        )
        registry = MagicMock()
        registry.status_update.return_value.__enter__.return_value = None
        registry.status_update.return_value.__exit__.return_value = None
        worker = Worker(
            tmp_path,
            gh,
            provider=provider,
            repo_name="owner/repo",
            registry=registry,
        )
        fido_dir = self._fido_dir(tmp_path)

        worker.find_next_issue(fido_dir, self._make_repo_ctx())

        registry.report_activity.assert_called_once_with(
            "owner/repo",
            "Paused for claude-code reset (95%, until a little while)",
            False,
        )

    def test_returns_issue_number_when_eligible_no_subissues(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        issue = {"number": 42, "title": "Do the thing", "subIssues": {"nodes": []}}
        gh.find_all_open_issues.return_value = [issue]
        gh.find_issues.return_value = [issue]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result == 42

    def test_returns_issue_number_when_all_subissues_closed(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        issue = {
            "number": 10,
            "title": "Parent task",
            "subIssues": {"nodes": [{"state": "CLOSED"}, {"state": "CLOSED"}]},
        }
        gh.find_all_open_issues.return_value = [issue]
        gh.find_issues.return_value = [issue]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result == 10

    def test_skips_tree_when_open_subissue_assigned_to_other(
        self, tmp_path: Path
    ) -> None:
        """If the only open child is someone else's, the whole tree is blocked."""
        worker, gh = self._make_worker(tmp_path)
        child = {
            "number": 30,
            "title": "Owned by other",
            "state": "OPEN",
            "assignees": {"nodes": [{"login": "someone-else"}]},
            "parent": {"number": 3},
            "subIssues": {"nodes": []},
        }
        parent_issue = {
            "number": 3,
            "title": "Blocked",
            "subIssues": {"nodes": [child]},
        }
        gh.find_all_open_issues.return_value = [parent_issue, child]
        gh.find_issues.return_value = [parent_issue]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result is None

    def test_picks_first_eligible_issue(self, tmp_path: Path) -> None:
        """With no sub-issues to descend into, picker uses creation order."""
        worker, gh = self._make_worker(tmp_path)
        issues = [
            {"number": 1, "title": "First", "subIssues": {"nodes": []}},
            {"number": 2, "title": "Second", "subIssues": {"nodes": []}},
        ]
        gh.find_all_open_issues.return_value = issues
        gh.find_issues.return_value = issues
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result == 1

    def test_saves_state_when_issue_found(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        issue = {"number": 7, "title": "Fetch!", "subIssues": {"nodes": []}}
        gh.find_all_open_issues.return_value = [issue]
        gh.find_issues.return_value = [issue]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        state = State(fido_dir).load()
        assert state["issue"] == 7
        assert state["issue_title"] == "Fetch!"
        assert "issue_started_at" in state

    def test_does_not_save_state_when_no_issue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_all_open_issues.return_value = []
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert State(fido_dir).load() == {}

    def test_calls_set_status_with_issue_info_when_found(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        issue = {"number": 5, "title": "Add tests", "subIssues": {"nodes": []}}
        gh.find_all_open_issues.return_value = [issue]
        gh.find_issues.return_value = [issue]
        fido_dir = self._fido_dir(tmp_path)
        mock_status = MagicMock()
        with patch.object(worker, "set_status", mock_status):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        mock_status.assert_called_once_with("Picking up issue #5: Add tests")

    def test_calls_set_status_done_when_no_issue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_all_open_issues.return_value = []
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        mock_status = MagicMock()
        with patch.object(worker, "set_status", mock_status):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        mock_status.assert_called_once_with("All done — no issues to fetch", busy=False)

    def test_passes_correct_args_to_find_issues(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_all_open_issues.return_value = []
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        repo_ctx = self._make_repo_ctx(owner="org", repo_name="myrepo", gh_user="bot")
        with patch.object(worker, "set_status"):
            worker.find_next_issue(fido_dir, repo_ctx)
        gh.find_all_open_issues.assert_called_once_with("org", "myrepo")
        gh.find_issues.assert_called_once_with("org", "myrepo", "bot")

    def test_logs_info_when_starting_issue(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        issue = {"number": 9, "title": "Chase squirrel", "subIssues": {"nodes": []}}
        gh.find_all_open_issues.return_value = [issue]
        gh.find_issues.return_value = [issue]
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert "9" in caplog.text

    def test_logs_info_when_no_eligible_issues(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.find_all_open_issues.return_value = []
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert "no eligible" in caplog.text

    def test_walks_up_via_issue_index(self, tmp_path: Path) -> None:
        """Assigned issue has a parent — worker uses the issue index built from
        find_all_open_issues to walk up to the root before descending."""
        worker, gh = self._make_worker(tmp_path)
        child = {
            "number": 200,
            "title": "child",
            "state": "OPEN",
            "assignees": {"nodes": [{"login": "fido-bot"}]},
            "parent": {"number": 100},
            "subIssues": {"nodes": []},
        }
        root = {
            "number": 100,
            "title": "root",
            "state": "OPEN",
            "assignees": {"nodes": []},
            "parent": None,
            "subIssues": {
                "nodes": [
                    {
                        "number": 200,
                        "title": "child",
                        "state": "OPEN",
                        "assignees": {"nodes": [{"login": "fido-bot"}]},
                        "parent": {"number": 100},
                    }
                ]
            },
        }
        gh.find_all_open_issues.return_value = [root, child]
        gh.find_issues.return_value = [child]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result == 200
        gh.find_all_open_issues.assert_called_once_with("alice", "proj")

    def test_picks_first_open_sub_issue_and_claims_it(self, tmp_path: Path) -> None:
        """When the assigned issue has an open, unassigned child, we claim
        and descend into that child — it must land before the parent."""
        worker, gh = self._make_worker(tmp_path)
        closed_child = {
            "number": 110,
            "title": "Closed work",
            "state": "CLOSED",
            "assignees": {"nodes": []},
            "parent": {"number": 11},
            "subIssues": {"nodes": []},
        }
        open_child = {
            "number": 111,
            "title": "Open child",
            "state": "OPEN",
            "assignees": {"nodes": []},
            "parent": {"number": 11},
            "subIssues": {"nodes": []},
        }
        parent_issue = {
            "number": 11,
            "title": "Parent",
            "subIssues": {"nodes": [closed_child, open_child]},
        }
        # find_all_open_issues returns only open issues; closed_child is absent
        gh.find_all_open_issues.return_value = [parent_issue, open_child]
        gh.find_issues.return_value = [parent_issue]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        # Picker chose the open child, and claimed it.
        assert result == 111
        gh.add_assignee.assert_called_once_with("alice/proj", 111, "fido-bot")


def _issue(
    number: int,
    title: str = "",
    *,
    state: str = "OPEN",
    milestone: str | None = None,
    assignees: list[str] | None = None,
    parent: int | None = None,
    sub_issues: list[dict] | None = None,
) -> dict:
    """Build a picker-shaped issue dict for tests."""
    node: dict = {
        "number": number,
        "title": title or f"issue {number}",
        "state": state,
        "assignees": {"nodes": [{"login": a} for a in (assignees or [])]},
    }
    if milestone is not None:
        node["milestone"] = {"title": milestone}
    if parent is not None:
        node["parent"] = {"number": parent}
    if sub_issues is not None:
        node["subIssues"] = {"nodes": sub_issues}
    return node


class TestNoCommitNudge:
    def test_early_attempt_is_gentle_and_includes_complete_command(self) -> None:
        from kennel.worker import _no_commit_nudge

        msg = _no_commit_nudge(1, "Fix widget", "task-7", "/repo/work", 42)
        assert "Fix widget" in msg
        assert "commit" in msg.lower()
        # Even early nudges include the exact mark-complete command so
        # claude can use it without guessing the task id.
        assert "kennel task complete /repo/work task-7" in msg
        # Early nudges don't threaten or list numbered actions.
        assert "attempt 1" not in msg.lower()
        assert "blocked" not in msg.lower()

    def test_late_attempt_offers_three_concrete_actions(self) -> None:
        from kennel.worker import _no_commit_nudge

        msg = _no_commit_nudge(3, "Fix widget", "task-7", "/repo/work", 42)
        assert "attempt 3" in msg.lower()
        # All three actions are concrete commands with real arguments.
        assert "git add" in msg.lower()
        assert "kennel task complete /repo/work task-7" in msg
        assert "gh pr comment 42 --body 'BLOCKED:" in msg

    def test_late_attempt_without_pr_uses_placeholder(self) -> None:
        from kennel.worker import _no_commit_nudge

        msg = _no_commit_nudge(3, "Fix widget", "task-7", "/repo/work", None)
        assert "gh pr comment <pr>" in msg


class TestFreshSessionNudge:
    def test_includes_context_reset_and_task_details(self) -> None:
        from kennel.worker import _fresh_session_nudge

        msg = _fresh_session_nudge("Fix widget", "task-7", "/repo/work", 42, "br-7")
        assert "session context was intentionally wiped" in msg
        assert "Task title: Fix widget" in msg
        assert "Task id: task-7" in msg
        assert "Branch: br-7" in msg
        assert "PR: 42" in msg
        assert "kennel task complete /repo/work task-7" in msg
        assert "gh pr comment 42 --body 'BLOCKED:" in msg


class TestPickNextIssue:
    """Direct unit tests for _pick_next_issue / _walk_to_root / _descend_issue."""

    def _claim_spy(self) -> tuple[list[int], Callable[[int], None]]:
        claimed: list[int] = []
        return claimed, claimed.append

    def test_single_assigned_issue_no_children_is_picked(self) -> None:
        from kennel.worker import _pick_next_issue

        issue = _issue(5, "Ready", assignees=["fido"], sub_issues=[])
        claimed, claim = self._claim_spy()
        choice = _pick_next_issue(
            [issue],
            "fido",
            issue_index={5: issue},
            claim=claim,
        )
        assert choice is not None and choice.number == 5
        assert claimed == []

    def test_milestone_beats_no_milestone(self) -> None:
        from kennel.worker import _pick_next_issue

        older = _issue(1, assignees=["fido"], sub_issues=[])
        newer_with_ms = _issue(99, assignees=["fido"], milestone="v1", sub_issues=[])
        choice = _pick_next_issue(
            [older, newer_with_ms],
            "fido",
            issue_index={1: older, 99: newer_with_ms},
            claim=lambda n: None,
        )
        assert choice is not None and choice.number == 99

    def test_walks_up_to_root_before_descending(self) -> None:
        """Assigned issue is a deep child — picker walks up to root, then
        descends to whatever's first-open at the top of the tree."""
        from kennel.worker import _pick_next_issue

        # Tree: #100 (root, unrelated) → [#200 (first, ours), #201 (second, ours)]
        # We're assigned #201 but #200 comes before it in sub-issue order, so
        # the descent from the root should pick #200.
        i200 = _issue(200, state="OPEN", assignees=["fido"], parent=100)
        i201 = _issue(201, state="OPEN", assignees=["fido"], parent=100)
        i100 = _issue(100, "root", sub_issues=[i200, i201])
        issue_index = {100: i100, 200: i200, 201: i201}

        # Only #201 is in the assigned list; picker must still walk up to 100.
        assigned = _issue(201, state="OPEN", assignees=["fido"], parent=100)
        choice = _pick_next_issue(
            [assigned],
            "fido",
            issue_index=issue_index,
            claim=lambda n: None,
        )
        assert choice is not None and choice.number == 200

    def test_blocked_when_earlier_sibling_is_someone_elses(self) -> None:
        from kennel.worker import _pick_next_issue

        # #100 → [#200 assigned to other, #201 assigned to fido]
        i200 = _issue(200, state="OPEN", assignees=["alice"], parent=100)
        i201 = _issue(201, state="OPEN", assignees=["fido"], parent=100)
        i100 = _issue(100, "root", sub_issues=[i200, i201])
        issue_index = {100: i100, 200: i200, 201: i201}
        # #200 blocks the earlier slot, but there's a later sibling (#201) that's ours.
        # The descent walks each open child in order — blocked, then eligible.
        choice = _pick_next_issue(
            [_issue(201, assignees=["fido"], parent=100)],
            "fido",
            issue_index=issue_index,
            claim=lambda n: None,
        )
        assert choice is not None and choice.number == 201

    def test_returns_none_when_only_open_children_are_others(self) -> None:
        from kennel.worker import _pick_next_issue

        i200 = _issue(200, state="OPEN", assignees=["alice"], parent=100)
        i201 = _issue(201, state="OPEN", assignees=["bob"], parent=100)
        i100 = _issue(100, "root", sub_issues=[i200, i201])
        issue_index = {100: i100, 200: i200, 201: i201}
        # Neither sibling is ours. Tree is blocked.
        choice = _pick_next_issue(
            [_issue(201, assignees=["bob"], parent=100)],
            "fido",
            issue_index=issue_index,
            claim=lambda n: None,
        )
        assert choice is None

    def test_claims_unassigned_child_then_descends(self) -> None:
        """Walks down a chain of unassigned descendants, claiming each."""
        from kennel.worker import _pick_next_issue

        # Tree: #100 → #200 (unassigned) → #300 (unassigned).
        # The full tree is in the index; descent claims both children.
        i300 = _issue(300, state="OPEN", assignees=[], parent=200, sub_issues=[])
        i200 = _issue(200, state="OPEN", assignees=[], parent=100, sub_issues=[i300])
        i100 = _issue(100, "root", assignees=["fido"], sub_issues=[i200])
        issue_index = {100: i100, 200: i200, 300: i300}

        claimed, claim = self._claim_spy()
        choice = _pick_next_issue(
            [i100],
            "fido",
            issue_index=issue_index,
            claim=claim,
        )
        assert choice is not None and choice.number == 300
        # Both #200 and #300 were claimed on the way down.
        assert claimed == [200, 300]

    def test_parent_cycle_is_broken_gracefully(self) -> None:
        """A self-referencing parent doesn't infinite-loop the walk."""
        from kennel.worker import _walk_to_root

        node = _issue(50, "A", parent=50)
        result = _walk_to_root(node, issue_index={50: node})
        # The walk bails at the first revisit; result is the node we end on.
        assert result["number"] == 50

    def test_reason_includes_descent_trail(self) -> None:
        from kennel.worker import _pick_next_issue

        # root → leaf, picker ends at leaf with descent trail in reason.
        leaf = _issue(20, state="OPEN", assignees=["fido"], parent=10)
        root = _issue(10, "root", assignees=["fido"], sub_issues=[leaf])
        issue_index = {10: root, 20: leaf}
        choice = _pick_next_issue(
            [root],
            "fido",
            issue_index=issue_index,
            claim=lambda n: None,
        )
        assert choice is not None
        assert choice.number == 20
        assert "#10" in choice.reason

    def test_dedupes_roots_across_multiple_assigned(self) -> None:
        """Two assigned issues sharing the same root should walk once."""
        from kennel.worker import _pick_next_issue

        child = _issue(2, state="OPEN", assignees=["fido"], parent=1)
        root = _issue(1, "root", sub_issues=[child])
        issue_index = {1: root, 2: child}

        a = _issue(2, state="OPEN", assignees=["fido"], parent=1)
        b = _issue(2, state="OPEN", assignees=["fido"], parent=1)
        choice = _pick_next_issue(
            [a, b],
            "fido",
            issue_index=issue_index,
            claim=lambda n: None,
        )
        assert choice is not None and choice.number == 2
        # Walks up for each of the two, but descent runs exactly once (roots are deduped).

    def test_milestone_inherited_via_parent_note_in_reason(self) -> None:
        from kennel.worker import _pick_next_issue

        child = _issue(20, state="OPEN", assignees=["fido"], parent=10)
        parent = _issue(10, "parent", milestone="v1", sub_issues=[child])
        issue_index = {10: parent, 20: child}
        assigned = _issue(20, state="OPEN", assignees=["fido"], parent=10)
        choice = _pick_next_issue(
            [assigned],
            "fido",
            issue_index=issue_index,
            claim=lambda n: None,
        )
        assert choice is not None
        assert choice.number == 20
        assert "milestone from parent #10" in choice.reason

    def test_missing_parent_in_index_treats_current_as_root(self) -> None:
        """When a candidate's parent is absent from the index (closed/blocked),
        the walk stops and treats the current node as the root."""
        from kennel.worker import _walk_to_root

        # #200 claims parent #100, but #100 is not in the index (closed).
        child = _issue(200, state="OPEN", assignees=["fido"], parent=100)
        result = _walk_to_root(child, issue_index={200: child})
        # Walk stops; the child itself is treated as the root.
        assert result["number"] == 200


class TestWorkerPostPickupComment:
    """Tests for Worker.post_pickup_comment."""

    def _make_worker(
        self, tmp_path: Path, provider_agent: MagicMock | None = None
    ) -> tuple["Worker", MagicMock]:
        gh = MagicMock()
        gh.view_issue.return_value = {"created_at": "2024-01-01T00:00:00Z"}
        gh.get_issue_events.return_value = []
        return Worker(tmp_path, gh, provider_agent=provider_agent), gh

    def test_skips_when_already_commented(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_issue_comments.return_value = [
            {
                "user": {"login": "fido-bot"},
                "body": "Woof!",
                "created_at": "2024-02-01T00:00:00Z",
            }
        ]
        worker.post_pickup_comment("owner/repo", 1, "Fix bug", "fido-bot")
        gh.comment_issue.assert_not_called()

    def test_posts_comment_when_no_previous_comment(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_reply.return_value = "Woof! On it!"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        worker._prompts = Prompts("I am Fido.")
        gh.get_issue_comments.return_value = [
            {"user": {"login": "other-user"}, "body": "Hi"}
        ]
        worker.post_pickup_comment("owner/repo", 1, "Fix bug", "fido-bot")
        gh.comment_issue.assert_called_once_with("owner/repo", 1, "Woof! On it!")

    def test_posts_comment_when_no_existing_comments(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_reply.return_value = "I am on it!"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        worker._prompts = Prompts("I am Fido.")
        gh.get_issue_comments.return_value = []
        worker.post_pickup_comment("owner/repo", 3, "Some task", "fido-bot")
        gh.comment_issue.assert_called_once()

    def test_falls_back_to_plain_text_when_claude_returns_empty(
        self, tmp_path: Path
    ) -> None:
        mock_client = _client()
        mock_client.generate_reply.return_value = ""
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        worker._prompts = Prompts("I am Fido.")
        gh.get_issue_comments.return_value = []
        worker.post_pickup_comment("owner/repo", 5, "A task", "fido-bot")
        gh.comment_issue.assert_called_once_with(
            "owner/repo", 5, "Picking up issue: A task"
        )

    def test_uses_persona_from_sub_dir(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_reply.return_value = "Fetched!"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        worker._prompts = Prompts("I am a very good dog.")
        gh.get_issue_comments.return_value = []
        worker.post_pickup_comment("owner/repo", 2, "Some work", "fido-bot")
        prompt_arg = mock_client.generate_reply.call_args[0][0]
        assert "I am a very good dog." in prompt_arg

    def test_falls_back_to_empty_persona_on_oserror(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_reply.return_value = "On it!"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        worker._prompts = Prompts("")
        gh.get_issue_comments.return_value = []
        worker.post_pickup_comment("owner/repo", 2, "Work item", "fido-bot")
        prompt_arg = mock_client.generate_reply.call_args[0][0]
        assert "Picking up issue: Work item" in prompt_arg

    def test_prompt_includes_issue_title(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_reply.return_value = "On it!"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        worker._prompts = Prompts("")
        gh.get_issue_comments.return_value = []
        worker.post_pickup_comment("owner/repo", 4, "Refactor auth", "fido-bot")
        prompt_arg = mock_client.generate_reply.call_args[0][0]
        assert "Refactor auth" in prompt_arg

    def test_checks_comments_for_correct_repo_and_issue(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_reply.return_value = "Arf!"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        worker._prompts = Prompts("")
        gh.get_issue_comments.return_value = []
        worker.post_pickup_comment("org/myrepo", 99, "Title", "fido-bot")
        gh.get_issue_comments.assert_called_once_with("org/myrepo", 99)

    def test_logs_info_when_skipping(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_client = _client()
        mock_client.generate_reply.return_value = "Woof!"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.get_issue_comments.return_value = [
            {
                "user": {"login": "fido-bot"},
                "body": "Woof!",
                "created_at": "2024-02-01T00:00:00Z",
            }
        ]
        with caplog.at_level(logging.INFO, logger="kennel"):
            worker.post_pickup_comment("owner/repo", 7, "Title", "fido-bot")
        assert "already exists" in caplog.text

    def test_posts_comment_on_reopened_issue(self, tmp_path: Path) -> None:
        """Old comment predates reopen, so a new pickup comment is posted."""
        mock_client = _client()
        mock_client.generate_reply.return_value = "Back on it!"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.get_issue_events.return_value = [
            {"event": "reopened", "created_at": "2024-06-01T00:00:00Z"}
        ]
        gh.get_issue_comments.return_value = [
            {
                "user": {"login": "fido-bot"},
                "body": "Woof!",
                "created_at": "2024-02-01T00:00:00Z",
            }
        ]
        worker._prompts = Prompts("I am Fido.")
        worker.post_pickup_comment("owner/repo", 1, "Fix bug", "fido-bot")
        gh.comment_issue.assert_called_once_with("owner/repo", 1, "Back on it!")


class TestRun:
    def test_creates_worker_and_calls_run(self, tmp_path: Path) -> None:
        mock_worker = MagicMock()
        mock_worker.run.return_value = 0
        mock_gh_cls = MagicMock()
        with patch("kennel.worker.Worker", return_value=mock_worker) as mock_worker_cls:
            result = run(tmp_path, _GitHub=mock_gh_cls)
        mock_worker_cls.assert_called_once_with(tmp_path, mock_gh_cls.return_value)
        mock_worker.run.assert_called_once()
        assert result == 0

    def test_returns_worker_run_result(self, tmp_path: Path) -> None:
        mock_worker = MagicMock()
        mock_worker.run.return_value = 2
        with patch("kennel.worker.Worker", return_value=mock_worker):
            assert run(tmp_path, _GitHub=MagicMock()) == 2


class TestCreateCompactScript:
    def test_creates_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        assert script.exists()

    def test_returns_path_in_fido_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        assert script == fido_dir / "compact.sh"

    def test_script_is_executable(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        assert script.stat().st_mode & 0o111  # any execute bit

    def test_script_has_shebang(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        assert script.read_text().startswith("#!/usr/bin/env bash\n")

    def test_script_references_sub_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        from kennel.worker import _sub_dir

        assert str(_sub_dir()) in script.read_text()

    def test_script_contains_post_compact_message(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        assert "PostCompact" in script.read_text()

    def test_script_contains_md_glob(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        assert "*.md" in script.read_text()


class TestSetupHooks:
    def test_returns_compact_and_sync_cmds(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        compact_cmd, sync_cmd = Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        assert compact_cmd.startswith("bash ")
        assert "sync-tasks" in sync_cmd

    def test_compact_cmd_references_compact_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        compact_cmd, _ = Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        assert "compact.sh" in compact_cmd

    def test_sync_cmd_references_sync_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        _, sync_cmd = Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        assert "kennel sync-tasks" in sync_cmd

    def test_sync_cmd_includes_work_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        _, sync_cmd = Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        assert str(tmp_path) in sync_cmd

    def test_creates_compact_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        assert (fido_dir / "compact.sh").exists()

    def test_adds_hooks_to_settings(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        settings = tmp_path / ".claude" / "settings.local.json"
        assert settings.exists()
        cfg = json.loads(settings.read_text())
        assert "hooks" in cfg

    def test_gitexcludes_settings(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        exclude = tmp_path / ".git" / "info" / "exclude"
        assert ".claude/settings.local.json" in exclude.read_text()


class TestTeardownHooks:
    def test_removes_compact_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        worker = Worker(tmp_path, MagicMock())
        compact_cmd, sync_cmd = worker.setup_hooks(fido_dir)
        worker.teardown_hooks(fido_dir, compact_cmd, sync_cmd)
        assert not (fido_dir / "compact.sh").exists()

    def test_removes_hooks_from_settings(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        worker = Worker(tmp_path, MagicMock())
        compact_cmd, sync_cmd = worker.setup_hooks(fido_dir)
        worker.teardown_hooks(fido_dir, compact_cmd, sync_cmd)
        settings = tmp_path / ".claude" / "settings.local.json"
        cfg = json.loads(settings.read_text())
        assert "hooks" not in cfg

    def test_noop_when_compact_script_missing(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        # Should not raise even when compact.sh does not exist
        Worker(tmp_path, MagicMock()).teardown_hooks(
            fido_dir, "bash /x/compact.sh", "bash sync.sh &"
        )

    def test_noop_when_settings_missing(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        # No settings file created — should not raise
        Worker(tmp_path, MagicMock()).teardown_hooks(
            fido_dir, "bash /x/compact.sh", "bash sync.sh &"
        )


class TestLoadState:
    def test_returns_empty_dict_when_absent(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        assert State(fido_dir).load() == {}

    def test_returns_state_when_present(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "state.json").write_text(json.dumps({"issue": 42}))
        assert State(fido_dir).load() == {"issue": 42}

    def test_returns_dict_with_arbitrary_keys(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "state.json").write_text(json.dumps({"issue": 7, "extra": "val"}))
        result = State(fido_dir).load()
        assert result["issue"] == 7
        assert result["extra"] == "val"


class TestSaveState:
    def test_creates_state_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 5})
        assert (fido_dir / "state.json").exists()

    def test_roundtrips_with_load_state(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 99})
        assert State(fido_dir).load() == {"issue": 99}

    def test_overwrites_existing_state(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 1})
        State(fido_dir).save({"issue": 2})
        assert State(fido_dir).load() == {"issue": 2}


class TestClearState:
    def test_removes_state_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 3})
        State(fido_dir).clear()
        assert not (fido_dir / "state.json").exists()

    def test_noop_when_absent(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        # Should not raise
        State(fido_dir).clear()

    def test_load_returns_empty_after_clear(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 10})
        State(fido_dir).clear()
        assert State(fido_dir).load() == {}


class TestState:
    def test_load_returns_empty_when_fido_dir_absent(self, tmp_path: Path) -> None:
        state = State(tmp_path / "nonexistent")
        assert state.load() == {}

    def test_load_returns_state_when_present(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "state.json").write_text(json.dumps({"issue": 7}))
        assert State(fido_dir).load() == {"issue": 7}

    def test_save_persists_data(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 42})
        assert State(fido_dir).load() == {"issue": 42}

    def test_clear_removes_state_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        State(fido_dir).save({"issue": 1})
        State(fido_dir).clear()
        assert not (fido_dir / "state.json").exists()


class TestBuildPrompt:
    """Tests for build_prompt."""

    def _setup_sub_dir(self, tmp_path: Path) -> Path:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "persona.md").write_text("I am Fido, a very good dog.")
        (sub / "task.md").write_text("Implement the task carefully.")
        return sub

    def test_returns_system_and_prompt_paths(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, prompt_file = build_prompt(fido_dir, "task", "context")
        assert sys_file == fido_dir / "system"
        assert prompt_file == fido_dir / "prompt"

    def test_system_file_contains_persona(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "task", "context")
        assert "I am Fido" in sys_file.read_text()

    def test_system_file_contains_skill(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "task", "context")
        assert "Implement the task carefully." in sys_file.read_text()

    def test_system_file_joins_with_blank_line(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "task", "context")
        content = sys_file.read_text()
        assert "I am Fido, a very good dog.\n\nImplement the task carefully." in content

    def test_prompt_file_contains_context(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            _, prompt_file = build_prompt(fido_dir, "task", "do the work")
        assert "do the work" in prompt_file.read_text()

    def test_system_file_ends_with_newline(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "task", "ctx")
        assert sys_file.read_text().endswith("\n")

    def test_prompt_file_ends_with_newline(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            _, prompt_file = build_prompt(fido_dir, "task", "ctx")
        assert prompt_file.read_text().endswith("\n")

    def test_uses_correct_subskill_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        (sub / "ci.md").write_text("Fix the CI failure.")
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "ci", "ctx")
        assert "Fix the CI failure." in sys_file.read_text()
        assert "Implement the task carefully." not in sys_file.read_text()

    def test_strips_trailing_whitespace_from_persona(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        (sub / "persona.md").write_text("Persona text\n\n\n")
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "task", "ctx")
        content = sys_file.read_text()
        assert content.startswith("Persona text\n\n")
        assert not content.startswith("Persona text\n\n\n\n")


class TestProviderStart:
    """Tests for provider_start."""

    def _setup_fido_dir(self, tmp_path: Path) -> Path:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "system").write_text("system prompt")
        (fido_dir / "skill").write_text("sub-skill instructions")
        (fido_dir / "prompt").write_text("user prompt")
        return fido_dir

    def test_returns_session_id_on_success(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        output = '{"type":"result","session_id":"sess-abc"}'
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = output
        mock_client.extract_session_id.return_value = "sess-abc"
        result = provider_start(
            fido_dir,
            agent=mock_client,
            model=mock_client.voice_model,
        )
        assert result == "sess-abc"

    def test_returns_empty_when_extract_fails(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        result = provider_start(
            fido_dir,
            agent=mock_client,
            model=mock_client.voice_model,
        )
        assert result == ""

    def test_calls_print_prompt_from_file_with_correct_files(
        self, tmp_path: Path
    ) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        provider_start(
            fido_dir,
            agent=mock_client,
            model=mock_client.voice_model,
        )
        mock_client.print_prompt_from_file.assert_called_once_with(
            fido_dir / "system",
            fido_dir / "prompt",
            "claude-opus-4-6",
            300,
            cwd=".",
        )

    def test_passes_custom_model(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        provider_start(
            fido_dir,
            agent=mock_client,
            model="claude-sonnet-4-6",
        )
        assert mock_client.print_prompt_from_file.call_args[0][2] == "claude-sonnet-4-6"

    def test_passes_custom_timeout(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        provider_start(
            fido_dir,
            agent=mock_client,
            model=mock_client.voice_model,
            timeout=600,
        )
        assert mock_client.print_prompt_from_file.call_args[0][3] == 600

    def test_default_model_is_opus(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        provider_start(
            fido_dir,
            agent=mock_client,
            model=mock_client.voice_model,
        )
        assert mock_client.print_prompt_from_file.call_args[0][2] == "claude-opus-4-6"

    def test_default_timeout_is_300(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        provider_start(
            fido_dir,
            agent=mock_client,
            model=mock_client.voice_model,
        )
        assert mock_client.print_prompt_from_file.call_args[0][3] == 300

    def test_passes_output_to_extract_session_id(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        raw = '{"type":"result","session_id":"xyz"}'
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = raw
        mock_client.extract_session_id.return_value = "xyz"
        provider_start(
            fido_dir,
            agent=mock_client,
            model=mock_client.voice_model,
        )
        mock_client.extract_session_id.assert_called_once_with(raw)

    def test_requires_agent(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with pytest.raises(ValueError, match="provider_start requires agent"):
            provider_start(fido_dir, model=ProviderModel("claude-opus-4-6"))

    # ── Session path ──────────────────────────────────────────────────────

    def test_session_path_returns_empty_string(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        session = self._mock_session()
        client = _client()
        client.session = session
        result = provider_start(fido_dir, agent=client, model=client.voice_model)
        assert result == ""

    def test_session_path_uses_agent_run_turn(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        session = self._mock_session()
        client = _client()
        client.session = session
        provider_start(fido_dir, agent=client, model=client.voice_model)
        client.run_turn.assert_called_once()
        assert client.run_turn.call_args.kwargs["retry_on_preempt"] is True

    def _mock_session(self) -> MagicMock:
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=None)
        session.last_turn_cancelled = False
        return session

    def test_session_path_sends_prompt_content(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        (fido_dir / "skill").write_text("setup instructions")
        (fido_dir / "prompt").write_text("the task prompt")
        session = self._mock_session()
        client = _client()
        client.session = session
        provider_start(fido_dir, agent=client, model=client.voice_model)
        client.run_turn.assert_called_once_with(
            "setup instructions\n\n---\n\nthe task prompt",
            model=client.voice_model,
            retry_on_preempt=True,
            session_mode=TurnSessionMode.REUSE,
        )

    def test_session_path_calls_agent_once(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        session = self._mock_session()
        client = _client()
        client.session = session
        provider_start(fido_dir, agent=client, model=client.voice_model)
        client.run_turn.assert_called_once()

    def test_session_path_does_not_call_subprocess(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        session = self._mock_session()
        mock_client = _client()
        mock_client.session = session
        provider_start(fido_dir, agent=mock_client, model=mock_client.voice_model)
        mock_client.print_prompt_from_file.assert_not_called()

    def test_session_path_uses_context_manager(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        session = self._mock_session()
        client = _client()
        client.session = session
        provider_start(fido_dir, agent=client, model=client.voice_model)
        session.__enter__.assert_not_called()
        session.__exit__.assert_not_called()


class TestProviderRun:
    """Tests for provider_run."""

    def _setup_fido_dir(self, tmp_path: Path) -> Path:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "system").write_text("system")
        (fido_dir / "skill").write_text("skill")
        (fido_dir / "prompt").write_text("prompt")
        return fido_dir

    # ── Start path ─────────────────────────────────────────────────────────

    def test_start_returns_new_session_id(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        raw = '{"type":"result","session_id":"new-sess"}'
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = raw
        mock_client.extract_session_id.return_value = "new-sess"
        session_id, _ = provider_run(
            fido_dir,
            agent=mock_client,
            model=mock_client.work_model,
        )
        assert session_id == "new-sess"

    def test_start_returns_raw_output(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        raw = '{"type":"result","session_id":"s"}'
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = raw
        _, output = provider_run(
            fido_dir,
            agent=mock_client,
            model=mock_client.work_model,
        )
        assert output == raw

    def test_start_calls_print_prompt_from_file(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        provider_run(
            fido_dir,
            agent=mock_client,
            model=mock_client.work_model,
        )
        mock_client.print_prompt_from_file.assert_called_once_with(
            fido_dir / "system",
            fido_dir / "prompt",
            "claude-sonnet-4-6",
            300,
            cwd=".",
        )

    def test_start_returns_empty_session_id_on_failure(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        session_id, _ = provider_run(
            fido_dir,
            agent=mock_client,
            model=mock_client.work_model,
        )
        assert session_id == ""

    def test_passes_custom_model(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        provider_run(
            fido_dir,
            agent=mock_client,
            model="claude-opus-4-6",
        )
        assert mock_client.print_prompt_from_file.call_args[0][2] == "claude-opus-4-6"

    def test_default_model_is_sonnet(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        provider_run(
            fido_dir,
            agent=mock_client,
            model=mock_client.work_model,
        )
        assert mock_client.print_prompt_from_file.call_args[0][2] == "claude-sonnet-4-6"

    def test_default_timeout_is_300(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        mock_client = _client()
        mock_client.print_prompt_from_file.return_value = ""
        provider_run(
            fido_dir,
            agent=mock_client,
            model=mock_client.work_model,
        )
        assert mock_client.print_prompt_from_file.call_args[0][3] == 300

    def test_requires_agent(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with pytest.raises(ValueError, match="provider_run requires agent"):
            provider_run(fido_dir, model=ProviderModel("claude-sonnet-4-6"))

    # ── Session path ──────────────────────────────────────────────────────

    def test_session_path_returns_empty_tuple(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        session = self._mock_session()
        client = _client()
        client.session = session
        result = provider_run(fido_dir, agent=client, model=client.work_model)
        assert result == ("", "")

    def _mock_session(self) -> MagicMock:
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=None)
        session.last_turn_cancelled = False
        return session

    def test_session_path_sends_prompt_content(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        (fido_dir / "skill").write_text("task instructions")
        (fido_dir / "prompt").write_text("run this task")
        session = self._mock_session()
        client = _client()
        client.session = session
        provider_run(fido_dir, agent=client, model=client.work_model)
        client.run_turn.assert_called_once_with(
            "task instructions\n\n---\n\nrun this task",
            model=client.work_model,
            retry_on_preempt=True,
            session_mode=TurnSessionMode.REUSE,
        )

    def test_session_path_calls_agent_once(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        session = self._mock_session()
        client = _client()
        client.session = session
        provider_run(fido_dir, agent=client, model=client.work_model)
        client.run_turn.assert_called_once()

    def test_session_path_does_not_call_subprocess(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        session = self._mock_session()
        mock_client = _client()
        mock_client.session = session
        provider_run(fido_dir, agent=mock_client, model=mock_client.work_model)
        mock_client.print_prompt_from_file.assert_not_called()

    def test_session_path_uses_context_manager(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        session = self._mock_session()
        client = _client()
        client.session = session
        provider_run(fido_dir, agent=client, model=client.work_model)
        session.__enter__.assert_not_called()
        session.__exit__.assert_not_called()


class TestSanitizeSlug:
    """Tests for _sanitize_slug."""

    def test_lowercases_input(self) -> None:
        assert _sanitize_slug("AddFeature", "fallback text") == "addfeature"

    def test_replaces_special_chars_with_hyphens(self) -> None:
        result = _sanitize_slug("add feature test", "fallback")
        assert result == "add-feature-test"

    def test_strips_leading_trailing_hyphens(self) -> None:
        result = _sanitize_slug("  add-feature  ", "fallback")
        assert result == "add-feature"

    def test_truncates_at_40_chars(self) -> None:
        long_input = "a" * 50
        result = _sanitize_slug(long_input, "fallback")
        assert len(result) <= 40

    def test_falls_back_when_too_short(self) -> None:
        result = _sanitize_slug("ab", "fix the login bug")
        assert "fix" in result or "login" in result or "bug" in result

    def test_falls_back_strips_closes_clause(self) -> None:
        result = _sanitize_slug("x", "fix bug (closes #42)")
        assert "42" not in result
        assert "closes" not in result

    def test_three_char_slug_is_valid(self) -> None:
        result = _sanitize_slug("abc", "fallback text")
        assert result == "abc"

    def test_two_char_slug_falls_back(self) -> None:
        result = _sanitize_slug("ab", "fix login issue")
        assert len(result) >= 3

    def test_empty_raw_falls_back(self) -> None:
        result = _sanitize_slug("", "implement auth")
        assert len(result) >= 3

    def test_fallback_also_sanitized(self) -> None:
        result = _sanitize_slug("x", "Add New Feature!")
        assert result == result.lower()
        assert "!" not in result


class TestSanitizeStatusText:
    """Tests for _sanitize_status_text."""

    def test_plain_text_unchanged(self) -> None:
        assert _sanitize_status_text("working on tests") == "working on tests"

    def test_strips_leading_whitespace(self) -> None:
        assert _sanitize_status_text("  hello") == "hello"

    def test_strips_trailing_whitespace(self) -> None:
        assert _sanitize_status_text("hello  ") == "hello"

    def test_collapses_newline_to_space(self) -> None:
        assert _sanitize_status_text("line one\nline two") == "line one line two"

    def test_collapses_newline_with_surrounding_whitespace(self) -> None:
        assert _sanitize_status_text("line one  \n  line two") == "line one line two"

    def test_collapses_multiple_newlines(self) -> None:
        assert _sanitize_status_text("a\nb\nc") == "a b c"

    def test_strips_and_collapses_combined(self) -> None:
        assert _sanitize_status_text("  foo\nbar  ") == "foo bar"


class TestParseStatusNudge:
    """Tests for _parse_status_nudge."""

    def test_empty_raw_returns_empty_tuple(self) -> None:
        from kennel.worker import _parse_status_nudge

        assert _parse_status_nudge("") == ("", "")

    def test_valid_json_returns_both_fields(self) -> None:
        from kennel.worker import _parse_status_nudge

        assert _parse_status_nudge('{"status": "chasing bugs", "emoji": ":dog:"}') == (
            "chasing bugs",
            ":dog:",
        )

    def test_json_with_preamble_still_parsed(self) -> None:
        from kennel.worker import _parse_status_nudge

        assert _parse_status_nudge(
            'Here you go: {"status": "ok", "emoji": ":wrench:"} thanks!'
        ) == ("ok", ":wrench:")

    def test_malformed_json_returns_empty_tuple(self) -> None:
        from kennel.worker import _parse_status_nudge

        assert _parse_status_nudge("{not json at all}") == ("", "")

    def test_missing_fields_returns_empty_tuple(self) -> None:
        from kennel.worker import _parse_status_nudge

        # Valid JSON but wrong shape — no status/emoji string fields
        assert _parse_status_nudge('{"other": "value"}') == ("", "")


class TestGit:
    """Tests for Worker._git helper."""

    def test_calls_subprocess_run_with_git_prefix(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        Worker(tmp_path, MagicMock())._git(["status"], _run=mock_run)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "git"
        assert args[1] == "status"

    def test_passes_work_dir_as_cwd(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        Worker(tmp_path, MagicMock())._git(["status"], _run=mock_run)
        assert mock_run.call_args[1]["cwd"] == tmp_path

    def test_check_true_by_default(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        Worker(tmp_path, MagicMock())._git(["status"], _run=mock_run)
        assert mock_run.call_args[1]["check"] is True

    def test_check_false_propagated(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(returncode=1))
        Worker(tmp_path, MagicMock())._git(["status"], check=False, _run=mock_run)
        assert mock_run.call_args[1]["check"] is False

    def test_propagates_called_process_error(self, tmp_path: Path) -> None:
        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(1, "git"))
        with pytest.raises(subprocess.CalledProcessError):
            Worker(tmp_path, MagicMock())._git(
                ["checkout", "no-such-branch"], _run=mock_run
            )

    def test_capture_output_and_text_set(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        Worker(tmp_path, MagicMock())._git(["log"], _run=mock_run)
        assert mock_run.call_args[1]["capture_output"] is True
        assert mock_run.call_args[1]["text"] is True


class TestGitClean:
    """Tests for Worker.git_clean."""

    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def _git_result(self, stdout: str = "") -> MagicMock:
        r = MagicMock()
        r.stdout = stdout
        return r

    def test_runs_checkout_to_restore_tracked_files(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        calls: list[list[str]] = []

        def capture(args, **kwargs):
            calls.append(args)
            return self._git_result()

        with patch.object(worker, "_git", side_effect=capture):
            worker.git_clean()

        assert ["checkout", "--", "."] in calls

    def test_runs_clean_to_remove_untracked_files(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        calls: list[list[str]] = []

        def capture(args, **kwargs):
            calls.append(args)
            return self._git_result()

        with patch.object(worker, "_git", side_effect=capture):
            worker.git_clean()

        assert ["clean", "-fd"] in calls

    def test_logs_removed_files_when_output_present(
        self, tmp_path: Path, caplog
    ) -> None:
        worker = self._make_worker(tmp_path)

        def fake_git(args, **kwargs):
            if args == ["clean", "-fd"]:
                return self._git_result("Removing foo.py\nRemoving bar/")
            return self._git_result()

        import logging

        with patch.object(worker, "_git", side_effect=fake_git):
            with caplog.at_level(logging.INFO):
                worker.git_clean()

        assert "Removing foo.py" in caplog.text

    def test_logs_nothing_to_remove_when_output_empty(
        self, tmp_path: Path, caplog
    ) -> None:
        worker = self._make_worker(tmp_path)

        with patch.object(worker, "_git", return_value=self._git_result("")):
            import logging

            with caplog.at_level(logging.INFO):
                worker.git_clean()

        assert "nothing to remove" in caplog.text

    def test_checkout_called_before_clean(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        order: list[str] = []

        def capture(args, **kwargs):
            if args == ["checkout", "--", "."]:
                order.append("checkout")
            elif args == ["clean", "-fd"]:
                order.append("clean")
            return self._git_result()

        with patch.object(worker, "_git", side_effect=capture):
            worker.git_clean()

        assert order == ["checkout", "clean"]


class TestExtractBody:
    """Tests for the module-level _extract_body helper."""

    def test_extracts_between_tags(self) -> None:
        from kennel.worker import _extract_body

        assert _extract_body("<body>hello</body>") == "hello"

    def test_strips_surrounding_whitespace(self) -> None:
        from kennel.worker import _extract_body

        assert _extract_body("<body>\n  hello  \n</body>") == "hello"

    def test_returns_empty_when_no_tags(self) -> None:
        from kennel.worker import _extract_body

        assert _extract_body("no tags here") == ""

    def test_returns_empty_for_none(self) -> None:
        from kennel.worker import _extract_body

        assert _extract_body(None) == ""

    def test_returns_empty_for_empty_string(self) -> None:
        from kennel.worker import _extract_body

        assert _extract_body("") == ""

    def test_handles_preamble_and_trailing_chatter(self) -> None:
        from kennel.worker import _extract_body

        raw = "Here's my take:\n<body>the real content</body>\nHope that helps!"
        assert _extract_body(raw) == "the real content"

    def test_handles_multiline_body(self) -> None:
        from kennel.worker import _extract_body

        raw = "<body>line 1\n\nline 2\n\nFixes #5.</body>"
        assert _extract_body(raw) == "line 1\n\nline 2\n\nFixes #5."

    def test_case_insensitive(self) -> None:
        from kennel.worker import _extract_body

        assert _extract_body("<BODY>upper</BODY>") == "upper"


class TestWritePrDescription:
    """Tests for the module-level _write_pr_description function."""

    def _pending_task(self, title: str, task_type: str = "spec") -> dict:
        return {"id": "1", "title": title, "status": "pending", "type": task_type}

    def _call(
        self,
        gh,
        task_list=None,
        existing_body="",
        print_return="<body>Desc.\n\nFixes #1.</body>",
        issue=1,
        pr_number=42,
    ):
        mock_cc = _client(print_return)
        return (
            _write_pr_description(
                gh,
                "owner/repo",
                pr_number,
                issue,
                task_list or [],
                existing_body,
                agent=mock_cc,
            ),
            mock_cc,
        )

    def test_writes_to_github(self) -> None:
        gh = MagicMock()
        self._call(gh)
        gh.edit_pr_body.assert_called_once()

    def test_raises_when_opus_returns_empty(self) -> None:
        gh = MagicMock()
        with pytest.raises(ValueError, match="no <body> content"):
            self._call(gh, print_return="", issue=7)
        gh.edit_pr_body.assert_not_called()

    def test_raises_when_no_divider_in_existing_body(self) -> None:
        gh = MagicMock()
        with pytest.raises(ValueError, match="no --- divider"):
            self._call(gh, existing_body="no divider here")
        gh.edit_pr_body.assert_not_called()

    def test_initial_write_contains_work_queue_start_marker(self) -> None:
        gh = MagicMock()
        self._call(gh)
        body = gh.edit_pr_body.call_args[0][2]
        assert "<!-- WORK_QUEUE_START -->" in body

    def test_initial_write_contains_work_queue_end_marker(self) -> None:
        gh = MagicMock()
        self._call(gh)
        body = gh.edit_pr_body.call_args[0][2]
        assert "<!-- WORK_QUEUE_END -->" in body

    def test_initial_write_contains_separator(self) -> None:
        gh = MagicMock()
        self._call(gh)
        body = gh.edit_pr_body.call_args[0][2]
        assert "---" in body

    def test_pending_tasks_shown_as_checkboxes(self) -> None:
        gh = MagicMock()
        tasks = [self._pending_task("Write tests"), self._pending_task("Fix lint")]
        self._call(gh, task_list=tasks)
        body = gh.edit_pr_body.call_args[0][2]
        assert "- [ ] Write tests" in body
        assert "- [ ] Fix lint" in body

    def test_first_task_has_next_marker(self) -> None:
        gh = MagicMock()
        tasks = [self._pending_task("First task"), self._pending_task("Second task")]
        self._call(gh, task_list=tasks)
        body = gh.edit_pr_body.call_args[0][2]
        assert "- [ ] First task **→ next**" in body

    def test_second_task_has_no_next_marker(self) -> None:
        gh = MagicMock()
        tasks = [self._pending_task("First task"), self._pending_task("Second task")]
        self._call(gh, task_list=tasks)
        body = gh.edit_pr_body.call_args[0][2]
        assert "- [ ] Second task **→ next**" not in body
        assert "- [ ] Second task\n" in body or body.endswith("- [ ] Second task")

    def test_next_marker_follows_pick_next_task_priority(self) -> None:
        """CI failure task second in list should still get the → next marker."""
        gh = MagicMock()
        regular = self._pending_task("Regular work")
        ci = {"id": "2", "title": "CI failure: lint", "status": "pending", "type": "ci"}
        self._call(gh, task_list=[regular, ci])
        body = gh.edit_pr_body.call_args[0][2]
        assert "- [ ] CI failure: lint **→ next**" in body
        assert "- [ ] Regular work **→ next**" not in body

    def test_no_tasks_shows_placeholder(self) -> None:
        gh = MagicMock()
        self._call(gh, task_list=[])
        body = gh.edit_pr_body.call_args[0][2]
        assert "<!-- no tasks yet -->" in body

    def test_skips_completed_tasks(self) -> None:
        gh = MagicMock()
        task_list = [
            {"id": "1", "title": "Done task", "status": "completed"},
            {"id": "2", "title": "Pending task", "status": "pending"},
        ]
        self._call(gh, task_list=task_list)
        body = gh.edit_pr_body.call_args[0][2]
        assert "Done task" not in body
        assert "Pending task" in body

    def test_fixes_line_appended_when_missing(self) -> None:
        gh = MagicMock()
        self._call(
            gh,
            print_return="<body>Summary without fixes line.</body>",
            issue=99,
            pr_number=5,
        )
        body = gh.edit_pr_body.call_args[0][2]
        assert "Fixes #99." in body

    def test_fixes_line_not_duplicated_when_present(self) -> None:
        gh = MagicMock()
        self._call(
            gh,
            print_return="<body>Summary.\n\nFixes #1.</body>",
            issue=1,
        )
        body = gh.edit_pr_body.call_args[0][2]
        assert body.count("Fixes #1.") == 1

    def test_strips_preamble_before_body_tag(self) -> None:
        """Claude's chatty preamble before <body> must be stripped from the PR."""
        gh = MagicMock()
        chatty = (
            "The current body is short. Here's the replacement:\n\n"
            "<body>Clean description.\n\nFixes #1.</body>\n\n"
            "Want me to update it directly?"
        )
        self._call(gh, print_return=chatty, issue=1)
        body = gh.edit_pr_body.call_args[0][2]
        assert "Clean description." in body
        assert "The current body is short" not in body
        assert "Want me to update it directly" not in body

    def test_raises_when_opus_returns_no_body_tags(self) -> None:
        """No body tags = garbage output; raise instead of silently falling back."""
        gh = MagicMock()
        with pytest.raises(ValueError, match="no <body> content"):
            self._call(gh, print_return="Bare text with no body tags.", issue=42)
        gh.edit_pr_body.assert_not_called()

    def test_body_tag_match_is_case_insensitive(self) -> None:
        gh = MagicMock()
        self._call(gh, print_return="<BODY>Upper case tag.</BODY>", issue=1)
        body = gh.edit_pr_body.call_args[0][2]
        assert "Upper case tag." in body

    def test_rewrite_preserves_rest_section(self) -> None:
        gh = MagicMock()
        existing = (
            "Old description.\n\nFixes #1.\n\n---\n\n"
            "## Work queue\n\n<!-- WORK_QUEUE_START -->\n"
            "- [ ] do a thing\n<!-- WORK_QUEUE_END -->"
        )
        self._call(gh, existing_body=existing)
        body = gh.edit_pr_body.call_args[0][2]
        assert "do a thing" in body
        assert "<!-- WORK_QUEUE_START -->" in body
        assert "Old description." not in body

    def test_rewrite_raises_when_no_divider(self) -> None:
        gh = MagicMock()
        with pytest.raises(ValueError, match="no --- divider"):
            self._call(gh, existing_body="no divider here")
        gh.edit_pr_body.assert_not_called()

    def test_requires_agent(self) -> None:
        gh = MagicMock()
        with pytest.raises(ValueError, match="_write_pr_description requires agent"):
            _write_pr_description(gh, "owner/repo", 99, 42, [])


class TestFindOrCreatePr:
    """Tests for Worker.find_or_create_pr."""

    def _make_worker(
        self, tmp_path: Path, provider_agent: MagicMock | None = None
    ) -> tuple["Worker", MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh, provider_agent=provider_agent), gh

    def _make_repo_ctx(
        self,
        repo: str = "owner/proj",
        owner: str = "owner",
        repo_name: str = "proj",
        gh_user: str = "fido-bot",
        default_branch: str = "main",
    ) -> "RepoContext":
        from kennel.worker import RepoContext

        return RepoContext(
            repo=repo,
            owner=owner,
            repo_name=repo_name,
            gh_user=gh_user,
            default_branch=default_branch,
            membership=RepoMembership(collaborators=frozenset({owner})),
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "fido"
        d.mkdir()
        return d

    def _open_pr(self, number: int = 10, slug: str = "fix-bug") -> dict:
        return {"number": number, "headRefName": slug, "state": "OPEN"}

    # --- Open PR (resume) path ---

    def test_open_pr_returns_pr_number_and_slug(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-branch")
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.tasks.Tasks.list", return_value=["a task"]),
        ):
            result = worker.find_or_create_pr(
                fido_dir, self._make_repo_ctx(), 5, "title"
            )
        assert result == (20, "my-branch", False)

    def test_open_pr_logs_resuming(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="fix-stuff")
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.tasks.Tasks.list", return_value=["a task"]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert "resuming" in caplog.text

    def test_open_pr_runs_setup_when_no_tasks(self, tmp_path: Path) -> None:
        mock_client = _client()
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        mock_build = MagicMock()
        mock_start = MagicMock(return_value="sess-1")
        with (
            patch.object(worker, "_git"),
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch("kennel.worker.build_prompt", mock_build),
            patch("kennel.worker.provider_start", mock_start),
            pytest.raises(RuntimeError),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        mock_build.assert_called_once_with(fido_dir, "setup", ANY)
        mock_start.assert_called_once_with(
            fido_dir,
            model=mock_client.voice_model,
            cwd=tmp_path,
            session=None,
            agent=mock_client,
            session_mode=TurnSessionMode.REUSE,
        )

    def test_open_pr_setup_context_includes_work_dir(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        mock_build = MagicMock()
        with (
            patch.object(worker, "_git"),
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch("kennel.worker.build_prompt", mock_build),
            patch("kennel.worker.provider_start", return_value="sess"),
            pytest.raises(RuntimeError),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        _, _, context = mock_build.call_args.args
        assert f"Work dir: {tmp_path}" in context

    def test_open_pr_setup_no_tasks_raises(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value="sess"),
            pytest.raises(RuntimeError, match="setup produced no tasks"),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")

    def test_open_pr_setup_persists_session_id(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 5})
        task = {"title": "t", "status": "pending"}
        call_count = 0

        def list_tasks_side_effect():
            nonlocal call_count
            call_count += 1
            return [] if call_count <= 2 else [task]

        with (
            patch.object(worker, "_git"),
            patch("kennel.tasks.Tasks.list", side_effect=list_tasks_side_effect),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value=""),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")

    def test_open_pr_seeds_from_pr_body_before_setup(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        # seed_tasks_from_pr_body populates tasks → setup not called
        call_order = []
        with (
            patch.object(worker, "_git"),
            patch(
                "kennel.tasks.Tasks.list",
                side_effect=[[], [{"id": "1", "title": "t", "status": "pending"}]],
            ),
            patch.object(
                worker,
                "seed_tasks_from_pr_body",
                side_effect=lambda *a: call_order.append("seed"),
            ),
            patch(
                "kennel.worker.build_prompt",
                side_effect=lambda *a: call_order.append("setup"),
            ),
            patch("kennel.worker.provider_start", return_value="sess"),
        ):
            result = worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")
        assert result == (20, "my-br", False)
        assert "seed" in call_order
        assert "setup" not in call_order

    def test_open_pr_setup_produces_tasks_returns_pr(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        task = {"title": "t", "status": "pending"}
        call_count = 0

        def list_tasks_side_effect():
            nonlocal call_count
            call_count += 1
            return [] if call_count == 1 else [task]

        with (
            patch.object(worker, "_git"),
            patch("kennel.tasks.Tasks.list", side_effect=list_tasks_side_effect),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value="sess"),
        ):
            result = worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")
        assert result == (20, "my-br", False)

    def test_open_pr_skips_setup_when_tasks_exist(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        mock_build = MagicMock()
        with (
            patch.object(worker, "_git"),
            patch(
                "kennel.tasks.Tasks.list",
                return_value=[{"title": "t", "status": "pending"}],
            ),
            patch("kennel.worker.build_prompt", mock_build),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        mock_build.assert_not_called()

    def test_open_pr_checkout_fallback_on_error(self, tmp_path: Path) -> None:
        """If git checkout slug fails, try checkout -b --track."""
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(slug="br")
        fido_dir = self._fido_dir(tmp_path)
        git_calls = []

        def side_effect(args, check=True):  # noqa: ARG001
            git_calls.append(args)
            if args == ["checkout", "br"]:
                raise subprocess.CalledProcessError(1, "git")
            return MagicMock()

        with (
            patch.object(worker, "_git", side_effect=side_effect),
            patch("kennel.tasks.Tasks.list", return_value=["t"]),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert ["checkout", "-b", "br", "--track", "origin/br"] in git_calls

    def test_open_pr_fetches_before_checkout(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(slug="br")
        fido_dir = self._fido_dir(tmp_path)
        git_calls = []

        def side_effect(args, check=True):  # noqa: ARG001
            git_calls.append(args)
            return MagicMock()

        with (
            patch.object(worker, "_git", side_effect=side_effect),
            patch("kennel.tasks.Tasks.list", return_value=["t"]),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert git_calls[0] == ["fetch", "origin"]

    # --- No PR (new branch) path ---

    def test_no_pr_returns_pr_number_and_slug(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_branch_name.return_value = "fix-bug"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/55"
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value="sess"),
            patch("kennel.worker._write_pr_description"),
            patch(
                "kennel.tasks.Tasks.list",
                return_value=[{"title": "Do thing", "status": "pending"}],
            ),
        ):
            result = worker.find_or_create_pr(
                fido_dir, self._make_repo_ctx(), 5, "Fix the bug"
            )
        assert result is not None
        pr_number, slug, is_fresh = result
        assert pr_number == 55
        assert slug == "fix-bug"
        assert is_fresh is True

    def test_no_pr_logs_new_branch(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_client = _client()
        mock_client.generate_branch_name.return_value = "do-work"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value=""),
            patch("kennel.worker._write_pr_description"),
            patch("kennel.tasks.Tasks.list", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
            pytest.raises(RuntimeError),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert "new branch" in caplog.text

    def test_no_pr_calls_setup(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_branch_name.return_value = "do-work"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        mock_build = MagicMock()
        mock_start = MagicMock(return_value="s")
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.build_prompt", mock_build),
            patch("kennel.worker.provider_start", mock_start),
            patch("kennel.worker._write_pr_description"),
            patch("kennel.tasks.Tasks.list", return_value=[]),
            pytest.raises(RuntimeError),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        mock_build.assert_called_once_with(fido_dir, "setup", ANY)
        mock_start.assert_called_once_with(
            fido_dir,
            model=mock_client.voice_model,
            cwd=tmp_path,
            session=None,
            agent=mock_client,
            session_mode=TurnSessionMode.REUSE,
        )

    def test_no_pr_setup_context_includes_work_dir(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_branch_name.return_value = "do-work"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        mock_build = MagicMock()
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.build_prompt", mock_build),
            patch("kennel.worker.provider_start", return_value="s"),
            patch("kennel.worker._write_pr_description"),
            patch("kennel.tasks.Tasks.list", return_value=[]),
            pytest.raises(RuntimeError),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        _, _, context = mock_build.call_args.args
        assert f"Work dir: {tmp_path}" in context

    def test_no_pr_creates_pr_with_correct_params(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_branch_name.return_value = "do-work"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/99"
        fido_dir = self._fido_dir(tmp_path)
        repo_ctx = self._make_repo_ctx(repo="owner/proj", default_branch="main")
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value=""),
            patch("kennel.worker._write_pr_description"),
            patch(
                "kennel.tasks.Tasks.list",
                return_value=[{"title": "t", "status": "pending"}],
            ),
        ):
            worker.find_or_create_pr(fido_dir, repo_ctx, 7, "Do the work")
        gh.create_pr.assert_called_once_with(
            "owner/proj",
            "Do the work (closes #7)",
            ANY,
            "main",
            "do-work",
        )

    def test_no_pr_git_operations_in_order(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_branch_name.return_value = "do-work"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        git_calls = []

        def side_effect(args, check=True):  # noqa: ARG001
            git_calls.append(list(args))
            return MagicMock()

        with (
            patch.object(worker, "_git", side_effect=side_effect),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value=""),
            patch("kennel.worker._write_pr_description"),
            patch("kennel.tasks.Tasks.list", return_value=[]),
            pytest.raises(RuntimeError),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert git_calls[0] == ["fetch", "origin"]
        assert git_calls[-2] == ["commit", "--allow-empty", "-m", "wip: start"]
        assert git_calls[-1] == ["push", "-u", "origin", "do-work"]

    def test_no_pr_deletes_existing_branch_before_creating(
        self, tmp_path: Path
    ) -> None:
        """Always start fresh — delete existing branch before checkout -b."""
        mock_client = _client()
        mock_client.generate_branch_name.return_value = "slug"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        git_calls = []

        def side_effect(args, check=True):  # noqa: ARG001
            git_calls.append(list(args))
            return MagicMock()

        with (
            patch.object(worker, "_git", side_effect=side_effect),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value=""),
            patch("kennel.worker._write_pr_description"),
            patch("kennel.tasks.Tasks.list", return_value=[]),
            pytest.raises(RuntimeError),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert ["branch", "-D", "slug"] in git_calls
        assert ["checkout", "-b", "slug", "origin/main"] in git_calls

    def test_no_pr_slug_sanitized(self, tmp_path: Path) -> None:
        mock_client = _client()
        mock_client.generate_branch_name.return_value = "Add New Feature!"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        git_calls = []

        def side_effect(args, check=True):  # noqa: ARG001
            git_calls.append(list(args))
            return MagicMock()

        with (
            patch.object(worker, "_git", side_effect=side_effect),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value=""),
            patch("kennel.worker._write_pr_description"),
            patch(
                "kennel.tasks.Tasks.list",
                return_value=[{"title": "t", "status": "pending"}],
            ),
        ):
            result = worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")
        assert result is not None
        _, slug, is_fresh = result
        assert slug == slug.lower()
        assert "!" not in slug
        assert is_fresh is True

    def test_no_pr_setup_no_tasks_raises(self, tmp_path: Path) -> None:
        """New-PR path: setup produces no tasks → raises RuntimeError, skips PR creation."""
        mock_client = _client()
        mock_client.generate_branch_name.return_value = "fix-bug"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = None
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value="sess"),
            patch("kennel.tasks.Tasks.list", return_value=[]),
            pytest.raises(RuntimeError, match="setup produced no tasks"),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")
        gh.create_pr.assert_not_called()

    def test_no_pr_logs_pr_number(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_client = _client()
        mock_client.generate_branch_name.return_value = "work"
        worker, gh = self._make_worker(tmp_path, provider_agent=mock_client)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/42"
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_start", return_value=""),
            patch("kennel.worker._write_pr_description"),
            patch(
                "kennel.tasks.Tasks.list",
                return_value=[{"title": "t", "status": "pending"}],
            ),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert "42" in caplog.text


class TestSeedTasksFromPrBody:
    """Tests for Worker.seed_tasks_from_pr_body."""

    def _make_worker(self, tmp_path: Path) -> tuple["Worker", MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def _pr_with_queue(self, *task_titles: str, task_type: str = "spec") -> dict:
        lines = "\n".join(f"- [ ] {t} <!-- type:{task_type} -->" for t in task_titles)
        body = (
            f"Description.\n\n"
            f"<!-- WORK_QUEUE_START -->\n{lines}\n<!-- WORK_QUEUE_END -->"
        )
        return {"body": body}

    def test_noop_when_tasks_exist(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        with patch("kennel.tasks.Tasks.list", return_value=[{"title": "t"}]):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        gh.get_pr.assert_not_called()

    def test_fetches_pr_body_when_no_tasks(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"body": ""}
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.seed_tasks_from_pr_body("owner/repo", 42)
        gh.get_pr.assert_called_once_with("owner/repo", 42)

    def test_noop_when_no_markers_in_body(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"body": "No markers here."}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        mock_add.assert_not_called()

    def test_noop_when_no_unchecked_tasks_in_queue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "body": (
                "<!-- WORK_QUEUE_START -->\n"
                "<!-- no tasks yet -->\n"
                "<!-- WORK_QUEUE_END -->"
            )
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        mock_add.assert_not_called()

    def test_adds_single_task_from_body(self, tmp_path: Path) -> None:
        from kennel.types import TaskType

        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_with_queue("Fix the bug")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        mock_add.assert_called_once_with("Fix the bug", TaskType.SPEC)

    def test_adds_multiple_tasks(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_with_queue(
            "Task one", "Task two", "Task three"
        )
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        assert mock_add.call_count == 3

    def test_skips_completed_tasks(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "body": (
                "<!-- WORK_QUEUE_START -->\n"
                "- [x] Already done <!-- type:spec -->\n"
                "- [x] Also done <!-- type:spec -->\n"
                "- [ ] Still pending <!-- type:spec -->\n"
                "<!-- WORK_QUEUE_END -->"
            )
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        # Only the unchecked task should have been added
        assert mock_add.call_count == 1
        assert mock_add.call_args.args[0] == "Still pending"

    def test_skips_all_when_only_completed(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "body": (
                "<!-- WORK_QUEUE_START -->\n"
                "- [x] Done one <!-- type:spec -->\n"
                "- [x] Done two <!-- type:spec -->\n"
                "<!-- WORK_QUEUE_END -->"
            )
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        mock_add.assert_not_called()

    def test_strips_next_marker(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "body": (
                "<!-- WORK_QUEUE_START -->\n"
                "- [ ] First task **→ next** <!-- type:spec -->\n"
                "- [ ] Second task <!-- type:spec -->\n"
                "<!-- WORK_QUEUE_END -->"
            )
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        titles = [call.args[0] for call in mock_add.call_args_list]
        assert titles[0] == "First task"
        assert "**→ next**" not in titles[0]

    def test_correct_task_titles_in_order(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_with_queue("Write the tests", "Fix the lint")
        received: list[str] = []
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch(
                "kennel.tasks.Tasks.add",
                side_effect=lambda t, tt, **kw: received.append(t),
            ),
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 5)
        assert received == ["Write the tests", "Fix the lint"]

    def test_noop_when_body_is_none(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"body": None}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        mock_add.assert_not_called()

    def test_skips_lines_without_type_comment(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "body": (
                "<!-- WORK_QUEUE_START -->\n"
                "- [ ] Task with type <!-- type:spec -->\n"
                "- [ ] Task without type\n"
                "<!-- WORK_QUEUE_END -->"
            )
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add") as mock_add,
            caplog.at_level(logging.WARNING, logger="kennel"),
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        assert mock_add.call_count == 1
        assert "without type marker" in caplog.text

    def test_logs_info_with_task_count(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_with_queue("T1", "T2")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        assert "seeded" in caplog.text
        assert "2" in caplog.text

    def test_does_not_log_when_no_tasks_found(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "body": "<!-- WORK_QUEUE_START -->\n<!-- WORK_QUEUE_END -->"
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        assert "seeded" not in caplog.text

    def test_skips_empty_title_after_stripping(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "body": (
                "<!-- WORK_QUEUE_START -->\n"
                "- [ ] <!-- type:spec -->\n"
                "- [ ] Real task <!-- type:spec -->\n"
                "<!-- WORK_QUEUE_END -->"
            )
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch("kennel.tasks.Tasks.add") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        assert mock_add.call_count == 1


class TestRunSeedTasksIntegration:
    """Tests for seed_tasks_from_pr_body being called from Worker.run()."""

    def _make_gh(self) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.get_default_branch.return_value = "main"
        gh.get_pr.return_value = {"body": ""}
        return gh

    def _make_mock_ctx(self, tmp_path: Path) -> MagicMock:
        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        mock_ctx.fido_dir = tmp_path / ".git" / "fido"
        return mock_ctx

    def _make_mock_repo_ctx(self) -> MagicMock:
        repo_ctx = MagicMock(spec=RepoContext)
        repo_ctx.repo = "owner/repo"
        repo_ctx.gh_user = "fido-bot"
        repo_ctx.default_branch = "main"
        repo_ctx.membership = RepoMembership()
        return repo_ctx

    def test_seed_called_after_find_or_create_pr(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "My task", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_seed = MagicMock()
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body", mock_seed),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
        ):
            worker.run()
        mock_seed.assert_called_once_with("owner/repo", 42)

    def test_seed_not_called_when_find_or_create_pr_raises(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Done", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_seed = MagicMock()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=3),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker,
                "find_or_create_pr",
                side_effect=RuntimeError("setup produced no tasks"),
            ),
            patch.object(worker, "seed_tasks_from_pr_body", mock_seed),
            pytest.raises(RuntimeError),
        ):
            worker.run()
        mock_seed.assert_not_called()


class TestExtractRunId:
    """Tests for Worker._extract_run_id."""

    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def test_extracts_id_from_standard_url(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert (
            w._extract_run_id(
                "https://github.com/owner/repo/actions/runs/123456789/job/987"
            )
            == "123456789"
        )

    def test_extracts_id_from_url_without_job(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert (
            w._extract_run_id("https://github.com/owner/repo/actions/runs/42") == "42"
        )

    def test_returns_empty_when_no_run_id(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert w._extract_run_id("https://github.com/owner/repo/actions") == ""

    def test_returns_empty_for_empty_link(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert w._extract_run_id("") == ""


class TestFilterCiThreads:
    """Tests for Worker._filter_ci_threads."""

    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def _make_node(
        self,
        *,
        resolved: bool = False,
        first_author: str = "reviewer",
        first_body: str = "CI is failing",
        last_author: str = "reviewer",
        last_body: str = "CI is failing",
        url: str = "https://example.com/comment",
    ) -> dict:
        return {
            "isResolved": resolved,
            "comments": {
                "nodes": [
                    {
                        "author": {"login": first_author},
                        "body": first_body,
                        "url": url,
                    },
                    {
                        "author": {"login": last_author},
                        "body": last_body,
                        "url": url,
                    },
                ]
            },
        }

    def test_returns_empty_when_no_threads(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert w._filter_ci_threads([], "fido-bot", "test") == []

    def test_excludes_resolved_threads(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert (
            w._filter_ci_threads(
                [self._make_node(resolved=True, first_body="ci failing")],
                "reviewer",
                "test",
            )
            == []
        )

    def test_excludes_threads_where_last_author_is_gh_user(
        self, tmp_path: Path
    ) -> None:
        w = self._make_worker(tmp_path)
        assert (
            w._filter_ci_threads(
                [self._make_node(last_author="fido-bot", last_body="ci fix")],
                "fido-bot",
                "test",
            )
            == []
        )

    def test_excludes_threads_without_ci_keywords(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert (
            w._filter_ci_threads(
                [
                    self._make_node(
                        first_body="style nit", last_body="please rename var"
                    )
                ],
                "fido-bot",
                "mycheck",
            )
            == []
        )

    def test_includes_thread_matching_check_name(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_ci_threads(
            [self._make_node(first_body="mycheck is red", last_body="please fix")],
            "fido-bot",
            "mycheck",
        )
        assert len(result) == 1

    def test_includes_thread_mentioning_ci(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_ci_threads(
            [self._make_node(first_body="CI broke after your commit")],
            "fido-bot",
            "unrelated-check",
        )
        assert len(result) == 1

    def test_includes_thread_mentioning_lint(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_ci_threads(
            [self._make_node(first_body="lint errors in this file")],
            "fido-bot",
            "check",
        )
        assert len(result) == 1

    def test_includes_thread_mentioning_format(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_ci_threads(
            [self._make_node(first_body="format issue here")], "fido-bot", "check"
        )
        assert len(result) == 1

    def test_maps_fields_correctly(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_ci_threads(
            [
                self._make_node(
                    first_author="alice",
                    first_body="CI is red",
                    last_author="bob",
                    last_body="still red",
                    url="https://github.com/x",
                )
            ],
            "fido-bot",
            "ci",
        )
        assert result[0] == {
            "first_author": "alice",
            "first_body": "CI is red",
            "last_author": "bob",
            "last_body": "still red",
            "url": "https://github.com/x",
        }

    def test_case_insensitive_keyword_match(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_ci_threads(
            [self._make_node(first_body="LINT errors found")], "fido-bot", "check"
        )
        assert len(result) == 1

    def test_single_comment_node(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = {
            "isResolved": False,
            "comments": {
                "nodes": [
                    {
                        "author": {"login": "alice"},
                        "body": "ci failed",
                        "url": "https://example.com",
                    }
                ]
            },
        }
        result = w._filter_ci_threads([node], "fido-bot", "check")
        assert len(result) == 1
        assert result[0]["first_author"] == "alice"
        assert result[0]["last_author"] == "alice"

    def test_skips_nodes_with_empty_comments(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = {"isResolved": False, "comments": {"nodes": []}}
        assert w._filter_ci_threads([node], "fido-bot", "check") == []


class TestHandleMergeConflict:
    """Tests for Worker.handle_merge_conflict."""

    def _make_worker(self, tmp_path: Path) -> tuple[Worker, MagicMock]:
        gh = MagicMock()
        gh.get_pr.return_value = {"mergeStateStatus": "DIRTY"}
        return Worker(tmp_path, gh), gh

    def _repo_ctx(self) -> RepoContext:
        return RepoContext(
            repo="owner/repo",
            owner="owner",
            repo_name="repo",
            gh_user="fido-bot",
            default_branch="main",
            membership=RepoMembership(collaborators=frozenset({"owner"})),
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_returns_false_when_not_dirty(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        fido_dir = self._fido_dir(tmp_path)
        assert (
            worker.handle_merge_conflict(fido_dir, self._repo_ctx(), 1, "branch")
            is False
        )

    def test_returns_false_when_blocked(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        fido_dir = self._fido_dir(tmp_path)
        assert (
            worker.handle_merge_conflict(fido_dir, self._repo_ctx(), 1, "branch")
            is False
        )

    def test_returns_false_when_missing_merge_state(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"body": ""}
        fido_dir = self._fido_dir(tmp_path)
        assert (
            worker.handle_merge_conflict(fido_dir, self._repo_ctx(), 1, "branch")
            is False
        )

    def test_returns_true_when_dirty(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sid", "")),
        ):
            result = worker.handle_merge_conflict(
                fido_dir, self._repo_ctx(), 1, "branch"
            )
        assert result is True

    def test_calls_set_status_with_pr_number(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status") as mock_status,
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
        ):
            worker.handle_merge_conflict(fido_dir, self._repo_ctx(), 42, "my-branch")
        mock_status.assert_called_once_with("Resolving merge conflicts on PR #42")

    def test_builds_merge_prompt(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.provider_run", return_value=("", "")),
        ):
            worker.handle_merge_conflict(fido_dir, self._repo_ctx(), 5, "fix-branch")
        mock_bp.assert_called_once()
        _, subskill, context = mock_bp.call_args[0]
        assert subskill == "merge"
        assert "fix-branch" in context
        assert "PR: 5" in context
        assert "origin/main" in context

    def test_runs_claude(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sess-1", "")) as mock_cr,
        ):
            worker.handle_merge_conflict(fido_dir, self._repo_ctx(), 1, "branch")
        mock_cr.assert_called_once_with(
            fido_dir,
            model=ClaudeClient.work_model,
            cwd=tmp_path,
            session=None,
            agent=ANY,
            session_mode=TurnSessionMode.REUSE,
        )

    def test_does_not_call_claude_when_not_dirty(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        fido_dir = self._fido_dir(tmp_path)
        with patch("kennel.worker.provider_run") as mock_cr:
            worker.handle_merge_conflict(fido_dir, self._repo_ctx(), 1, "branch")
        mock_cr.assert_not_called()

    def test_checks_pr_merge_state(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
        ):
            worker.handle_merge_conflict(fido_dir, self._repo_ctx(), 7, "branch")
        gh.get_pr.assert_called_once_with("owner/repo", 7)


class TestRunHandleMergeConflictIntegration:
    """Tests that Worker.run() calls handle_merge_conflict before handle_ci."""

    def _make_gh(self) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.get_default_branch.return_value = "main"
        gh.get_pr.return_value = {"body": ""}
        return gh

    def _make_mock_ctx(self, tmp_path: Path) -> MagicMock:
        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        mock_ctx.fido_dir = tmp_path / ".git" / "fido"
        return mock_ctx

    def _make_mock_repo_ctx(self) -> MagicMock:
        repo_ctx = MagicMock(spec=RepoContext)
        repo_ctx.repo = "owner/repo"
        repo_ctx.gh_user = "fido-bot"
        repo_ctx.default_branch = "main"
        repo_ctx.membership = RepoMembership()
        return repo_ctx

    def test_handle_merge_conflict_called_with_pr_and_slug(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_handle_mc = MagicMock(return_value=False)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_merge_conflict", mock_handle_mc),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
        ):
            worker.run()
        mock_handle_mc.assert_called_once_with(
            mock_ctx.fido_dir, repo_ctx, 42, "fix-bug"
        )

    def test_returns_1_when_merge_conflict_handled(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_merge_conflict", return_value=True),
        ):
            result = worker.run()
        assert result == 1

    def test_handle_ci_not_called_when_merge_conflict_handled(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        mock_ci = MagicMock(return_value=False)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_merge_conflict", return_value=True),
            patch.object(worker, "handle_ci", mock_ci),
        ):
            worker.run()
        mock_ci.assert_not_called()

    def test_handle_merge_conflict_not_called_on_fresh_pr(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        mock_mc = MagicMock(return_value=False)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", True)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_merge_conflict", mock_mc),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=0),
        ):
            worker.run()
        mock_mc.assert_not_called()


class TestHandleCi:
    """Tests for Worker.handle_ci."""

    def _make_worker(self, tmp_path: Path) -> tuple[Worker, MagicMock]:
        gh = MagicMock()
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        return Worker(tmp_path, gh), gh

    def _repo_ctx(self) -> RepoContext:
        return RepoContext(
            repo="owner/repo",
            owner="owner",
            repo_name="repo",
            gh_user="fido-bot",
            default_branch="main",
            membership=RepoMembership(collaborators=frozenset({"owner"})),
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        (tmp_path / "sub").mkdir(exist_ok=True)
        return d

    def test_returns_false_when_no_checks(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        assert worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch") is False

    def test_returns_false_when_all_checks_pass(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "SUCCESS", "link": ""},
        ]
        fido_dir = self._fido_dir(tmp_path)
        assert worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch") is False

    def test_returns_true_on_failure(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "FAILURE", "link": "runs/99"},
        ]
        gh.get_run_log.return_value = "line1\nline2"
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sid", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            result = worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is True

    def test_returns_true_on_error_state(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "lint", "state": "ERROR", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sid", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            result = worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is True

    def test_calls_set_status_with_check_name_and_pr(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "unit-tests", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status") as mock_status,
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 7, "branch")
        mock_status.assert_called_once_with("Fixing CI: unit-tests on PR #7")

    def test_fetches_run_log_when_run_id_present(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {
                "name": "build",
                "state": "FAILURE",
                "link": "https://github.com/owner/repo/actions/runs/55555/job/1",
            }
        ]
        gh.get_run_log.return_value = "some log output"
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        gh.get_run_log.assert_called_once_with("owner/repo", "55555")

    def test_skips_run_log_fetch_when_no_run_id(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "build", "state": "FAILURE", "link": "no-run-id-here"},
        ]
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        gh.get_run_log.assert_not_called()

    def test_truncates_log_to_last_200_lines(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "FAILURE", "link": "runs/1"},
        ]
        long_log = "\n".join(f"line {i}" for i in range(300))
        gh.get_run_log.return_value = long_log
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        captured_context = {}
        with (
            patch.object(worker, "set_status"),
            patch(
                "kennel.worker.build_prompt",
                side_effect=lambda fd, sk, ctx: captured_context.update({"ctx": ctx}),
            ),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        log_section = captured_context["ctx"].split("Failure log")[1]
        assert "line 99\n" not in log_section  # line 99 is before the last 200
        assert "line 100\n" in log_section  # line 100 starts the last 200

    def test_fetches_review_threads_for_repo(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 42, "branch")
        gh.get_review_threads.assert_called_once_with("owner", "repo", 42)

    def test_builds_ci_prompt(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "my-check", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 5, "fix-branch")
        mock_bp.assert_called_once()
        _, subskill, context = mock_bp.call_args[0]
        assert subskill == "ci"
        assert "my-check" in context
        assert "fix-branch" in context
        assert "PR: 5" in context

    def test_runs_claude(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sess-1", "")) as mock_cr,
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        mock_cr.assert_called_once_with(
            fido_dir,
            model=ClaudeClient.work_model,
            cwd=tmp_path,
            session=None,
            agent=ANY,
            session_mode=TurnSessionMode.REUSE,
        )

    def test_does_not_complete_ci_task(self, tmp_path: Path) -> None:
        """CI failures have no task entry — no complete call needed."""
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "my-check", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        mock_complete.assert_not_called()

    def test_spawns_sync_script(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks") as mock_sync,
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        mock_sync.assert_called_once_with(tmp_path, gh)

    def test_picks_first_failing_check(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "pass-check", "state": "SUCCESS", "link": ""},
            {"name": "fail-check", "state": "FAILURE", "link": ""},
            {"name": "err-check", "state": "ERROR", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status") as mock_status,
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        # First failing check is "fail-check"
        mock_status.assert_called_once_with("Fixing CI: fail-check on PR #1")

    def test_logs_ci_failing(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "flaky", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        assert "flaky" in caplog.text

    def test_returns_false_when_merge_state_not_blocked(self, tmp_path: Path) -> None:
        """Non-required check failures (UNSTABLE) must not trigger CI fixing."""
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"mergeStateStatus": "UNSTABLE"}
        fido_dir = self._fido_dir(tmp_path)
        assert worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch") is False
        gh.pr_checks.assert_not_called()

    def test_returns_false_when_merge_state_clean(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        fido_dir = self._fido_dir(tmp_path)
        assert worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch") is False
        gh.pr_checks.assert_not_called()

    def test_proceeds_when_merge_state_blocked(self, tmp_path: Path) -> None:
        """BLOCKED merge state should still trigger CI fixing on failure."""
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        gh.pr_checks.return_value = [
            {"name": "required-check", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sid", "")),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            result = worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is True

    def test_proceeds_when_merge_state_dirty(self, tmp_path: Path) -> None:
        """DIRTY merge state (merge conflicts) should still check CI."""
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"mergeStateStatus": "DIRTY"}
        gh.pr_checks.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        assert worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch") is False
        gh.pr_checks.assert_called_once()


class TestRunHandleCiIntegration:
    """Tests that Worker.run() calls handle_ci and respects its return value."""

    def _make_gh(self) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.get_default_branch.return_value = "main"
        gh.get_pr.return_value = {"body": ""}
        return gh

    def _make_mock_ctx(self, tmp_path: Path) -> MagicMock:
        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        mock_ctx.fido_dir = tmp_path / ".git" / "fido"
        return mock_ctx

    def _make_mock_repo_ctx(self) -> MagicMock:
        repo_ctx = MagicMock(spec=RepoContext)
        repo_ctx.repo = "owner/repo"
        repo_ctx.gh_user = "fido-bot"
        repo_ctx.default_branch = "main"
        repo_ctx.membership = RepoMembership()
        return repo_ctx

    def test_handle_ci_called_with_pr_and_slug(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_handle_ci = MagicMock(return_value=False)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", mock_handle_ci),
            patch.object(worker, "handle_threads", return_value=False),
        ):
            worker.run()
        mock_handle_ci.assert_called_once_with(
            mock_ctx.fido_dir, repo_ctx, 42, "fix-bug"
        )

    def test_returns_1_when_ci_handled(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=True),
        ):
            result = worker.run()
        assert result == 1

    def test_returns_0_when_ci_passes(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
        ):
            result = worker.run()
        assert result == 0

    def test_handle_ci_not_called_when_find_or_create_pr_raises(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Done", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_handle_ci = MagicMock(return_value=False)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=3),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker,
                "find_or_create_pr",
                side_effect=RuntimeError("setup produced no tasks"),
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", mock_handle_ci),
            pytest.raises(RuntimeError),
        ):
            worker.run()
        mock_handle_ci.assert_not_called()


class TestFilterThreads:
    """Tests for Worker._filter_threads."""

    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def _make_node(
        self,
        *,
        resolved: bool = False,
        node_id: str = "thread-1",
        first_author: str = "owner",
        first_db_id: int = 1,
        first_body: str = "please fix this",
        last_author: str | None = None,
        last_body: str = "please fix this",
        url: str = "https://example.com/comment",
        extra_comments: list | None = None,
    ) -> dict:
        """Build a review-thread node.

        If *last_author* is given (and differs from first_author or last_body
        differs from first_body) a second comment is appended.  *extra_comments*
        are inserted between first and last.
        """
        first = {
            "author": {"login": first_author},
            "body": first_body,
            "url": url,
            "databaseId": first_db_id,
        }
        comments: list = [first]
        if extra_comments:
            comments.extend(extra_comments)
        if last_author is not None and (
            last_author != first_author or last_body != first_body
        ):
            comments.append(
                {
                    "author": {"login": last_author},
                    "body": last_body,
                    "url": url,
                    "databaseId": first_db_id + len(comments),
                }
            )
        return {
            "id": node_id,
            "isResolved": resolved,
            "comments": {"nodes": comments},
        }

    def test_returns_empty_when_no_nodes(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert w._filter_threads([], "fido-bot", frozenset({"owner"})) == []

    def test_excludes_resolved_threads(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert (
            w._filter_threads(
                [self._make_node(resolved=True)], "fido-bot", frozenset({"owner"})
            )
            == []
        )

    def test_excludes_threads_where_last_author_is_gh_user(
        self, tmp_path: Path
    ) -> None:
        w = self._make_worker(tmp_path)
        assert (
            w._filter_threads(
                [self._make_node(last_author="fido-bot", last_body="done")],
                "fido-bot",
                frozenset({"owner"}),
            )
            == []
        )

    def test_excludes_threads_where_last_author_is_neither_owner_nor_bot(
        self, tmp_path: Path
    ) -> None:
        w = self._make_worker(tmp_path)
        assert (
            w._filter_threads(
                [self._make_node(last_author="random-user", last_body="comment")],
                "fido-bot",
                frozenset({"owner"}),
            )
            == []
        )

    def test_includes_thread_from_owner(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_threads(
            [self._make_node(last_author="owner")], "fido-bot", frozenset({"owner"})
        )
        assert len(result) == 1

    def test_includes_thread_from_bot(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_threads(
            [self._make_node(last_author="my-app[bot]", last_body="bot comment")],
            "fido-bot",
            frozenset({"owner"}),
        )
        assert len(result) == 1

    def test_includes_thread_from_any_collaborator(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_threads(
            [self._make_node(last_author="bob")],
            "fido-bot",
            frozenset({"alice", "bob", "carol"}),
        )
        assert len(result) == 1

    def test_maps_fields_correctly(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_threads(
            [
                self._make_node(
                    node_id="tid-42",
                    first_author="owner",
                    first_db_id=99,
                    first_body="first comment",
                    last_author="owner",
                    last_body="last comment",
                    url="https://github.com/x",
                )
            ],
            "fido-bot",
            frozenset({"owner"}),
        )
        assert result[0]["id"] == "tid-42"
        assert result[0]["first_author"] == "owner"
        assert result[0]["first_db_id"] == 99
        assert result[0]["first_body"] == "first comment"
        assert result[0]["last_author"] == "owner"
        assert result[0]["last_body"] == "last comment"
        assert result[0]["url"] == "https://github.com/x"
        assert result[0]["total"] == 2

    def test_is_bot_true_when_first_author_ends_with_bot(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_threads(
            [self._make_node(first_author="my-app[bot]", last_author="owner")],
            "fido-bot",
            frozenset({"owner"}),
        )
        assert result[0]["is_bot"] is True

    def test_is_bot_false_when_first_author_is_human(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_threads(
            [self._make_node(first_author="owner", last_author="owner")],
            "fido-bot",
            frozenset({"owner"}),
        )
        assert result[0]["is_bot"] is False

    def test_excludes_threads_with_no_comments(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = {"id": "x", "isResolved": False, "comments": {"nodes": []}}
        assert w._filter_threads([node], "fido-bot", frozenset({"owner"})) == []

    def test_total_counts_all_comments(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        result = w._filter_threads(
            [
                self._make_node(
                    first_author="owner",
                    last_author="owner",
                    last_body="final comment",
                    extra_comments=[
                        {
                            "author": {"login": "owner"},
                            "body": "mid",
                            "url": "u",
                            "databaseId": 2,
                        }
                    ],
                )
            ],
            "fido-bot",
            frozenset({"owner"}),
        )
        assert result[0]["total"] == 3


class TestResolveAddressedThreads:
    """Tests for Worker.resolve_addressed_threads."""

    def _make_worker(self, tmp_path: Path) -> tuple[Worker, MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def _repo_ctx(self) -> RepoContext:
        return RepoContext(
            repo="owner/repo",
            owner="owner",
            repo_name="repo",
            gh_user="fido-bot",
            default_branch="main",
            membership=RepoMembership(collaborators=frozenset({"owner"})),
        )

    def _make_node(
        self,
        *,
        resolved: bool = False,
        node_id: str = "thread-1",
        last_author: str = "fido-bot",
    ) -> dict:
        return {
            "id": node_id,
            "isResolved": resolved,
            "comments": {
                "nodes": [
                    {
                        "author": {"login": "owner"},
                        "body": "please fix",
                        "url": "https://example.com",
                        "databaseId": 1,
                    },
                    {
                        "author": {"login": last_author},
                        "body": "done",
                        "url": "https://example.com",
                        "databaseId": 2,
                    },
                ]
            },
        }

    def test_returns_false_when_no_threads(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = []
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is False
        gh.resolve_thread.assert_not_called()

    def test_returns_false_when_all_threads_resolved(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = [self._make_node(resolved=True)]
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is False
        gh.resolve_thread.assert_not_called()

    def test_returns_false_when_last_author_is_not_gh_user(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = [self._make_node(last_author="owner")]
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is False
        gh.resolve_thread.assert_not_called()

    def test_returns_false_when_thread_has_no_comments(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = [
            {"id": "t1", "isResolved": False, "comments": {"nodes": []}}
        ]
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is False
        gh.resolve_thread.assert_not_called()

    def test_resolves_thread_where_gh_user_is_last_author(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = [
            self._make_node(node_id="tid-99", last_author="fido-bot")
        ]
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is True
        gh.resolve_thread.assert_called_once_with("tid-99")

    def test_resolves_multiple_threads(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = [
            self._make_node(node_id="t1", last_author="fido-bot"),
            self._make_node(node_id="t2", last_author="fido-bot"),
        ]
        result = worker.resolve_addressed_threads(self._repo_ctx(), 5)
        assert result is True
        assert gh.resolve_thread.call_count == 2
        gh.resolve_thread.assert_any_call("t1")
        gh.resolve_thread.assert_any_call("t2")

    def test_skips_already_resolved_among_mixed(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = [
            self._make_node(node_id="t1", last_author="fido-bot", resolved=True),
            self._make_node(node_id="t2", last_author="fido-bot"),
        ]
        worker.resolve_addressed_threads(self._repo_ctx(), 1)
        gh.resolve_thread.assert_called_once_with("t2")

    def test_calls_get_review_threads_with_correct_args(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = []
        worker.resolve_addressed_threads(self._repo_ctx(), 42)
        gh.get_review_threads.assert_called_once_with("owner", "repo", 42)

    def test_skips_resolve_when_pending_sibling_tasks_remain(
        self, tmp_path: Path
    ) -> None:
        from kennel import tasks as tasks_mod

        worker, gh = self._make_worker(tmp_path)
        # node whose originating comment has databaseId=55
        node = {
            "id": "tid-skip",
            "isResolved": False,
            "comments": {
                "nodes": [
                    {"author": {"login": "owner"}, "databaseId": 55},
                    {"author": {"login": "fido-bot"}, "databaseId": 56},
                ]
            },
        }
        gh.get_review_threads.return_value = [node]
        from kennel.types import TaskType

        tasks_mod.add_task(
            tmp_path,
            title="pending sibling",
            task_type=TaskType.THREAD,
            thread={"repo": "owner/repo", "pr": 1, "comment_id": 55},
        )
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is False
        gh.resolve_thread.assert_not_called()

    def test_skips_resolve_when_pending_task_references_reply_comment(
        self, tmp_path: Path
    ) -> None:
        """Task comment_id matches a non-root comment — must still block resolution."""
        from kennel import tasks as tasks_mod
        from kennel.types import TaskType

        worker, gh = self._make_worker(tmp_path)
        # Thread with root comment databaseId=10, reply databaseId=11
        node = {
            "id": "tid-reply",
            "isResolved": False,
            "comments": {
                "nodes": [
                    {"author": {"login": "owner"}, "databaseId": 10},
                    {"author": {"login": "owner"}, "databaseId": 11},
                    {"author": {"login": "fido-bot"}, "databaseId": 12},
                ]
            },
        }
        gh.get_review_threads.return_value = [node]
        # Task references reply comment (id=11), not the root (id=10)
        tasks_mod.add_task(
            tmp_path,
            title="pending reply task",
            task_type=TaskType.THREAD,
            thread={"repo": "owner/repo", "pr": 1, "comment_id": 11},
        )
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is False
        gh.resolve_thread.assert_not_called()

    def test_resolves_when_all_sibling_tasks_complete(self, tmp_path: Path) -> None:
        from kennel import tasks as tasks_mod

        worker, gh = self._make_worker(tmp_path)
        node = {
            "id": "tid-resolve",
            "isResolved": False,
            "comments": {
                "nodes": [
                    {"author": {"login": "owner"}, "databaseId": 77},
                    {"author": {"login": "fido-bot"}, "databaseId": 78},
                ]
            },
        }
        gh.get_review_threads.return_value = [node]
        from kennel.types import TaskType

        task = tasks_mod.add_task(
            tmp_path,
            title="completed sibling",
            task_type=TaskType.THREAD,
            thread={"repo": "owner/repo", "pr": 1, "comment_id": 77},
        )
        tasks_mod.complete_by_id(tmp_path, task["id"])
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is True
        gh.resolve_thread.assert_called_once_with("tid-resolve")


class TestHandleThreads:
    """Tests for Worker.handle_threads."""

    def _make_worker(self, tmp_path: Path) -> tuple[Worker, MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def _repo_ctx(self) -> RepoContext:
        return RepoContext(
            repo="owner/repo",
            owner="owner",
            repo_name="repo",
            gh_user="fido-bot",
            default_branch="main",
            membership=RepoMembership(collaborators=frozenset({"owner"})),
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _open_thread_node(
        self, *, last_author: str = "owner", first_author: str = "owner"
    ) -> dict:
        return {
            "id": "thread-1",
            "isResolved": False,
            "comments": {
                "nodes": [
                    {
                        "author": {"login": first_author},
                        "body": "please fix",
                        "url": "https://example.com",
                        "databaseId": 1,
                    },
                    {
                        "author": {"login": last_author},
                        "body": "still open",
                        "url": "https://example.com",
                        "databaseId": 2,
                    },
                ]
            },
        }

    def test_returns_false_when_no_threads(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        assert worker.handle_threads(fido_dir, self._repo_ctx(), 1, "branch") is False

    def test_returns_true_when_threads_exist(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._open_thread_node()
        gh.get_review_threads.return_value = [node]
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sid", "")),
            patch("kennel.tasks.sync_tasks_background"),
        ):
            result = worker.handle_threads(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is True

    def test_calls_get_review_threads_with_correct_args(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        worker.handle_threads(fido_dir, self._repo_ctx(), 42, "branch")
        gh.get_review_threads.assert_called_once_with("owner", "repo", 42)

    def test_builds_comments_prompt(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._open_thread_node()
        gh.get_review_threads.return_value = [node]
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.provider_run", return_value=("sid", "")),
            patch("kennel.tasks.sync_tasks_background"),
        ):
            worker.handle_threads(fido_dir, self._repo_ctx(), 5, "my-branch")
        mock_bp.assert_called_once()
        _, subskill, context = mock_bp.call_args[0]
        assert subskill == "comments"
        assert "PR: 5" in context
        assert "my-branch" in context

    def test_runs_claude(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._open_thread_node()
        gh.get_review_threads.return_value = [node]
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sess-1", "")) as mock_cr,
            patch("kennel.tasks.sync_tasks_background"),
        ):
            worker.handle_threads(fido_dir, self._repo_ctx(), 1, "branch")
        mock_cr.assert_called_once_with(
            fido_dir,
            model=ClaudeClient.work_model,
            cwd=tmp_path,
            session=None,
            agent=ANY,
            session_mode=TurnSessionMode.REUSE,
        )

    def test_spawns_sync_script(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._open_thread_node()
        gh.get_review_threads.return_value = [node]
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.sync_tasks_background") as mock_sync,
        ):
            worker.handle_threads(fido_dir, self._repo_ctx(), 1, "branch")
        mock_sync.assert_called_once_with(tmp_path, gh)

    def test_logs_thread_count(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        node = self._open_thread_node()
        gh.get_review_threads.return_value = [node]
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch("kennel.tasks.sync_tasks_background"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_threads(fido_dir, self._repo_ctx(), 1, "branch")
        assert "unresolved threads" in caplog.text


class TestRunThreadsIntegration:
    """Tests that Worker.run() calls handle_threads after handle_ci."""

    def _make_gh(self) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.get_default_branch.return_value = "main"
        gh.get_pr.return_value = {"body": ""}
        return gh

    def _make_mock_ctx(self, tmp_path: Path) -> MagicMock:
        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        mock_ctx.fido_dir = tmp_path / ".git" / "fido"
        return mock_ctx

    def _make_mock_repo_ctx(self) -> MagicMock:
        repo_ctx = MagicMock(spec=RepoContext)
        repo_ctx.repo = "owner/repo"
        repo_ctx.gh_user = "fido-bot"
        repo_ctx.default_branch = "main"
        repo_ctx.membership = RepoMembership()
        return repo_ctx

    def test_handle_threads_called_when_review_feedback_passes(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_threads = MagicMock(return_value=False)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", mock_threads),
        ):
            worker.run()
        mock_threads.assert_called_once_with(mock_ctx.fido_dir, repo_ctx, 42, "fix-bug")

    def test_returns_1_when_threads_handled(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=True),
        ):
            result = worker.run()
        assert result == 1

    def test_returns_0_when_no_work(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
        ):
            result = worker.run()
        assert result == 0


class TestPickNextTask:
    """Tests for the module-level _pick_next_task helper."""

    def _task(
        self,
        title: str,
        status: str = "pending",
        task_type: str = "spec",
        thread: dict | None = None,
    ) -> dict:
        t: dict = {"id": "x", "title": title, "status": status, "type": task_type}
        if thread is not None:
            t["thread"] = thread
        return t

    def test_returns_none_for_empty_list(self) -> None:
        assert _pick_next_task([]) is None

    def test_returns_none_when_all_completed(self) -> None:
        tasks = [self._task("Fix it", status="completed")]
        assert _pick_next_task(tasks) is None

    def test_skips_ask_prefix_lowercase(self) -> None:
        tasks = [self._task("ask: clarify this")]
        assert _pick_next_task(tasks) is None

    def test_skips_ask_prefix_uppercase(self) -> None:
        tasks = [self._task("ASK: clarify this")]
        assert _pick_next_task(tasks) is None

    def test_skips_ask_prefix_mixed_case(self) -> None:
        tasks = [self._task("Ask: clarify this")]
        assert _pick_next_task(tasks) is None

    def test_skips_defer_prefix_lowercase(self) -> None:
        tasks = [self._task("defer: until later")]
        assert _pick_next_task(tasks) is None

    def test_skips_defer_prefix_uppercase(self) -> None:
        tasks = [self._task("DEFER: until later")]
        assert _pick_next_task(tasks) is None

    def test_returns_none_when_only_ask_and_defer(self) -> None:
        tasks = [
            self._task("ask: need info"),
            self._task("defer: not now"),
        ]
        assert _pick_next_task(tasks) is None

    def test_returns_regular_task(self) -> None:
        t = self._task("Implement feature X")
        assert _pick_next_task([t]) is t

    def test_ci_type_takes_priority_over_regular(self) -> None:
        regular = self._task("Implement feature X", task_type="spec")
        ci = self._task("Fix lint", task_type="ci")
        assert _pick_next_task([regular, ci]) is ci

    def test_ci_type_takes_priority_over_thread(self) -> None:
        thread_task = self._task("PR comment: fix nit", task_type="thread")
        ci = self._task("Fix tests", task_type="ci")
        assert _pick_next_task([thread_task, ci]) is ci

    def test_thread_type_uses_list_order_same_as_spec(self) -> None:
        regular = self._task("Regular work", task_type="spec")
        thread_task = self._task("PR comment: rename var", task_type="thread")
        # Thread tasks no longer jump the queue — first in list wins.
        assert _pick_next_task([regular, thread_task]) is regular
        assert _pick_next_task([thread_task, regular]) is thread_task

    def test_returns_first_pending_when_no_special(self) -> None:
        first = self._task("First task")
        second = self._task("Second task")
        assert _pick_next_task([first, second]) is first

    def test_ignores_completed_when_selecting(self) -> None:
        completed = self._task("Done task", status="completed")
        pending = self._task("Pending task")
        assert _pick_next_task([completed, pending]) is pending

    def test_task_with_spec_type_not_prioritised(self) -> None:
        """A task with spec type is not treated as thread-originated."""
        spec = self._task("Regular task", task_type="spec")
        assert spec.get("type") == "spec"
        result = _pick_next_task([spec])
        assert result is spec  # returned via fallthrough, not thread path


class TestRescopeBeforePick:
    """Tests for Worker.rescope_before_pick."""

    def _pending(self, title: str = "do something") -> dict:
        return {"id": "t1", "title": title, "status": "pending", "type": "spec"}

    def _completed(self, title: str = "done") -> dict:
        return {"id": "t2", "title": title, "status": "completed", "type": "spec"}

    def _make_worker(self, tmp_path: Path, *, with_config: bool = True) -> Worker:
        from kennel.config import Config, RepoConfig

        cfg = RepoConfig(
            name="owner/repo", work_dir=tmp_path, provider=ProviderID.CLAUDE_CODE
        )
        config = Config(
            port=9000,
            secret=b"s",
            repos={"owner/repo": cfg},
            allowed_bots=frozenset(),
            log_level="DEBUG",
            sub_dir=tmp_path,
        )
        if with_config:
            return Worker(tmp_path, MagicMock(), config=config, repo_cfg=cfg)
        return Worker(tmp_path, MagicMock())

    def test_skips_when_config_is_none(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock())
        mock_tasks = MagicMock()
        mock_tasks.list.return_value = [self._pending(), self._pending("task2")]
        worker._tasks = mock_tasks
        with patch("kennel.tasks.reorder_tasks") as mock_reorder:
            worker.rescope_before_pick()
        mock_reorder.assert_not_called()

    def test_skips_when_repo_cfg_is_none(self, tmp_path: Path) -> None:
        from kennel.config import Config

        config = Config(
            port=9000,
            secret=b"s",
            repos={},
            allowed_bots=frozenset(),
            log_level="DEBUG",
            sub_dir=tmp_path,
        )
        worker = Worker(tmp_path, MagicMock(), config=config, repo_cfg=None)
        mock_tasks = MagicMock()
        mock_tasks.list.return_value = [self._pending(), self._pending("task2")]
        worker._tasks = mock_tasks
        with patch("kennel.tasks.reorder_tasks") as mock_reorder:
            worker.rescope_before_pick()
        mock_reorder.assert_not_called()

    def test_skips_when_no_pending_tasks(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        mock_tasks = MagicMock()
        mock_tasks.list.return_value = []
        worker._tasks = mock_tasks
        with patch("kennel.tasks.reorder_tasks") as mock_reorder:
            worker.rescope_before_pick()
        mock_reorder.assert_not_called()

    def test_skips_when_exactly_one_pending_task(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        mock_tasks = MagicMock()
        mock_tasks.list.return_value = [self._pending()]
        worker._tasks = mock_tasks
        with patch("kennel.tasks.reorder_tasks") as mock_reorder:
            worker.rescope_before_pick()
        mock_reorder.assert_not_called()

    def test_skips_completed_tasks_in_pending_count(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        mock_tasks = MagicMock()
        # one pending, one completed — should skip
        mock_tasks.list.return_value = [self._pending(), self._completed()]
        worker._tasks = mock_tasks
        with patch("kennel.tasks.reorder_tasks") as mock_reorder:
            worker.rescope_before_pick()
        mock_reorder.assert_not_called()

    def test_calls_reorder_when_two_pending_tasks(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        mock_tasks = MagicMock()
        mock_tasks.list.return_value = [self._pending(), self._pending("task2")]
        worker._tasks = mock_tasks
        with (
            patch("kennel.tasks.reorder_tasks") as mock_reorder,
            patch("kennel.events._get_commit_summary", return_value="abc def"),
            patch("kennel.events._make_reorder_kwargs", return_value={}),
        ):
            worker.rescope_before_pick()
        mock_reorder.assert_called_once()

    def test_passes_work_dir_to_reorder(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        mock_tasks = MagicMock()
        mock_tasks.list.return_value = [self._pending(), self._pending("task2")]
        worker._tasks = mock_tasks
        with (
            patch("kennel.tasks.reorder_tasks") as mock_reorder,
            patch("kennel.events._get_commit_summary", return_value=""),
            patch("kennel.events._make_reorder_kwargs", return_value={}),
        ):
            worker.rescope_before_pick()
        assert mock_reorder.call_args[0][0] == tmp_path

    def test_passes_commit_summary_to_reorder(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        mock_tasks = MagicMock()
        mock_tasks.list.return_value = [self._pending(), self._pending("task2")]
        worker._tasks = mock_tasks
        with (
            patch("kennel.tasks.reorder_tasks") as mock_reorder,
            patch(
                "kennel.events._get_commit_summary", return_value="abc123 first commit"
            ),
            patch("kennel.events._make_reorder_kwargs", return_value={}),
        ):
            worker.rescope_before_pick()
        assert mock_reorder.call_args[0][1] == "abc123 first commit"

    def test_no_inprogress_affected_callback(self, tmp_path: Path) -> None:
        """Registry is passed as None so _on_inprogress_affected is not registered."""
        worker = self._make_worker(tmp_path)
        mock_tasks = MagicMock()
        mock_tasks.list.return_value = [self._pending(), self._pending("task2")]
        worker._tasks = mock_tasks
        captured_registry: list = []
        with (
            patch("kennel.tasks.reorder_tasks"),
            patch("kennel.events._get_commit_summary", return_value=""),
            patch(
                "kennel.events._make_reorder_kwargs",
                side_effect=lambda wd, cfg, repo_cfg, reg, *a, **kw: (
                    captured_registry.append(reg) or {}
                ),
            ),
        ):
            worker.rescope_before_pick()
        assert captured_registry == [None]

    def test_calls_reorder_with_three_or_more_pending_tasks(
        self, tmp_path: Path
    ) -> None:
        worker = self._make_worker(tmp_path)
        mock_tasks = MagicMock()
        mock_tasks.list.return_value = [
            self._pending("a"),
            self._pending("b"),
            self._pending("c"),
        ]
        worker._tasks = mock_tasks
        with (
            patch("kennel.tasks.reorder_tasks") as mock_reorder,
            patch("kennel.events._get_commit_summary", return_value=""),
            patch("kennel.events._make_reorder_kwargs", return_value={}),
        ):
            worker.rescope_before_pick()
        mock_reorder.assert_called_once()


class TestRunRescopeIntegration:
    """Tests that Worker.run() calls rescope_before_pick before task selection."""

    def _make_gh(self) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.get_default_branch.return_value = "main"
        gh.get_pr.return_value = {"body": ""}
        return gh

    def _make_mock_ctx(self, tmp_path: Path) -> MagicMock:
        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        mock_ctx.fido_dir = tmp_path / ".git" / "fido"
        return mock_ctx

    def _make_mock_repo_ctx(self) -> MagicMock:
        repo_ctx = MagicMock(spec=RepoContext)
        repo_ctx.repo = "owner/repo"
        repo_ctx.gh_user = "fido-bot"
        repo_ctx.default_branch = "main"
        repo_ctx.membership = RepoMembership()
        return repo_ctx

    def test_rescope_before_pick_called_from_run(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        mock_rescope = MagicMock()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-it", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "rescope_before_pick", mock_rescope),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
        ):
            worker.run()
        mock_rescope.assert_called_once_with()

    def test_rescope_called_before_handle_ci(self, tmp_path: Path) -> None:
        """rescope_before_pick must execute before handle_ci sees the task list."""
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        call_order: list[str] = []

        def record_rescope() -> None:
            call_order.append("rescope")

        def record_ci(*_a: object, **_kw: object) -> bool:
            call_order.append("handle_ci")
            return False

        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-it", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "rescope_before_pick", record_rescope),
            patch.object(worker, "handle_ci", record_ci),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
        ):
            worker.run()
        assert call_order == ["rescope", "handle_ci"]


class TestEnsurePushed:
    """Tests for Worker.ensure_pushed."""

    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def _git_result(
        self, returncode: int = 0, stdout: str = "", stderr: str = ""
    ) -> MagicMock:
        r = MagicMock()
        r.returncode = returncode
        r.stdout = stdout
        r.stderr = stderr
        return r

    def test_returns_none_when_already_in_sync(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        sha = "abc123"
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._git_result(stdout=sha + "\n"),  # rev-parse HEAD
                self._git_result(stdout=sha + "\n"),  # rev-parse remote/slug
            ],
        ) as mock_git:
            result = worker.ensure_pushed("origin", "my-branch")
        assert result is None
        assert mock_git.call_count == 2
        assert ["push", "-u", "origin", "my-branch"] not in [
            c[0][0] for c in mock_git.call_args_list
        ]

    def test_pushes_when_remote_ref_missing(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._git_result(stdout="abc\n"),  # rev-parse HEAD
                self._git_result(returncode=128),  # rev-parse remote/slug — not found
                self._git_result(),  # push succeeds
            ],
        ):
            result = worker.ensure_pushed("origin", "my-branch")
        assert result is True

    def test_pushes_when_shas_differ(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._git_result(stdout="local-sha\n"),
                self._git_result(stdout="remote-sha\n"),
                self._git_result(),
            ],
        ):
            result = worker.ensure_pushed("origin", "my-branch")
        assert result is True

    def test_returns_false_when_push_fails(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._git_result(stdout="local\n"),
                self._git_result(returncode=128),
                self._git_result(returncode=1, stderr="rejected"),
            ],
        ):
            result = worker.ensure_pushed("origin", "my-branch")
        assert result is False

    def test_logs_warning_on_push_failure(self, tmp_path: Path, caplog) -> None:
        import logging

        worker = self._make_worker(tmp_path)
        with (
            patch.object(
                worker,
                "_git",
                side_effect=[
                    self._git_result(stdout="local\n"),
                    self._git_result(returncode=128),
                    self._git_result(returncode=1, stderr="push rejected"),
                ],
            ),
            caplog.at_level(logging.WARNING, logger="kennel"),
        ):
            worker.ensure_pushed("origin", "br")
        assert "push failed" in caplog.text

    def test_passes_correct_remote_and_slug(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._git_result(stdout="sha1\n"),
                self._git_result(returncode=128),
                self._git_result(),
            ],
        ) as mock_git:
            worker.ensure_pushed("fork", "feature-branch")
        push_call = mock_git.call_args_list[2][0][0]
        assert push_call == ["push", "-u", "fork", "feature-branch"]


class TestSquashWipCommit:
    """Tests for Worker._squash_wip_commit."""

    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def _ok(self, stdout: str = "") -> MagicMock:
        r = MagicMock()
        r.returncode = 0
        r.stdout = stdout
        r.stderr = ""
        return r

    def _fail(self, stderr: str = "") -> MagicMock:
        r = MagicMock()
        r.returncode = 1
        r.stdout = ""
        r.stderr = stderr
        return r

    def _wip_git_side_effects(
        self,
        base_sha: str = "base000",
        wip_sha: str = "wip111",
        extra_subject: str = "feat: real work",
    ) -> list[MagicMock]:
        """Return _git side effects for the happy path."""
        return [
            self._ok(stdout=base_sha + "\n"),  # merge-base
            self._ok(stdout=f"{wip_sha} wip: start\nabc123 {extra_subject}\n"),  # log
            self._ok(stdout=""),  # diff-tree (empty commit)
            self._ok(),  # rebase
            self._ok(),  # push --force-with-lease
        ]

    def test_returns_true_on_success(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(worker, "_git", side_effect=self._wip_git_side_effects()):
            result = worker._squash_wip_commit("origin", "my-branch", "main")
        assert result is True

    def test_returns_false_when_merge_base_fails(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(worker, "_git", side_effect=[self._fail()]):
            result = worker._squash_wip_commit("origin", "my-branch", "main")
        assert result is False

    def test_returns_false_when_log_empty(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._ok(stdout="base000\n"),  # merge-base
                self._ok(stdout=""),  # empty log
            ],
        ):
            result = worker._squash_wip_commit("origin", "my-branch", "main")
        assert result is False

    def test_returns_false_when_log_fails(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._ok(stdout="base000\n"),  # merge-base
                self._fail(),  # log fails
            ],
        ):
            result = worker._squash_wip_commit("origin", "my-branch", "main")
        assert result is False

    def test_returns_false_when_first_commit_not_wip(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._ok(stdout="base000\n"),
                self._ok(stdout="abc123 feat: already real\n"),
            ],
        ):
            result = worker._squash_wip_commit("origin", "my-branch", "main")
        assert result is False

    def test_returns_false_when_diff_tree_fails(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._ok(stdout="base000\n"),
                self._ok(stdout="wip111 wip: start\n"),
                self._fail(),  # diff-tree fails
            ],
        ):
            result = worker._squash_wip_commit("origin", "my-branch", "main")
        assert result is False

    def test_returns_false_when_wip_commit_has_files(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._ok(stdout="base000\n"),
                self._ok(stdout="wip111 wip: start\n"),
                self._ok(stdout="100644 blob abc  file.txt\n"),  # non-empty diff-tree
            ],
        ):
            result = worker._squash_wip_commit("origin", "my-branch", "main")
        assert result is False

    def test_returns_false_when_rebase_fails(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._ok(stdout="base000\n"),
                self._ok(stdout="wip111 wip: start\nabc123 feat: real\n"),
                self._ok(stdout=""),  # diff-tree empty
                self._fail(stderr="conflict"),  # rebase fails
            ],
        ):
            result = worker._squash_wip_commit("origin", "my-branch", "main")
        assert result is False

    def test_logs_warning_when_rebase_fails(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        worker = self._make_worker(tmp_path)
        with (
            patch.object(
                worker,
                "_git",
                side_effect=[
                    self._ok(stdout="base000\n"),
                    self._ok(stdout="wip111 wip: start\nabc123 feat: real\n"),
                    self._ok(stdout=""),
                    self._fail(stderr="conflict error"),
                ],
            ),
            caplog.at_level(logging.WARNING, logger="kennel"),
        ):
            worker._squash_wip_commit("origin", "my-branch", "main")
        assert "rebase failed" in caplog.text

    def test_returns_false_when_force_push_fails(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._ok(stdout="base000\n"),
                self._ok(stdout="wip111 wip: start\nabc123 feat: real\n"),
                self._ok(stdout=""),
                self._ok(),  # rebase ok
                self._fail(stderr="rejected"),  # push fails
            ],
        ):
            result = worker._squash_wip_commit("origin", "my-branch", "main")
        assert result is False

    def test_logs_warning_when_force_push_fails(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        worker = self._make_worker(tmp_path)
        with (
            patch.object(
                worker,
                "_git",
                side_effect=[
                    self._ok(stdout="base000\n"),
                    self._ok(stdout="wip111 wip: start\nabc123 feat: real\n"),
                    self._ok(stdout=""),
                    self._ok(),
                    self._fail(stderr="push rejected"),
                ],
            ),
            caplog.at_level(logging.WARNING, logger="kennel"),
        ):
            worker._squash_wip_commit("origin", "my-branch", "main")
        assert "force-push failed" in caplog.text

    def test_logs_info_on_success(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        worker = self._make_worker(tmp_path)
        with (
            patch.object(worker, "_git", side_effect=self._wip_git_side_effects()),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker._squash_wip_commit("origin", "my-branch", "main")
        assert "squashed wip: start" in caplog.text

    def test_uses_correct_remote_and_default_branch(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker, "_git", side_effect=self._wip_git_side_effects()
        ) as mock_git:
            worker._squash_wip_commit("fork", "feat-branch", "develop")
        merge_base_call = mock_git.call_args_list[0][0][0]
        assert merge_base_call == ["merge-base", "HEAD", "fork/develop"]

    def test_force_push_uses_correct_args(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=self._wip_git_side_effects(wip_sha="wip999"),
        ) as mock_git:
            worker._squash_wip_commit("origin", "my-branch", "main")
        push_call = mock_git.call_args_list[4][0][0]
        assert push_call == ["push", "--force-with-lease", "-u", "origin", "my-branch"]

    def test_rebase_uses_correct_args(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        base = "baseaaa"
        wip = "wip999"
        with patch.object(
            worker,
            "_git",
            side_effect=self._wip_git_side_effects(base_sha=base, wip_sha=wip),
        ) as mock_git:
            worker._squash_wip_commit("origin", "my-branch", "main")
        rebase_call = mock_git.call_args_list[3][0][0]
        assert rebase_call == ["rebase", "--onto", base, wip, "my-branch"]


class TestExecuteTask:
    """Tests for Worker.execute_task."""

    def _make_worker(self, tmp_path: Path) -> tuple[Worker, MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def _repo_ctx(self) -> RepoContext:
        return RepoContext(
            repo="owner/repo",
            owner="owner",
            repo_name="repo",
            gh_user="fido-bot",
            default_branch="main",
            membership=RepoMembership(collaborators=frozenset({"owner"})),
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _pending_task(self, title: str, task_type: str = "spec") -> dict:
        return {"id": "t1", "title": title, "status": "pending", "type": task_type}

    @staticmethod
    def _git_with_new_commits():
        """Mock _git so rev-parse HEAD returns different SHAs before/after."""
        shas = iter(["aaa", "bbb"])
        orig = MagicMock()

        def side_effect(args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = next(shas, "bbb") if args == ["rev-parse", "HEAD"] else ""
            result.stderr = ""
            return result

        orig.side_effect = side_effect
        return orig

    def test_returns_false_when_no_tasks(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is False

    def test_returns_true_when_task_found(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Implement feature")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sid", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is True

    def test_calls_set_status_with_task_title(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Write the tests")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status") as mock_status,
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 5, "my-branch")
        mock_status.assert_called_once_with("Working on: Write the tests")

    def test_builds_task_prompt_with_correct_skill(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Fix the bug")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 7, "fix-branch")
        _, skill, _ = mock_bp.call_args[0]
        assert skill == "task"

    def test_context_includes_pr_and_repo(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Do work")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 42, "my-slug")
        _, _, context = mock_bp.call_args[0]
        assert "PR: 42" in context
        assert "Repo: owner/repo" in context
        assert "Branch: my-slug" in context
        assert "Upstream: origin/main" in context

    def test_context_includes_task_title(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("The special task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        _, _, context = mock_bp.call_args[0]
        assert "Task title: The special task" in context

    def test_context_includes_thread_metadata(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = {
            "id": "t1",
            "title": "Fix the thing",
            "status": "pending",
            "thread": {
                "repo": "owner/repo",
                "pr": 42,
                "comment_id": 12345,
                "url": "https://github.com/owner/repo/pull/42#discussion_r12345",
            },
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 42, "br")
        _, _, context = mock_bp.call_args[0]
        assert "comment_id: 12345" in context
        assert "review comment" in context
        assert "Thread URL:" in context

    def test_context_omits_thread_when_no_thread(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Plain task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        _, _, context = mock_bp.call_args[0]
        assert "comment_id" not in context
        assert "review comment" not in context

    def test_calls_provider_run(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sess", "")) as mock_run,
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_run.assert_called_once_with(
            fido_dir,
            model=ClaudeClient.work_model,
            cwd=tmp_path,
            session=None,
            agent=ANY,
            session_mode=TurnSessionMode.REUSE,
        )

    def test_calls_ensure_pushed_with_origin_and_slug(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True) as mock_push,
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "my-slug")
        mock_push.assert_called_once_with("origin", "my-slug")

    def test_completes_task_by_id(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("My task title")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_complete.assert_called_once_with(task["id"])

    def test_skips_complete_when_push_fails(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=False),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_complete.assert_not_called()

    def test_returns_true_when_push_fails(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=False),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert result is True

    def test_skips_sync_when_push_fails(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=False),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks") as mock_sync,
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_sync.assert_not_called()

    def test_completes_when_already_in_sync(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=None),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_complete.assert_called_once_with(task["id"])

    def test_returns_true_when_already_in_sync(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=None),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert result is True

    def test_syncs_when_already_in_sync(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=None),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks") as mock_sync,
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_sync.assert_called_once_with(tmp_path, gh)

    def test_syncs_work_queue_after_completion(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks") as mock_sync,
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_sync.assert_called_once_with(tmp_path, gh)

    def test_logs_task_name(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Log me please")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert "Log me please" in caplog.text

    def test_logs_task_done_with_session_id(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("my-session", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert "my-session" in caplog.text

    def test_resumes_session_until_commits_appear(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Big refactor")
        # head_before=aaa, head_after_1=aaa (no change), head_after_2=bbb (commit)
        shas = iter(["aaa", "aaa", "bbb"])
        git_mock = MagicMock(
            side_effect=lambda args, **kw: MagicMock(
                returncode=0,
                stdout=next(shas, "bbb") if args == ["rev-parse", "HEAD"] else "",
                stderr="",
            )
        )
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch(
                "kennel.worker.provider_run",
                side_effect=[("sess-1", "output1"), ("sess-1", "output2")],
            ) as mock_run,
            patch.object(worker, "_git", git_mock),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert mock_run.call_count == 2

    def test_starts_fresh_when_no_session_id(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        # No session_id on first run, retry writes nudge to prompt_file (no
        # build_prompt), second run produces commits
        shas = iter(["aaa", "aaa", "bbb"])
        git_mock = MagicMock(
            side_effect=lambda args, **kw: MagicMock(
                returncode=0,
                stdout=next(shas, "bbb") if args == ["rev-parse", "HEAD"] else "",
                stderr="",
            )
        )
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch(
                "kennel.worker.provider_run",
                side_effect=[("", "output"), ("sess-2", "output2")],
            ) as mock_run,
            patch.object(worker, "_git", git_mock),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        # build_prompt only called for initial setup now — retries nudge
        # in place via prompt_file directly.
        assert mock_bp.call_count == 1
        assert mock_run.call_count == 2
        # prompt_file should contain the first nudge after the retry.
        assert "commit" in (fido_dir / "prompt").read_text().lower()

    def test_keeps_retrying_across_multiple_resumes(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Big task")
        # 3 resumes before commits appear
        shas = iter(["aaa", "aaa", "aaa", "aaa", "bbb"])
        git_mock = MagicMock(
            side_effect=lambda args, **kw: MagicMock(
                returncode=0,
                stdout=next(shas, "bbb") if args == ["rev-parse", "HEAD"] else "",
                stderr="",
            )
        )
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch(
                "kennel.worker.provider_run",
                side_effect=[
                    ("sess-1", "o1"),
                    ("sess-1", "o2"),
                    ("sess-1", "o3"),
                    ("sess-1", "o4"),
                ],
            ) as mock_run,
            patch.object(worker, "_git", git_mock),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert mock_run.call_count == 4
        mock_complete.assert_called_once()

    def test_breaks_retry_loop_when_task_externally_completed(
        self, tmp_path: Path
    ) -> None:
        # Task is pending on first list_tasks call (execute_task picks it up),
        # then externally completed before the retry loop re-checks — loop
        # should break without calling provider_run a second time.
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Already done task")
        completed_task = {**task, "status": "completed"}
        # HEAD never changes — no commits will ever appear.
        git_mock = MagicMock(
            side_effect=lambda args, **kw: MagicMock(
                returncode=0,
                stdout="aaa" if args == ["rev-parse", "HEAD"] else "",
                stderr="",
            )
        )
        list_tasks_calls = iter([[task], [completed_task]])
        with (
            patch(
                "kennel.tasks.Tasks.list",
                side_effect=lambda *a, **kw: next(list_tasks_calls),
            ),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch(
                "kennel.worker.provider_run", return_value=("sess-1", "output")
            ) as mock_run,
            patch.object(worker, "_git", git_mock),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        # provider_run called exactly once (initial dispatch), not again after break
        mock_run.assert_called_once()
        # complete_by_id still called (idempotent — task already completed externally)
        mock_complete.assert_called_once_with(task["id"])

    def test_uses_fresh_session_mode_once_after_repeated_no_commit_nudges(
        self, tmp_path: Path
    ) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        (fido_dir / "prompt").write_text("initial prompt")
        task = self._pending_task("Fix widget")
        git_mock = MagicMock(
            side_effect=lambda args, **kw: MagicMock(
                returncode=0,
                stdout="aaa" if args == ["rev-parse", "HEAD"] else "",
                stderr="",
            )
        )
        prompt_snapshots: list[str] = []
        run_calls = 0

        def fake_run(fd, *args, **kwargs):
            nonlocal run_calls
            run_calls += 1
            prompt_snapshots.append((fd / "prompt").read_text())
            if run_calls == 6:
                worker._abort_task.set()
            return ("sess", "")

        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", side_effect=fake_run) as mock_run,
            patch.object(worker, "_git", git_mock),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 42, "br-42")

        session_modes = [
            call.kwargs["session_mode"] for call in mock_run.call_args_list
        ]
        assert session_modes == [
            TurnSessionMode.REUSE,
            TurnSessionMode.REUSE,
            TurnSessionMode.REUSE,
            TurnSessionMode.REUSE,
            TurnSessionMode.FRESH,
            TurnSessionMode.REUSE,
        ]
        assert "session context was intentionally wiped" in prompt_snapshots[4]
        assert "Task title: Fix widget" in prompt_snapshots[4]
        assert "Branch: br-42" in prompt_snapshots[4]

    def test_saves_current_task_id_to_state_before_provider_run(
        self, tmp_path: Path
    ) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 1})
        task = {"id": "task-99", "title": "Do stuff", "status": "pending"}
        captured: dict = {}

        def capture(fd, *args, **kwargs):
            captured.update(State(fd).load())
            return ("sess", "")

        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", side_effect=capture),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert captured.get("current_task_id") == "task-99"

    def test_clears_current_task_id_after_completion(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 1})
        task = {"id": "task-77", "title": "Complete me", "status": "pending"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sess", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert "current_task_id" not in State(fido_dir).load()

    def test_preserves_other_state_keys_when_saving_current_task_id(
        self, tmp_path: Path
    ) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 5})
        task = {"id": "t-111", "title": "Preserve", "status": "pending"}
        captured: dict = {}

        def capture(fd, *args, **kwargs):
            captured.update(State(fd).load())
            return ("sess", "")

        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", side_effect=capture),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 5, "br")
        assert captured.get("issue") == 5
        assert captured.get("current_task_id") == "t-111"

    def test_current_task_id_not_cleared_when_push_fails(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 1})
        task = {"id": "task-push-fail", "title": "Push me", "status": "pending"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sess", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=False),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert State(fido_dir).load().get("current_task_id") == "task-push-fail"

    @staticmethod
    def _git_same_sha():
        """Mock _git so rev-parse HEAD always returns the same SHA (no new commits)."""
        orig = MagicMock()

        def side_effect(args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "aaa" if args == ["rev-parse", "HEAD"] else ""
            result.stderr = ""
            return result

        orig.side_effect = side_effect
        return orig

    def test_abort_after_initial_run_removes_task_and_returns_true(
        self, tmp_path: Path
    ) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 1, "current_task_id": "t-abort"})
        task = {"id": "t-abort", "title": "Abort me", "status": "pending"}
        worker._abort_task.set()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sid", "")),
            patch.object(worker, "_git", self._git_same_sha()),
            patch.object(worker, "git_clean") as mock_clean,
            patch("kennel.tasks.Tasks.remove") as mock_remove,
            patch("kennel.tasks.sync_tasks") as mock_sync,
        ):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert result is True
        mock_clean.assert_called_once()
        mock_remove.assert_called_once_with("t-abort")
        assert "current_task_id" not in State(fido_dir).load()
        assert not worker._abort_task.is_set()
        mock_sync.assert_called()

    def test_abort_during_resume_loop_removes_task_and_returns_true(
        self, tmp_path: Path
    ) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 1})
        task = {"id": "t-resume-abort", "title": "Resume abort", "status": "pending"}
        call_count = 0

        def set_abort_on_second(fd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                worker._abort_task.set()
            return ("sid", "")

        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", side_effect=set_abort_on_second),
            patch.object(worker, "_git", self._git_same_sha()),
            patch.object(worker, "git_clean") as mock_clean,
            patch("kennel.tasks.Tasks.remove") as mock_remove,
            patch("kennel.tasks.sync_tasks") as mock_sync,
        ):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert result is True
        mock_clean.assert_called_once()
        mock_remove.assert_called_once_with("t-resume-abort")
        assert "current_task_id" not in State(fido_dir).load()
        assert not worker._abort_task.is_set()
        mock_sync.assert_called()

    def test_abort_does_not_call_complete_by_id(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 1})
        task = {"id": "t-no-complete", "title": "No complete", "status": "pending"}
        worker._abort_task.set()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("sid", "")),
            patch.object(worker, "_git", self._git_same_sha()),
            patch.object(worker, "git_clean"),
            patch("kennel.tasks.Tasks.remove"),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_complete.assert_not_called()

    def test_calls_squash_wip_commit_with_correct_args(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Do work")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(
                worker, "_squash_wip_commit", return_value=False
            ) as mock_squash,
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 7, "feat-branch")
        mock_squash.assert_called_once_with("origin", "feat-branch", "main")

    def test_squash_wip_commit_called_before_ensure_pushed(
        self, tmp_path: Path
    ) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Do work")
        call_order: list[str] = []
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(
                worker,
                "_squash_wip_commit",
                side_effect=lambda *a: call_order.append("squash") or False,
            ),
            patch.object(
                worker,
                "ensure_pushed",
                side_effect=lambda *a: call_order.append("push") or True,
            ),
            patch("kennel.tasks.Tasks.complete_by_id"),
            patch("kennel.tasks.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert call_order == ["squash", "push"]

    def test_task_completes_when_squash_returns_true(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("First task")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.provider_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "_squash_wip_commit", return_value=True),
            patch.object(worker, "ensure_pushed", return_value=None),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
            patch("kennel.tasks.sync_tasks"),
        ):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert result is True
        mock_complete.assert_called_once()


class TestRunExecuteTaskIntegration:
    """Tests that Worker.run() calls execute_task after handle_threads."""

    def _make_gh(self) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.get_default_branch.return_value = "main"
        gh.get_pr.return_value = {"body": ""}
        return gh

    def _make_mock_ctx(self, tmp_path: Path) -> MagicMock:
        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        mock_ctx.fido_dir = tmp_path / ".git" / "fido"
        return mock_ctx

    def _make_mock_repo_ctx(self) -> MagicMock:
        repo_ctx = MagicMock(spec=RepoContext)
        repo_ctx.repo = "owner/repo"
        repo_ctx.gh_user = "fido-bot"
        repo_ctx.default_branch = "main"
        repo_ctx.membership = RepoMembership()
        return repo_ctx

    def test_execute_task_called_when_all_others_return_false(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_execute = MagicMock(return_value=False)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", mock_execute),
        ):
            worker.run()
        mock_execute.assert_called_once_with(mock_ctx.fido_dir, repo_ctx, 42, "fix-bug")

    def test_returns_1_when_execute_task_done(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=True),
            patch.object(worker, "resolve_addressed_threads"),
        ):
            result = worker.run()
        assert result == 1

    def test_execute_task_not_called_when_threads_handled(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_execute = MagicMock(return_value=False)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=True),
            patch.object(worker, "execute_task", mock_execute),
        ):
            worker.run()
        mock_execute.assert_not_called()


class TestLatestDecisiveReview:
    """Tests for the latest_decisive_review module-level helper."""

    def test_empty_returns_none(self) -> None:
        assert latest_decisive_review([]) is None

    def test_approved_returned(self) -> None:
        r = {"state": "APPROVED"}
        assert latest_decisive_review([r]) is r

    def test_changes_requested_returned(self) -> None:
        r = {"state": "CHANGES_REQUESTED"}
        assert latest_decisive_review([r]) is r

    def test_commented_returns_none(self) -> None:
        assert latest_decisive_review([{"state": "COMMENTED"}]) is None

    def test_returns_last_decisive_not_last_overall(self) -> None:
        """APPROVED followed by COMMENTED → APPROVED is the decisive one."""
        approved = {"state": "APPROVED"}
        commented = {"state": "COMMENTED"}
        assert latest_decisive_review([approved, commented]) is approved

    def test_changes_requested_then_commented_returns_changes_requested(self) -> None:
        cr = {"state": "CHANGES_REQUESTED"}
        commented = {"state": "COMMENTED"}
        assert latest_decisive_review([cr, commented]) is cr

    def test_changes_requested_then_approved(self) -> None:
        cr = {"state": "CHANGES_REQUESTED"}
        approved = {"state": "APPROVED"}
        assert latest_decisive_review([cr, approved]) is approved

    def test_approved_then_changes_requested(self) -> None:
        approved = {"state": "APPROVED"}
        cr = {"state": "CHANGES_REQUESTED"}
        assert latest_decisive_review([approved, cr]) is cr


class TestShouldRerequestReview:
    """Tests for the should_rerequest_review module-level helper."""

    def _review(
        self,
        state: str = "CHANGES_REQUESTED",
        submitted_at: str = "",
    ) -> dict:
        r: dict = {"state": state}
        if submitted_at:
            r["submittedAt"] = submitted_at
        return r

    def _commit(self, committed_date: str) -> dict:
        return {"committedDate": committed_date}

    def test_empty_reviews_returns_false(self) -> None:
        assert should_rerequest_review([], []) is False

    def test_approved_returns_false(self) -> None:
        assert should_rerequest_review([self._review("APPROVED")], []) is False

    def test_commented_returns_false(self) -> None:
        assert should_rerequest_review([self._review("COMMENTED")], []) is False

    def test_changes_requested_no_dates_returns_true(self) -> None:
        assert should_rerequest_review([self._review()], []) is True

    def test_changes_requested_no_submitted_at_returns_true(self) -> None:
        commits = [self._commit("2024-01-02T12:00:00Z")]
        assert should_rerequest_review([self._review()], commits) is True

    def test_changes_requested_no_commits_returns_true(self) -> None:
        review = self._review(submitted_at="2024-01-02T12:00:00Z")
        assert should_rerequest_review([review], []) is True

    def test_review_older_than_commit_returns_true(self) -> None:
        """Review pre-dates latest commit — we addressed it."""
        review = self._review(submitted_at="2024-01-01T10:00:00Z")
        commits = [self._commit("2024-01-02T12:00:00Z")]
        assert should_rerequest_review([review], commits) is True

    def test_review_newer_than_commit_returns_false(self) -> None:
        """Review post-dates latest commit — new feedback, not yet addressed."""
        review = self._review(submitted_at="2024-01-02T12:00:00Z")
        commits = [self._commit("2024-01-01T10:00:00Z")]
        assert should_rerequest_review([review], commits) is False

    def test_uses_latest_commit_date(self) -> None:
        """Max commit date is used, not the first one."""
        review = self._review(submitted_at="2024-01-02T10:00:00Z")
        commits = [
            self._commit("2024-01-01T08:00:00Z"),
            self._commit("2024-01-03T08:00:00Z"),
        ]
        assert should_rerequest_review([review], commits) is True

    def test_uses_latest_owner_review(self) -> None:
        """Only the last review matters."""
        reviews = [
            self._review("APPROVED"),
            self._review("CHANGES_REQUESTED"),
        ]
        assert should_rerequest_review(reviews, []) is True

    def test_last_review_approved_overrides_earlier_changes_requested(self) -> None:
        reviews = [
            self._review("CHANGES_REQUESTED"),
            self._review("APPROVED"),
        ]
        assert should_rerequest_review(reviews, []) is False

    def test_changes_requested_then_commented_still_rerequests(self) -> None:
        """COMMENTED after CHANGES_REQUESTED does not override the decisive state."""
        reviews = [
            self._review("CHANGES_REQUESTED"),
            self._review("COMMENTED"),
        ]
        assert should_rerequest_review(reviews, []) is True

    def test_approved_then_commented_returns_false(self) -> None:
        """COMMENTED after APPROVED does not override the decisive APPROVED."""
        reviews = [
            self._review("APPROVED"),
            self._review("COMMENTED"),
        ]
        assert should_rerequest_review(reviews, []) is False


class TestCiReadyForReview:
    """Tests for the ci_ready_for_review module-level helper."""

    def _check(self, name: str, state: str) -> dict:
        return {"name": name, "state": state, "link": "http://..."}

    def test_no_required_checks_returns_true(self) -> None:
        checks = [self._check("ci", "IN_PROGRESS")]
        assert ci_ready_for_review(checks, []) is True

    def test_no_required_checks_no_checks_returns_true(self) -> None:
        assert ci_ready_for_review([], []) is True

    def test_all_required_passing_returns_true(self) -> None:
        checks = [
            self._check("ci / test", "SUCCESS"),
            self._check("ci / lint", "SUCCESS"),
        ]
        assert ci_ready_for_review(checks, ["ci / test", "ci / lint"]) is True

    def test_required_check_in_progress_returns_false(self) -> None:
        checks = [self._check("ci / test", "IN_PROGRESS")]
        assert ci_ready_for_review(checks, ["ci / test"]) is False

    def test_required_check_failing_returns_false(self) -> None:
        checks = [self._check("ci / test", "FAILURE")]
        assert ci_ready_for_review(checks, ["ci / test"]) is False

    def test_required_check_missing_from_checks_returns_false(self) -> None:
        checks = [self._check("ci / lint", "SUCCESS")]
        assert ci_ready_for_review(checks, ["ci / test"]) is False

    def test_partial_required_passing_returns_false(self) -> None:
        checks = [
            self._check("ci / test", "SUCCESS"),
            self._check("ci / lint", "IN_PROGRESS"),
        ]
        assert ci_ready_for_review(checks, ["ci / test", "ci / lint"]) is False

    def test_non_required_check_failing_does_not_block(self) -> None:
        checks = [
            self._check("ci / test", "SUCCESS"),
            self._check("ci / lint", "FAILURE"),
        ]
        assert ci_ready_for_review(checks, ["ci / test"]) is True


class TestHandlePromoteMerge:
    """Tests for Worker.handle_promote_merge."""

    def _make_worker(self, tmp_path: Path) -> tuple[Worker, MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def _repo_ctx(self, owner: str = "rhencke") -> RepoContext:
        return RepoContext(
            repo=f"{owner}/myrepo",
            owner=owner,
            repo_name="myrepo",
            gh_user="fido-bot",
            default_branch="main",
            membership=RepoMembership(collaborators=frozenset({owner})),
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _reviews(
        self,
        owner: str = "rhencke",
        state: str = "APPROVED",
        is_draft: bool = False,
        commits: list | None = None,
    ) -> dict:
        return {
            "reviews": [{"author": {"login": owner}, "state": state}],
            "commits": commits if commits is not None else [],
            "isDraft": is_draft,
        }

    # --- get_reviews call ---

    def test_calls_get_reviews_with_correct_args(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 7, "my-branch", 3)
        gh.get_reviews.assert_called_once_with("rhencke/myrepo", 7)

    # --- merge on approval ---

    def test_approved_not_draft_no_pending_calls_get_pr(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.get_pr.assert_called_once_with("rhencke/myrepo", 9)

    def test_approved_not_draft_no_pending_merges_squash(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_called_once_with("rhencke/myrepo", 9, squash=True)

    def test_approved_not_draft_no_pending_resets_tasks_json(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        tasks_file = fido_dir / "tasks.json"
        tasks_file.write_text('[{"id":"x","status":"completed"}]')
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert tasks_file.read_text() == "[]"

    def test_approved_not_draft_no_pending_clears_state(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 5})
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert State(fido_dir).load() == {}

    def test_approved_not_draft_no_pending_git_checkout_default(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["checkout", "main"] in calls

    def test_approved_not_draft_no_pending_git_pull_ff_only(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["pull", "origin", "main", "--ff-only"] in calls

    def test_approved_not_draft_no_pending_git_branch_delete(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["branch", "-d", "fix-slug"] in calls

    def test_approved_not_draft_no_pending_git_push_delete_remote_branch(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["push", "origin", "--delete", "fix-slug"] in calls

    def test_approved_not_draft_no_pending_sets_merged_status(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_status = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status", mock_status),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        msg = mock_status.call_args[0][0]
        assert "Merged" in msg
        assert "9" in msg
        assert "5" in msg

    def test_approved_not_draft_no_pending_returns_1(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 1

    def test_approved_not_draft_no_pending_blocked_enables_auto_merge(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_called_once_with("rhencke/myrepo", 9, squash=True, auto=True)

    def test_approved_not_draft_no_pending_blocked_returns_0(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_approved_with_pending_skips_merge(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        pending = [{"id": "t1", "title": "Do work", "status": "pending"}]
        with (
            patch("kennel.tasks.Tasks.list", return_value=pending),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_not_called()

    def test_approved_but_draft_skips_merge(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=True)
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with (
            patch("kennel.tasks.Tasks.list", return_value=completed),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_not_called()

    def test_not_approved_skips_merge(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(state="COMMENTED", is_draft=False)
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_not_called()

    # --- changes requested ---

    def test_changes_requested_ci_passing_adds_reviewer(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_called_once_with("rhencke/myrepo", 9, ["rhencke"])

    def test_changes_requested_ci_failing_skips_reviewer(self, tmp_path: Path) -> None:
        """CI not passing — re-request deferred until CI is green."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        gh.pr_checks.return_value = [{"name": "ci", "state": "FAILURE"}]
        gh.get_required_checks.return_value = ["ci"]
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_not_called()

    def test_changes_requested_ci_failing_returns_0(self, tmp_path: Path) -> None:
        """CI not passing — returns 0 (waiting for CI)."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        gh.pr_checks.return_value = [{"name": "ci", "state": "FAILURE"}]
        gh.get_required_checks.return_value = ["ci"]
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_changes_requested_returns_0(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_changes_requested_draft_marks_ready_before_rerequest(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=True
        )
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 1
        ready_call = call.pr_ready("rhencke/myrepo", 9)
        rerequest_call = call.add_pr_reviewers("rhencke/myrepo", 9, ["rhencke"])
        assert ready_call in gh.mock_calls
        assert rerequest_call in gh.mock_calls
        assert gh.mock_calls.index(ready_call) < gh.mock_calls.index(rerequest_call)

    def test_changes_requested_draft_marks_ready_without_readding_reviewer(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [{"author": {"login": "rhencke"}, "state": "CHANGES_REQUESTED"}],
            "commits": [],
            "isDraft": True,
            "requestedReviewers": ["rhencke"],
        }
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 1
        gh.pr_ready.assert_called_once_with("rhencke/myrepo", 9)
        gh.add_pr_reviewers.assert_not_called()

    def test_changes_requested_does_not_merge(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_not_called()

    def test_uses_latest_review_for_changes_requested(self, tmp_path: Path) -> None:
        """Later APPROVED overrides earlier CHANGES_REQUESTED."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {"author": {"login": "rhencke"}, "state": "CHANGES_REQUESTED"},
                {"author": {"login": "rhencke"}, "state": "APPROVED"},
            ],
            "commits": [],
            "isDraft": False,
        }
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        # APPROVED is latest — should merge, not re-request
        gh.pr_merge.assert_called_once()
        gh.add_pr_reviewers.assert_not_called()

    # --- CHANGES_REQUESTED then APPROVED merge scenario ---

    def _changes_requested_then_approved_reviews(self, is_draft: bool = False) -> dict:
        """Reviews list: earlier CHANGES_REQUESTED superseded by later APPROVED."""
        return {
            "reviews": [
                {"author": {"login": "rhencke"}, "state": "CHANGES_REQUESTED"},
                {"author": {"login": "rhencke"}, "state": "APPROVED"},
            ],
            "commits": [],
            "isDraft": is_draft,
        }

    def test_changes_req_then_approved_calls_get_pr(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.get_pr.assert_called_once_with("rhencke/myrepo", 9)

    def test_changes_req_then_approved_merges_squash(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_called_once_with("rhencke/myrepo", 9, squash=True)

    def test_changes_req_then_approved_resets_tasks_json(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        tasks_file = fido_dir / "tasks.json"
        tasks_file.write_text('[{"id":"x","status":"completed"}]')
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert tasks_file.read_text() == "[]"

    def test_changes_req_then_approved_clears_state(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 5})
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert State(fido_dir).load() == {}

    def test_changes_req_then_approved_git_checkout_default(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["checkout", "main"] in calls

    def test_changes_req_then_approved_git_pull_ff_only(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["pull", "origin", "main", "--ff-only"] in calls

    def test_changes_req_then_approved_git_branch_delete(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["branch", "-d", "fix-slug"] in calls

    def test_changes_req_then_approved_git_push_delete_remote_branch(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["push", "origin", "--delete", "fix-slug"] in calls

    def test_changes_req_then_approved_sets_merged_status(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_status = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status", mock_status),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        msg = mock_status.call_args[0][0]
        assert "Merged" in msg
        assert "9" in msg
        assert "5" in msg

    def test_changes_req_then_approved_returns_1(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 1

    def test_changes_req_then_approved_blocked_enables_auto_merge(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_called_once_with("rhencke/myrepo", 9, squash=True, auto=True)

    def test_changes_req_then_approved_blocked_returns_0(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._changes_requested_then_approved_reviews()
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_latest_changes_requested_overrides_earlier_approved(
        self, tmp_path: Path
    ) -> None:
        """Later CHANGES_REQUESTED overrides earlier APPROVED — must not merge."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "APPROVED",
                    "submittedAt": "2024-01-01T10:00:00Z",
                },
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-02T12:00:00Z",
                },
            ],
            "commits": [{"committedDate": "2024-01-01T08:00:00Z"}],
            "isDraft": False,
        }
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        # CHANGES_REQUESTED is latest — must not merge
        gh.pr_merge.assert_not_called()

    def test_changes_requested_newer_than_commit_skips_re_request(
        self, tmp_path: Path
    ) -> None:
        """Review submitted after latest commit — new feedback, don't re-request."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-02T12:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-01T10:00:00Z"}],
            "isDraft": False,
        }
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_not_called()

    def test_changes_requested_newer_than_commit_returns_0(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-02T12:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-01T10:00:00Z"}],
            "isDraft": False,
        }
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_changes_requested_older_than_commit_re_requests(
        self, tmp_path: Path
    ) -> None:
        """Review submitted before latest commit — we addressed it, re-request."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-01T10:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-02T12:00:00Z"}],
            "isDraft": False,
        }
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_called_once_with("rhencke/myrepo", 9, ["rhencke"])

    def test_changes_requested_older_than_commit_returns_0(
        self, tmp_path: Path
    ) -> None:
        """Regression for #124: after addressing feedback and pushing new commits,
        handle_promote_merge returns 0 (waiting for re-review, no more work)."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-01T10:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-02T12:00:00Z"}],
            "isDraft": False,
        }
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_changes_requested_no_submitted_at_re_requests(
        self, tmp_path: Path
    ) -> None:
        """No submittedAt on review — fall back to re-requesting."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_called_once()

    def test_changes_requested_no_commits_re_requests(self, tmp_path: Path) -> None:
        """No commits in data — fall back to re-requesting."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-02T12:00:00Z",
                }
            ],
            "commits": [],
            "isDraft": False,
        }
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_called_once()

    def test_changes_requested_newer_than_commit_logs_skip(
        self, tmp_path: Path, caplog
    ) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-02T12:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-01T10:00:00Z"}],
            "isDraft": False,
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "skipping re-request" in caplog.text

    def test_changes_requested_owner_already_requested_skips_add(
        self, tmp_path: Path
    ) -> None:
        """Owner already in requested_reviewers — don't add again."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-01T10:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-02T12:00:00Z"}],
            "isDraft": False,
            "requestedReviewers": ["rhencke"],
        }
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_not_called()

    # --- decisive review state (APPROVED/CHANGES_REQUESTED vs COMMENTED) ---

    def test_approved_then_commented_merges(self, tmp_path: Path) -> None:
        """APPROVED followed by COMMENTED: decisive state is APPROVED — must merge."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {"author": {"login": "rhencke"}, "state": "APPROVED"},
                {"author": {"login": "rhencke"}, "state": "COMMENTED"},
            ],
            "commits": [],
            "isDraft": False,
        }
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_called_once()

    def test_approved_then_commented_does_not_rerequest(self, tmp_path: Path) -> None:
        """APPROVED followed by COMMENTED: should not re-request review."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {"author": {"login": "rhencke"}, "state": "APPROVED"},
                {"author": {"login": "rhencke"}, "state": "COMMENTED"},
            ],
            "commits": [],
            "isDraft": False,
        }
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_not_called()

    def test_changes_requested_then_commented_rerequests(self, tmp_path: Path) -> None:
        """CHANGES_REQUESTED followed by COMMENTED: decisive state is
        CHANGES_REQUESTED — should re-request review, not treat as COMMENTED."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {"author": {"login": "rhencke"}, "state": "CHANGES_REQUESTED"},
                {"author": {"login": "rhencke"}, "state": "COMMENTED"},
            ],
            "commits": [],
            "isDraft": False,
        }
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_called_once_with("rhencke/myrepo", 9, ["rhencke"])

    def test_changes_requested_then_commented_does_not_merge(
        self, tmp_path: Path
    ) -> None:
        """CHANGES_REQUESTED followed by COMMENTED: must not merge."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {"author": {"login": "rhencke"}, "state": "CHANGES_REQUESTED"},
                {"author": {"login": "rhencke"}, "state": "COMMENTED"},
            ],
            "commits": [],
            "isDraft": False,
        }
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_not_called()

    # --- draft promote ---

    def test_draft_no_completed_tasks_returns_0(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_draft_no_completed_tasks_does_not_promote(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_ready.assert_not_called()

    def test_draft_with_completed_tasks_calls_pr_ready(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = []
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with patch("kennel.tasks.Tasks.list", return_value=completed):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_ready.assert_called_once_with("rhencke/myrepo", 9)

    def test_draft_with_completed_tasks_adds_reviewer(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = []
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with patch("kennel.tasks.Tasks.list", return_value=completed):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_called_once_with("rhencke/myrepo", 9, ["rhencke"])

    def test_draft_with_completed_tasks_returns_1(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = []
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with patch("kennel.tasks.Tasks.list", return_value=completed):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 1

    def test_draft_owner_already_requested_skips_add(self, tmp_path: Path) -> None:
        """Owner already in requested_reviewers — don't add again."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [],
            "commits": [],
            "isDraft": True,
            "requestedReviewers": ["rhencke"],
        }
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = []
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with patch("kennel.tasks.Tasks.list", return_value=completed):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_not_called()

    def test_draft_pending_tasks_block_promote(self, tmp_path: Path) -> None:
        """Pending tasks prevent promote — all tasks must be complete."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        tasks_list = [
            {"id": "t1", "title": "Done", "status": "completed", "type": "spec"},
            {"id": "t2", "title": "Next", "status": "pending", "type": "spec"},
        ]
        with patch("kennel.tasks.Tasks.list", return_value=tasks_list):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0
        gh.pr_ready.assert_not_called()

    # --- CI gate on draft promotion ---

    def _completed_tasks(self) -> list:
        return [{"id": "t1", "title": "Done", "status": "completed"}]

    def _passing_checks(self, name: str = "ci / test") -> list:
        return [{"name": name, "state": "SUCCESS", "link": "http://..."}]

    def test_draft_promote_ci_passing_requests_review(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = self._passing_checks()
        gh.get_required_checks.return_value = ["ci / test"]
        gh.get_review_threads.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=self._completed_tasks()):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_called_once_with("rhencke/myrepo", 9, ["rhencke"])

    def test_draft_promote_ci_not_passing_defers_review(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = [
            {"name": "ci / test", "state": "IN_PROGRESS", "link": "http://..."}
        ]
        gh.get_required_checks.return_value = ["ci / test"]
        with patch("kennel.tasks.Tasks.list", return_value=self._completed_tasks()):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_not_called()

    def test_draft_promote_ci_not_passing_does_not_call_pr_ready(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = [
            {"name": "ci / test", "state": "FAILURE", "link": "http://..."}
        ]
        gh.get_required_checks.return_value = ["ci / test"]
        with patch("kennel.tasks.Tasks.list", return_value=self._completed_tasks()):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_ready.assert_not_called()

    def test_draft_promote_ci_not_passing_returns_0(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = [
            {"name": "ci / test", "state": "IN_PROGRESS", "link": "http://..."}
        ]
        gh.get_required_checks.return_value = ["ci / test"]
        with patch("kennel.tasks.Tasks.list", return_value=self._completed_tasks()):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_draft_promote_no_required_checks_requests_review(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=self._completed_tasks()):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_called_once_with("rhencke/myrepo", 9, ["rhencke"])

    def test_draft_promote_uses_default_branch_for_required_checks(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=self._completed_tasks()):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.get_required_checks.assert_called_once_with("rhencke/myrepo", "main")

    # --- unresolved threads gate on draft promotion ---

    def _unresolved_thread(self) -> dict:
        return {"isResolved": False, "comments": {"nodes": []}}

    def _resolved_thread(self) -> dict:
        return {"isResolved": True, "comments": {"nodes": []}}

    def test_draft_promote_unresolved_threads_block_promote(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = [self._unresolved_thread()]
        with patch("kennel.tasks.Tasks.list", return_value=self._completed_tasks()):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0
        gh.pr_ready.assert_not_called()

    def test_draft_promote_resolved_threads_allow_promote(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = [self._resolved_thread()]
        with patch("kennel.tasks.Tasks.list", return_value=self._completed_tasks()):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_ready.assert_called_once_with("rhencke/myrepo", 9)

    def test_draft_promote_unresolved_threads_logs_deferring(
        self, tmp_path: Path, caplog
    ) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = [self._unresolved_thread()]
        with (
            patch(
                "kennel.tasks.Tasks.list",
                return_value=self._completed_tasks(),
            ),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "unresolved review threads" in caplog.text

    def test_draft_promote_threads_uses_correct_repo_args(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = []
        with patch("kennel.tasks.Tasks.list", return_value=self._completed_tasks()):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.get_review_threads.assert_called_once_with("rhencke", "myrepo", 9)

    # --- CI gate for non-draft PRs with no review yet ---

    def test_non_draft_no_review_ci_passing_requests_review(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        gh.pr_checks.return_value = self._passing_checks()
        gh.get_required_checks.return_value = ["ci / test"]
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_called_once_with("rhencke/myrepo", 9, ["rhencke"])

    def test_non_draft_no_review_ci_not_passing_skips_review(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        gh.pr_checks.return_value = [
            {"name": "ci / test", "state": "IN_PROGRESS", "link": "http://..."}
        ]
        gh.get_required_checks.return_value = ["ci / test"]
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_not_called()

    def test_non_draft_no_review_returns_0(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        gh.pr_checks.return_value = self._passing_checks()
        gh.get_required_checks.return_value = ["ci / test"]
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_non_draft_review_already_requested_skips_ci_check(
        self, tmp_path: Path
    ) -> None:
        """Owner already in requestedReviewers — don't poll CI, just nap."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [],
            "commits": [],
            "isDraft": False,
            "requestedReviewers": ["rhencke"],
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_checks.assert_not_called()
        gh.add_pr_reviewers.assert_not_called()

    def test_non_draft_commented_review_polls_ci(self, tmp_path: Path) -> None:
        """COMMENTED review has no decisive state — treated as NONE, CI polled."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(state="COMMENTED", is_draft=False)
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_checks.assert_called_once()

    # --- idle / no work ---

    def test_not_draft_not_approved_idle_sets_status(self, tmp_path: Path) -> None:
        """Review already requested — napping while waiting for it."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [],
            "commits": [],
            "isDraft": False,
            "requestedReviewers": ["rhencke"],
        }
        mock_status = MagicMock()
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "set_status", mock_status),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        mock_status.assert_called_once_with("Napping — waiting for work", busy=False)

    def test_not_draft_not_approved_idle_returns_0(self, tmp_path: Path) -> None:
        """Review already requested — returns 0 (waiting for review)."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [],
            "commits": [],
            "isDraft": False,
            "requestedReviewers": ["rhencke"],
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "set_status"),
        ):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    # --- logging ---

    def test_logs_review_status_check(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "set_status"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "review status" in caplog.text

    def test_logs_merge(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "merging" in caplog.text

    def test_logs_auto_merge(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "auto-merge" in caplog.text

    def test_logs_changes_requested(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "changes requested" in caplog.text

    def test_logs_not_promoting(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "not promoting" in caplog.text

    def test_logs_marking_ready(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = []
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with (
            patch("kennel.tasks.Tasks.list", return_value=completed),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "marking ready" in caplog.text

    def test_logs_no_work(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [],
            "commits": [],
            "isDraft": False,
            "requestedReviewers": ["rhencke"],
        }
        with (
            patch("kennel.tasks.Tasks.list", return_value=[]),
            patch.object(worker, "set_status"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "no work" in caplog.text

    # --- open questions (pending ASK tasks) ---

    def _ask_task(self) -> dict:
        return {
            "id": "a1",
            "title": "ASK: should I also handle X?",
            "status": "pending",
        }

    def test_pending_ask_draft_not_promoted(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        tasks_list = [
            {"id": "t1", "title": "Done", "status": "completed"},
            self._ask_task(),
        ]
        with patch("kennel.tasks.Tasks.list", return_value=tasks_list):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_ready.assert_not_called()

    def test_pending_ask_draft_review_not_requested(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        tasks_list = [
            {"id": "t1", "title": "Done", "status": "completed"},
            self._ask_task(),
        ]
        with patch("kennel.tasks.Tasks.list", return_value=tasks_list):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_not_called()

    def test_pending_ask_non_draft_review_not_requested(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        with patch("kennel.tasks.Tasks.list", return_value=[self._ask_task()]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_not_called()

    def test_pending_ask_changes_requested_rerequest_skipped(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        with patch("kennel.tasks.Tasks.list", return_value=[self._ask_task()]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewers.assert_not_called()

    def test_pending_ask_returns_zero(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        tasks_list = [
            {"id": "t1", "title": "Done", "status": "completed"},
            self._ask_task(),
        ]
        with patch("kennel.tasks.Tasks.list", return_value=tasks_list):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_pending_ask_logs_deferring(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        tasks_list = [
            {"id": "t1", "title": "Done", "status": "completed"},
            self._ask_task(),
        ]
        with (
            patch("kennel.tasks.Tasks.list", return_value=tasks_list),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "open questions" in caplog.text

    def test_does_not_promote_draft_when_pending_tasks_remain(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=True, state="NONE")
        completed = {"id": "t1", "title": "Done", "status": "completed", "type": "spec"}
        pending = {"id": "t2", "title": "Not done", "status": "pending", "type": "spec"}
        with patch("kennel.tasks.Tasks.list", return_value=[completed, pending]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 1, "branch", 1
            )
        assert result == 0
        gh.pr_ready.assert_not_called()

    def test_promotes_draft_when_all_tasks_completed(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=True, state="NONE")
        gh.pr_checks.return_value = []
        gh.get_required_checks.return_value = []
        gh.get_review_threads.return_value = []
        completed = {"id": "t1", "title": "Done", "status": "completed", "type": "spec"}
        with patch("kennel.tasks.Tasks.list", return_value=[completed]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 1, "branch", 1
            )
        assert result == 1
        gh.pr_ready.assert_called_once()


class TestRunPromoteMergeIntegration:
    """Tests that Worker.run() calls handle_promote_merge after execute_task."""

    def _make_gh(self) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.get_default_branch.return_value = "main"
        gh.get_pr.return_value = {"body": ""}
        return gh

    def _make_mock_ctx(self, tmp_path: Path) -> MagicMock:
        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        mock_ctx.fido_dir = tmp_path / ".git" / "fido"
        return mock_ctx

    def _make_mock_repo_ctx(self) -> MagicMock:
        repo_ctx = MagicMock(spec=RepoContext)
        repo_ctx.repo = "owner/repo"
        repo_ctx.gh_user = "fido-bot"
        repo_ctx.default_branch = "main"
        repo_ctx.membership = RepoMembership()
        return repo_ctx

    def test_handle_promote_merge_called_when_execute_task_returns_false(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_hpm = MagicMock(return_value=0)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", mock_hpm),
        ):
            worker.run()
        mock_hpm.assert_called_once_with(mock_ctx.fido_dir, repo_ctx, 42, "fix-bug", 7)

    def test_run_returns_0_when_handle_promote_merge_returns_0(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=0),
        ):
            result = worker.run()
        assert result == 0

    def test_run_returns_1_when_handle_promote_merge_returns_1(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=1),
        ):
            result = worker.run()
        assert result == 1

    def test_handle_promote_merge_not_called_when_execute_task_returns_true(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_hpm = MagicMock(return_value=0)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(
                worker, "find_or_create_pr", return_value=(42, "fix-bug", False)
            ),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=True),
            patch.object(worker, "resolve_addressed_threads"),
            patch.object(worker, "handle_promote_merge", mock_hpm),
        ):
            worker.run()
        mock_hpm.assert_not_called()


class TestResolveGitDirModuleLevel:
    def test_returns_path_from_stdout(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="/some/repo/.git\n"))
        assert _resolve_git_dir(tmp_path, _run=mock_run) == Path("/some/repo/.git")

    def test_passes_absolute_git_dir_flag(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="/repo/.git\n"))
        _resolve_git_dir(tmp_path, _run=mock_run)
        assert "--absolute-git-dir" in mock_run.call_args[0][0]

    def test_raises_on_subprocess_failure(self, tmp_path: Path) -> None:
        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(1, "git"))
        with pytest.raises(subprocess.CalledProcessError):
            _resolve_git_dir(tmp_path, _run=mock_run)

    def test_passes_cwd(self, tmp_path: Path) -> None:
        mock_run = MagicMock(return_value=MagicMock(stdout="/repo/.git\n"))
        _resolve_git_dir(tmp_path, _run=mock_run)
        assert mock_run.call_args.kwargs["cwd"] == tmp_path


class TestFormatWorkQueue:
    def test_empty_list_returns_empty_string(self) -> None:
        assert _format_work_queue([]) == ""

    def test_pending_task_appears_as_unchecked(self) -> None:
        tasks = [{"title": "Do work", "status": "pending", "type": "spec"}]
        result = _format_work_queue(tasks)
        assert "- [ ] Do work **→ next** <!-- type:spec -->" in result

    def test_first_pending_has_next_marker(self) -> None:
        tasks = [
            {"title": "Task A", "status": "pending", "type": "spec"},
            {"title": "Task B", "status": "pending", "type": "spec"},
        ]
        result = _format_work_queue(tasks)
        assert "Task A **→ next**" in result
        assert "Task B **→ next**" not in result

    def test_completed_tasks_appear_in_details(self) -> None:
        tasks = [{"title": "Done", "status": "completed", "type": "spec"}]
        result = _format_work_queue(tasks)
        assert "<details>" in result
        assert "- [x] Done <!-- type:spec -->" in result

    def test_ci_failure_has_priority_over_others(self) -> None:
        tasks = [
            {"title": "Normal task", "status": "pending", "type": "spec"},
            {"title": "CI failure: lint", "status": "pending", "type": "spec"},
        ]
        result = _format_work_queue(tasks)
        lines = [ln for ln in result.splitlines() if "- [ ]" in ln]
        assert "CI failure: lint" in lines[0]
        assert "Normal task" in lines[1]

    def test_thread_task_preserves_list_order_same_as_spec(self) -> None:
        tasks = [
            {"title": "Normal", "status": "pending", "type": "spec"},
            {
                "title": "Thread task",
                "status": "pending",
                "type": "thread",
                "thread": {"url": ""},
            },
        ]
        result = _format_work_queue(tasks)
        lines = [ln for ln in result.splitlines() if "- [ ]" in ln]
        # Thread tasks no longer jump the queue — list order is preserved.
        assert "Normal" in lines[0]
        assert "Thread task" in lines[1]

    def test_thread_task_with_url_becomes_link(self) -> None:
        tasks = [
            {
                "title": "Fix it",
                "status": "pending",
                "type": "thread",
                "thread": {"url": "https://github.com/comment/1"},
            }
        ]
        result = _format_work_queue(tasks)
        assert "[Fix it](https://github.com/comment/1)" in result

    def test_completed_count_in_summary(self) -> None:
        tasks = [
            {"title": "A", "status": "completed", "type": "spec"},
            {"title": "B", "status": "completed", "type": "spec"},
        ]
        result = _format_work_queue(tasks)
        assert "Completed (2)" in result

    def test_in_progress_treated_as_pending(self) -> None:
        tasks = [{"title": "Running", "status": "in_progress", "type": "spec"}]
        result = _format_work_queue(tasks)
        assert "- [ ] Running" in result

    def test_type_comment_appears_on_all_lines(self) -> None:
        tasks = [
            {"title": "Pending", "status": "pending", "type": "ci"},
            {"title": "Done", "status": "completed", "type": "thread"},
        ]
        result = _format_work_queue(tasks)
        assert "<!-- type:ci -->" in result
        assert "<!-- type:thread -->" in result


class TestApplyQueueToBody:
    def test_replaces_queue_section(self) -> None:
        body = "Before\n<!-- WORK_QUEUE_START -->\nold\n<!-- WORK_QUEUE_END -->\nAfter"
        result = _apply_queue_to_body(body, "new")
        assert "new" in result
        assert "old" not in result

    def test_preserves_content_before_and_after(self) -> None:
        body = "Before\n<!-- WORK_QUEUE_START -->\nold\n<!-- WORK_QUEUE_END -->\nAfter"
        result = _apply_queue_to_body(body, "new")
        assert result.startswith("Before")
        assert result.endswith("After")

    def test_returns_body_unchanged_when_no_start_marker(self) -> None:
        body = "No markers here\n<!-- WORK_QUEUE_END -->"
        result = _apply_queue_to_body(body, "queue")
        assert result == body

    def test_returns_body_unchanged_when_no_end_marker(self) -> None:
        body = "<!-- WORK_QUEUE_START -->\nno end marker"
        result = _apply_queue_to_body(body, "queue")
        assert result == body

    def test_returns_body_unchanged_when_no_markers(self) -> None:
        body = "plain body text"
        result = _apply_queue_to_body(body, "queue")
        assert result == body


class TestAutoCompleteAskTasks:
    def _ask_task(self, comment_id: int, task_id: str = "ask-1") -> dict:
        return {
            "id": task_id,
            "title": "ASK: some question",
            "status": "pending",
            "type": "thread",
            "thread": {"comment_id": comment_id, "url": "https://example.com"},
        }

    def _resolved_node(self, db_id: int) -> dict:
        return {
            "isResolved": True,
            "comments": {"nodes": [{"databaseId": db_id}]},
        }

    def test_no_ask_tasks_does_nothing(self, tmp_path: Path) -> None:
        gh = MagicMock()
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        gh.get_review_threads.assert_not_called()

    def test_completes_ask_task_when_thread_resolved(self, tmp_path: Path) -> None:
        gh = MagicMock()
        task = self._ask_task(42)
        gh.get_review_threads.return_value = [self._resolved_node(42)]
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_called_once_with("ask-1")

    def test_does_not_complete_ask_task_when_thread_not_resolved(
        self, tmp_path: Path
    ) -> None:
        gh = MagicMock()
        task = self._ask_task(42)
        gh.get_review_threads.return_value = [
            {"isResolved": False, "comments": {"nodes": [{"databaseId": 42}]}}
        ]
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_not_called()

    def test_get_review_threads_exception_propagates(self, tmp_path: Path) -> None:
        gh = MagicMock()
        task = self._ask_task(1)
        gh.get_review_threads.side_effect = RuntimeError("api fail")
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
            pytest.raises(RuntimeError, match="api fail"),
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_not_called()

    def test_non_ask_tasks_ignored(self, tmp_path: Path) -> None:
        gh = MagicMock()
        task = {
            "title": "Normal task",
            "status": "pending",
            "thread": {"comment_id": 1},
        }
        gh.get_review_threads.return_value = [self._resolved_node(1)]
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_not_called()

    def test_ask_task_without_thread_ignored(self, tmp_path: Path) -> None:
        gh = MagicMock()
        task = {"title": "ASK: question", "status": "pending"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_not_called()
        gh.get_review_threads.assert_not_called()

    def test_resolved_node_without_database_id_skipped(self, tmp_path: Path) -> None:
        gh = MagicMock()
        task = self._ask_task(42)
        gh.get_review_threads.return_value = [
            {"isResolved": True, "comments": {"nodes": [{}]}}
        ]
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            patch("kennel.tasks.Tasks.complete_by_id") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_not_called()


class TestSyncTasks:
    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _state_with_issue(self, fido_dir: Path, issue: int = 1) -> None:
        (fido_dir / "state.json").write_text(f'{{"issue": {issue}}}')

    def _sync_kwargs(self, fido_dir: Path) -> dict:
        """Return injection kwargs for sync_tasks pointing _resolve_git_dir at fido_dir.parent."""
        return {
            "_resolve_git_dir_fn": MagicMock(return_value=fido_dir.parent),
            "_auto_complete_ask_tasks_fn": MagicMock(),
        }

    def test_warns_when_git_dir_not_resolved(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = MagicMock()
        with caplog.at_level(logging.WARNING, logger="kennel"):
            sync_tasks(
                tmp_path,
                gh,
                _resolve_git_dir_fn=MagicMock(
                    side_effect=subprocess.CalledProcessError(1, "git")
                ),
            )
        assert "could not resolve git dir" in caplog.text

    def test_returns_early_when_no_issue_in_state(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))
        gh.find_pr.assert_not_called()

    def test_returns_early_when_lock_held(self, tmp_path: Path) -> None:
        import fcntl

        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        sync_lock_path = fido_dir / "sync.lock"
        with open(sync_lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))
        gh.find_pr.assert_not_called()

    def test_returns_early_when_no_open_pr(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = None
        sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))
        gh.get_pr_body.assert_not_called()

    def test_returns_early_when_pr_not_open(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "MERGED"}
        sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))
        gh.get_pr_body.assert_not_called()

    def test_returns_early_when_no_tasks(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        with patch("kennel.tasks.Tasks.list", return_value=[]):
            sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))
        gh.get_pr_body.assert_not_called()

    def test_syncs_pr_body_when_queue_markers_present(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        body = "desc\n<!-- WORK_QUEUE_START -->\nold\n<!-- WORK_QUEUE_END -->\nfooter"
        gh.get_pr_body.return_value = body
        task = {"title": "Do it", "status": "pending"}
        with patch("kennel.tasks.Tasks.list", return_value=[task]):
            sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))
        gh.edit_pr_body.assert_called_once()
        new_body = gh.edit_pr_body.call_args[0][2]
        assert "Do it" in new_body
        assert "old" not in new_body

    def test_description_section_preserved_after_sync(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        desc = "My PR description.\n\nFixes #1."
        body = (
            f"{desc}\n\n---\n\n## Work queue\n\n"
            "<!-- WORK_QUEUE_START -->\nold queue\n<!-- WORK_QUEUE_END -->"
        )
        gh.get_pr_body.return_value = body
        task = {"title": "Do it", "status": "pending", "type": "spec"}
        with patch("kennel.tasks.Tasks.list", return_value=[task]):
            sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))
        gh.edit_pr_body.assert_called_once()
        new_body = gh.edit_pr_body.call_args[0][2]
        assert "My PR description." in new_body
        assert "Fixes #1." in new_body

    def test_skips_edit_when_no_queue_markers(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        gh.get_pr_body.return_value = "no markers here"
        task = {"title": "Do it", "status": "pending"}
        with patch("kennel.tasks.Tasks.list", return_value=[task]):
            sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))
        gh.edit_pr_body.assert_not_called()

    def test_get_repo_info_exception_propagates(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.side_effect = RuntimeError("no remote")
        with pytest.raises(RuntimeError, match="no remote"):
            sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))
        gh.find_pr.assert_not_called()

    def test_get_pr_body_exception_propagates(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        gh.get_pr_body.side_effect = RuntimeError("api down")
        task = {"title": "Do it", "status": "pending"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            pytest.raises(RuntimeError, match="api down"),
        ):
            sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))
        gh.edit_pr_body.assert_not_called()

    def test_edit_pr_body_exception_propagates(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        body = "desc\n<!-- WORK_QUEUE_START -->\nold\n<!-- WORK_QUEUE_END -->"
        gh.get_pr_body.return_value = body
        gh.edit_pr_body.side_effect = RuntimeError("api down")
        task = {"title": "Do it", "status": "pending"}
        with (
            patch("kennel.tasks.Tasks.list", return_value=[task]),
            pytest.raises(RuntimeError, match="api down"),
        ):
            sync_tasks(tmp_path, gh, **self._sync_kwargs(fido_dir))


class TestSyncTasksBackground:
    def test_starts_daemon_thread(self, tmp_path: Path) -> None:
        gh = MagicMock()
        started: list = []
        sync_tasks_background(tmp_path, gh, _start=started.append)
        assert len(started) == 1
        assert started[0].daemon is True

    def test_thread_name_includes_dir_name(self, tmp_path: Path) -> None:
        gh = MagicMock()
        captured: list = []
        sync_tasks_background(tmp_path, gh, _start=captured.append)
        assert tmp_path.name in captured[0].name

    def test_thread_target_is_sync_tasks(self, tmp_path: Path) -> None:
        gh = MagicMock()
        captured: list = []
        sync_tasks_background(tmp_path, gh, _start=captured.append)
        assert captured[0]._target is sync_tasks


class TestWorkerThread:
    def _make_thread(self, tmp_path: Path) -> WorkerThread:
        return WorkerThread(tmp_path, "owner/repo", MagicMock())

    # ── constructor / attributes ──────────────────────────────────────────

    def test_is_daemon(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert wt.daemon is True

    def test_name_includes_work_dir_name(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert tmp_path.name in wt.name

    def test_work_dir_stored(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert wt.work_dir == tmp_path

    # ── wake / stop ───────────────────────────────────────────────────────

    def test_wake_sets_event(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert not wt._wake.is_set()
        wt.wake()
        assert wt._wake.is_set()

    def test_stop_sets_flag_and_wakes(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        wt.stop()
        assert wt._stop is True
        assert wt._wake.is_set()

    # ── loop behaviour ────────────────────────────────────────────────────

    def _run_thread(self, wt: "WorkerThread", timeout: float = 5.0) -> None:
        """Start thread and join with timeout. Fail if it hangs."""
        wt.start()
        wt.join(timeout=timeout)
        assert not wt.is_alive(), "WorkerThread hung — did not exit within timeout"

    def test_return_1_loops_immediately_without_waiting(self, tmp_path: Path) -> None:
        """Return 1 (did work) should never call _wake.wait."""
        wt = self._make_thread(tmp_path)
        mock_wake = MagicMock()
        wt._wake = mock_wake
        calls: list[int] = []

        def fake_worker_run(self_ignored=None) -> int:
            calls.append(len(calls))
            if len(calls) < 3:
                return 1
            wt._stop = True
            return 1

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        assert len(calls) == 3
        mock_wake.wait.assert_not_called()

    def test_return_0_waits_with_idle_timeout(self, tmp_path: Path) -> None:
        """Return 0 (no work) should call _wake.wait with _IDLE_TIMEOUT."""
        import kennel.worker as wmod

        wt = self._make_thread(tmp_path)
        mock_wake = MagicMock()
        wt._wake = mock_wake

        def fake_worker_run(self_ignored=None) -> int:
            wt._stop = True
            return 0

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        mock_wake.wait.assert_called_once_with(timeout=wmod._IDLE_TIMEOUT)

    def test_return_2_waits_with_retry_timeout(self, tmp_path: Path) -> None:
        """Return 2 (lock contended) should call _wake.wait with _RETRY_TIMEOUT."""
        import kennel.worker as wmod

        wt = self._make_thread(tmp_path)
        mock_wake = MagicMock()
        wt._wake = mock_wake

        def fake_worker_run(self_ignored=None) -> int:
            wt._stop = True
            return 2

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        mock_wake.wait.assert_called_once_with(timeout=wmod._RETRY_TIMEOUT)

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_exception_kills_thread(self, tmp_path: Path) -> None:
        """An unexpected exception propagates and kills the thread."""
        wt = self._make_thread(tmp_path)

        def fake_worker_run(self_ignored=None) -> int:
            raise RuntimeError("boom")

        with patch.object(Worker, "run", fake_worker_run):
            wt.start()
            wt.join(timeout=5.0)

        assert not wt.is_alive()

    def test_crash_error_is_none_initially(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert wt.crash_error is None

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_crash_error_set_on_exception(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)

        def fake_worker_run(self_ignored=None) -> int:
            raise ValueError("bad value")

        with patch.object(Worker, "run", fake_worker_run):
            wt.start()
            wt.join(timeout=5.0)

        assert wt.crash_error == "ValueError: bad value"

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_crash_error_includes_exception_type(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)

        def fake_worker_run(self_ignored=None) -> int:
            raise RuntimeError("boom")

        with patch.object(Worker, "run", fake_worker_run):
            wt.start()
            wt.join(timeout=5.0)

        assert wt.crash_error is not None
        assert wt.crash_error.startswith("RuntimeError:")

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_crash_error_logs_exception(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)

        def fake_worker_run(self_ignored=None) -> int:
            raise RuntimeError("boom")

        with patch.object(Worker, "run", fake_worker_run):
            with patch("kennel.worker.log") as mock_log:
                wt.start()
                wt.join(timeout=5.0)

        mock_log.exception.assert_called_once()

    def test_stop_flag_exits_loop_before_next_iteration(self, tmp_path: Path) -> None:
        """Setting stop before run() starts should cause immediate exit."""
        wt = self._make_thread(tmp_path)
        wt.stop()
        with patch.object(Worker, "run") as mock_run:
            wt.run()
        mock_run.assert_not_called()

    def test_wake_clears_event_after_wait(self, tmp_path: Path) -> None:
        """_wake event is cleared after each wait so the next idle wait blocks."""
        wt = self._make_thread(tmp_path)
        mock_wake = MagicMock()
        wt._wake = mock_wake
        call_count = 0

        def fake_worker_run(self_ignored=None) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                wt._stop = True
            return 0

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        # clear() called after each wait
        assert mock_wake.clear.call_count == 2

    # ── abort_task ────────────────────────────────────────────────────────

    def test_abort_task_sets_abort_event(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert not wt._abort_task.is_set()
        wt.abort_task()
        assert wt._abort_task.is_set()

    def test_abort_task_also_wakes_thread(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert not wt._wake.is_set()
        wt.abort_task()
        assert wt._wake.is_set()

    def test_run_sets_thread_local_repo_name(self, tmp_path: Path) -> None:
        """WorkerThread.run() sets _thread_repo.repo_name to the short name."""
        wt = WorkerThread(tmp_path, "owner/myrepo", MagicMock())
        wt._wake = MagicMock()
        captured: list[str] = []

        def fake_worker_run(self_w):
            import kennel.worker as wmod

            captured.append(getattr(wmod._thread_repo, "repo_name", None))
            wt._stop = True
            return 0

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        assert captured == ["myrepo"]

    def test_abort_event_passed_to_worker(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        wt._wake = MagicMock()
        captured: list = []

        def fake_worker_init(
            self_w,
            work_dir,
            gh,
            abort_task=None,
            repo_name="",
            registry=None,
            membership=None,
            session=None,
            session_issue=None,
            config=None,
            repo_cfg=None,
            provider_factory=None,
        ):
            del provider_factory
            captured.append(abort_task)
            self_w.work_dir = work_dir
            self_w.gh = gh
            self_w._abort_task = abort_task
            self_w._session = session
            self_w._session_issue = session_issue

        def fake_worker_run(self_w):
            wt._stop = True
            return 0

        with (
            patch.object(Worker, "__init__", fake_worker_init),
            patch.object(Worker, "run", fake_worker_run),
        ):
            self._run_thread(wt)

        assert len(captured) == 1
        assert captured[0] is wt._abort_task

    def test_heartbeat_emitted_each_iteration(self, tmp_path: Path) -> None:
        """report_activity is called at the top of each loop iteration."""
        mock_registry = MagicMock()
        wt = WorkerThread(tmp_path, "owner/repo", MagicMock(), registry=mock_registry)
        wt._wake = MagicMock()
        call_count = 0

        def fake_worker_run(self_ignored=None) -> int:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                wt._stop = True
            return 0

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        assert call_count == 2
        assert mock_registry.report_activity.call_count == 2
        for c in mock_registry.report_activity.call_args_list:
            assert c.args[0] == "owner/repo"
            assert c.kwargs.get("busy") is False or c.args[2] is False

    def test_heartbeat_not_emitted_when_no_registry(self, tmp_path: Path) -> None:
        """WorkerThread without a registry must not crash on the heartbeat path."""
        wt = WorkerThread(tmp_path, "owner/repo", MagicMock(), registry=None)
        wt._wake = MagicMock()

        def fake_worker_run(self_ignored=None) -> int:
            wt._stop = True
            return 0

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)  # must not raise

    # ── session lifecycle ─────────────────────────────────────────────────

    def test_session_carried_to_next_iteration(self, tmp_path: Path) -> None:
        """Session pre-created by WorkerThread on first iteration is carried forward."""
        wt = self._make_thread(tmp_path)
        wt._wake = MagicMock()
        pre_session = MagicMock()
        wt._create_session = MagicMock(return_value=pre_session)
        sessions_received: list = []

        def fake_worker_init(
            self_w,
            work_dir,
            gh,
            abort_task=None,
            repo_name="",
            registry=None,
            membership=None,
            session=None,
            session_issue=None,
            config=None,
            repo_cfg=None,
            provider_factory=None,
        ) -> None:
            del provider_factory
            self_w.work_dir = work_dir
            self_w.gh = gh
            self_w._abort_task = abort_task
            self_w._session = session
            self_w._session_issue = session_issue
            sessions_received.append(session)

        call_count = 0

        def fake_worker_run(self_w) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1
            wt._stop = True
            return 0

        with (
            patch.object(Worker, "__init__", fake_worker_init),
            patch.object(Worker, "run", fake_worker_run),
        ):
            self._run_thread(wt)

        assert len(sessions_received) == 2
        # WorkerThread pre-creates the session before the first Worker runs,
        # so the first Worker already inherits it, and subsequent Workers
        # continue to inherit it (no carry-over / re-create on iteration 2).
        assert sessions_received[0] is pre_session
        assert sessions_received[1] is pre_session
        wt._create_session.assert_called_once()

    def test_session_issue_carried_to_next_iteration(self, tmp_path: Path) -> None:
        """session_issue set by one Worker.run() is passed to the next."""
        wt = self._make_thread(tmp_path)
        wt._wake = MagicMock()
        issues_received: list = []

        def fake_worker_init(
            self_w,
            work_dir,
            gh,
            abort_task=None,
            repo_name="",
            registry=None,
            membership=None,
            session=None,
            session_issue=None,
            config=None,
            repo_cfg=None,
            provider_factory=None,
        ) -> None:
            del provider_factory
            self_w.work_dir = work_dir
            self_w.gh = gh
            self_w._abort_task = abort_task
            self_w._session = session
            self_w._session_issue = session_issue
            issues_received.append(session_issue)

        call_count = 0

        def fake_worker_run(self_w) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                self_w._session_issue = 42  # first worker picks issue 42
                return 1
            wt._stop = True
            return 0

        with (
            patch.object(Worker, "__init__", fake_worker_init),
            patch.object(Worker, "run", fake_worker_run),
        ):
            self._run_thread(wt)

        assert len(issues_received) == 2
        assert issues_received[0] is None  # no carry-over on first iteration
        assert issues_received[1] == 42  # carried forward

    def test_create_session_raises_when_provider_does_not_attach_one(
        self, tmp_path: Path
    ) -> None:
        wt = self._make_thread(tmp_path)
        provider = MagicMock()
        provider.agent.session = None
        wt._provider = provider
        with pytest.raises(
            RuntimeError, match="provider.ensure_session\\(\\) returned no session"
        ):
            wt._create_session()

    def test_session_stopped_when_thread_exits(self, tmp_path: Path) -> None:
        """WorkerThread stops the session when its loop finishes."""
        wt = self._make_thread(tmp_path)
        mock_session = MagicMock()
        wt._session = mock_session
        wt._stop = True  # exit immediately without running any Worker

        with patch.object(Worker, "run"):
            wt.run()

        mock_session.stop.assert_called_once()
        assert wt._session is None

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_session_preserved_when_worker_raises(self, tmp_path: Path) -> None:
        """WorkerThread leaves the session alive when Worker.run() raises.

        The registry rescues the live session and passes it to the replacement
        thread so the persistent ClaudeSession survives the crash.
        """
        wt = self._make_thread(tmp_path)
        mock_session = MagicMock()

        def fake_worker_run(self_w) -> int:
            self_w._session = mock_session
            raise RuntimeError("boom")

        with patch.object(Worker, "run", fake_worker_run):
            wt.start()
            wt.join(timeout=5.0)

        assert not wt.is_alive()
        mock_session.stop.assert_not_called()  # session must NOT be stopped
        assert wt._session is mock_session  # still reachable for registry to rescue

    def test_session_accepted_via_constructor(self, tmp_path: Path) -> None:
        """Session passed to WorkerThread constructor is used as the initial session."""
        mock_session = MagicMock()
        wt = WorkerThread(
            tmp_path, "owner/repo", MagicMock(), session=mock_session, session_issue=7
        )
        assert wt._session is mock_session
        assert wt._session_issue == 7

    def test_provider_accepted_via_constructor(self, tmp_path: Path) -> None:
        provider = MagicMock()
        provider.agent = MagicMock(spec=ClaudeClient)
        provider.agent.session = MagicMock()
        session = MagicMock()
        wt = WorkerThread(
            tmp_path,
            "owner/repo",
            MagicMock(),
            provider=provider,
            session=session,
            session_issue=7,
        )
        provider.agent.attach_session.assert_called_once_with(session)
        assert wt.current_provider() is provider
        assert wt._session is provider.agent.session
        assert wt._session_issue == 7

    def test_detach_provider_returns_and_clears(self, tmp_path: Path) -> None:
        provider = MagicMock()
        provider.agent = MagicMock(spec=ClaudeClient)
        wt = WorkerThread(tmp_path, "owner/repo", MagicMock(), provider=provider)
        assert wt.detach_provider() is provider
        assert wt.current_provider() is None

    def test_session_setter_recreates_provider_when_detached(
        self, tmp_path: Path
    ) -> None:
        wt = self._make_thread(tmp_path)
        wt.detach_provider()
        session = MagicMock()
        wt._session = session
        assert wt.current_provider() is not None
        assert wt._session is session

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_run_recreates_provider_when_missing(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        wt.detach_provider()
        wt._wake = MagicMock()
        carried_session = MagicMock()

        def fake_worker_init(
            self_w,
            work_dir,
            gh,
            abort_task=None,
            repo_name="",
            registry=None,
            membership=None,
            session=None,
            session_issue=None,
            config=None,
            repo_cfg=None,
            provider_factory=None,
        ) -> None:
            del provider_factory
            self_w.work_dir = work_dir
            self_w.gh = gh
            self_w._abort_task = abort_task
            self_w._session = session
            self_w._session_issue = session_issue

        def fake_worker_run(self_w) -> int:
            wt._stop = True
            return 0

        with (
            patch.object(
                WorkerThread, "_session", new_callable=PropertyMock
            ) as session_prop,
            patch.object(Worker, "__init__", fake_worker_init),
            patch.object(Worker, "run", fake_worker_run),
        ):
            session_prop.return_value = carried_session
            self._run_thread(wt)

        assert wt.current_provider() is not None

    def test_session_owner_returns_none_when_no_session(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert wt._session is None
        assert wt.session_owner is None

    def test_session_owner_delegates_to_session(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        mock_session = MagicMock()
        mock_session.owner = "worker-home"
        wt._session = mock_session
        assert wt.session_owner == "worker-home"

    def test_session_alive_false_when_no_session(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert wt._session is None
        assert wt.session_alive is False

    def test_session_pid_none_when_no_session(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert wt._session is None
        assert wt.session_pid is None

    def test_session_pid_delegates_to_session_pid(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        mock_session = MagicMock()
        mock_session.pid = 54321
        wt._session = mock_session
        assert wt.session_pid == 54321

    def test_session_alive_delegates_to_session_is_alive(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        mock_session = MagicMock()
        mock_session.is_alive.return_value = True
        wt._session = mock_session
        assert wt.session_alive is True
        mock_session.is_alive.return_value = False
        assert wt.session_alive is False

    def test_run_halts_on_claude_leak_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ClaudeLeakError in the worker loop calls os._exit(3)."""
        from kennel import claude
        from kennel import worker as worker_module

        wt = self._make_thread(tmp_path)
        exits: list[int] = []
        monkeypatch.setattr(worker_module.os, "_exit", exits.append)
        # Force the loop to raise a leak error on the first iteration.
        wt._registry = MagicMock()
        wt._registry.report_activity.side_effect = claude.ClaudeLeakError("leak")
        wt.run()
        assert exits == [3]

    # ── config / repo_cfg injection ───────────────────────────────────────

    def test_config_stored_when_passed(self, tmp_path: Path) -> None:
        from kennel.config import Config, RepoConfig

        cfg = RepoConfig(
            name="owner/repo", work_dir=tmp_path, provider=ProviderID.CLAUDE_CODE
        )
        config = Config(
            port=9000,
            secret=b"s",
            repos={"owner/repo": cfg},
            allowed_bots=frozenset(),
            log_level="DEBUG",
            sub_dir=tmp_path,
        )
        wt = WorkerThread(tmp_path, "owner/repo", MagicMock(), config=config)
        assert wt._config is config

    def test_repo_cfg_stored_when_passed(self, tmp_path: Path) -> None:
        from kennel.config import RepoConfig

        cfg = RepoConfig(
            name="owner/repo", work_dir=tmp_path, provider=ProviderID.CLAUDE_CODE
        )
        wt = WorkerThread(tmp_path, "owner/repo", MagicMock(), repo_cfg=cfg)
        assert wt._repo_cfg is cfg

    def test_repo_cfg_provider_selects_copilot_provider(self, tmp_path: Path) -> None:
        from kennel.config import RepoConfig

        wt = WorkerThread(
            tmp_path,
            "owner/repo",
            MagicMock(),
            repo_cfg=RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.COPILOT_CLI,
            ),
        )
        assert wt._provider is not None  # pyright: ignore[reportPrivateUsage]
        assert wt._provider.provider_id == ProviderID.COPILOT_CLI  # pyright: ignore[reportPrivateUsage]

    def test_config_defaults_to_none(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert wt._config is None

    def test_repo_cfg_defaults_to_none(self, tmp_path: Path) -> None:
        wt = WorkerThread(
            tmp_path, "owner/repo", MagicMock(), repo_cfg=None, provider=MagicMock()
        )
        assert wt._repo_cfg is None

    def test_provider_defaults_to_none_when_repo_cfg_is_none(
        self, tmp_path: Path
    ) -> None:
        wt = WorkerThread(tmp_path, "owner/repo", MagicMock(), repo_cfg=None)
        assert wt.current_provider() is None
        assert wt._session is None

    def test_session_getter_reads_bootstrap_session_when_provider_missing(
        self, tmp_path: Path
    ) -> None:
        session = MagicMock()
        wt = WorkerThread(
            tmp_path, "owner/repo", MagicMock(), repo_cfg=None, session=session
        )
        assert wt.current_provider() is None
        assert wt._session is session

    def test_session_setter_stores_bootstrap_session_when_repo_cfg_none(
        self, tmp_path: Path
    ) -> None:
        wt = WorkerThread(tmp_path, "owner/repo", MagicMock(), repo_cfg=None)
        session = MagicMock()
        wt._session = session
        assert wt.current_provider() is None
        assert wt._session is session

    def test_ensure_provider_requires_repo_cfg(self, tmp_path: Path) -> None:
        wt = WorkerThread(tmp_path, "owner/repo", MagicMock(), repo_cfg=None)
        with pytest.raises(
            RuntimeError, match="worker thread provider requires explicit repo_cfg"
        ):
            wt._ensure_provider()  # pyright: ignore[reportPrivateUsage]

    def test_config_and_repo_cfg_passed_to_worker(self, tmp_path: Path) -> None:
        """WorkerThread.run() forwards config and repo_cfg to every Worker."""
        from kennel.config import Config, RepoConfig

        cfg = RepoConfig(
            name="owner/repo", work_dir=tmp_path, provider=ProviderID.CLAUDE_CODE
        )
        config = Config(
            port=9000,
            secret=b"s",
            repos={"owner/repo": cfg},
            allowed_bots=frozenset(),
            log_level="DEBUG",
            sub_dir=tmp_path,
        )
        wt = WorkerThread(
            tmp_path, "owner/repo", MagicMock(), config=config, repo_cfg=cfg
        )
        wt._wake = MagicMock()
        received_config: list = []
        received_repo_cfg: list = []

        def fake_worker_init(
            self_w,
            work_dir,
            gh,
            abort_task=None,
            repo_name="",
            registry=None,
            membership=None,
            session=None,
            session_issue=None,
            config=None,
            repo_cfg=None,
            provider_factory=None,
        ) -> None:
            del provider_factory
            self_w.work_dir = work_dir
            self_w.gh = gh
            self_w._abort_task = abort_task
            self_w._session = session
            self_w._session_issue = session_issue
            received_config.append(config)
            received_repo_cfg.append(repo_cfg)

        def fake_worker_run(self_w) -> int:
            wt._stop = True
            return 0

        with (
            patch.object(Worker, "__init__", fake_worker_init),
            patch.object(Worker, "run", fake_worker_run),
        ):
            wt.run()

        assert received_config == [config]
        assert received_repo_cfg == [cfg]
