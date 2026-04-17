"""Tests for Worker._commit_provider_leftovers_if_any (#654)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from kennel.github import GitHub
from kennel.worker import Worker


def _init_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=tmp_path, check=True)


def _worker(tmp_path: Path) -> Worker:
    gh = MagicMock(spec=GitHub)
    return Worker(tmp_path, gh)


def _head(tmp_path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_noop_when_head_already_moved(tmp_path: Path) -> None:
    """Provider made its own commit → helper returns current HEAD unchanged."""
    _init_repo(tmp_path)
    head_before = _head(tmp_path)
    (tmp_path / "a.txt").write_text("a\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "real provider commit"],
        cwd=tmp_path,
        check=True,
    )
    worker = _worker(tmp_path)
    new_head = worker._commit_provider_leftovers_if_any("Fix thing", head_before)
    assert new_head != head_before
    # Only one commit was added (no synthetic wip).
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "provider didn't commit" not in log


def test_noop_when_worktree_clean(tmp_path: Path) -> None:
    """Provider produced nothing — helper returns HEAD unchanged, no commit."""
    _init_repo(tmp_path)
    head_before = _head(tmp_path)
    worker = _worker(tmp_path)
    new_head = worker._commit_provider_leftovers_if_any("Empty task", head_before)
    assert new_head == head_before
    log = (
        subprocess.run(
            ["git", "log", "--oneline"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    assert len(log) == 1  # just the seed commit


def test_commits_uncommitted_modifications(tmp_path: Path) -> None:
    """Modified files with no commit → helper commits them itself."""
    _init_repo(tmp_path)
    head_before = _head(tmp_path)
    (tmp_path / "seed.txt").write_text("modified\n")
    worker = _worker(tmp_path)
    new_head = worker._commit_provider_leftovers_if_any("Fix seed", head_before)
    assert new_head != head_before
    # Commit message includes the synthetic marker.
    msg = subprocess.run(
        ["git", "log", "-1", "--format=%s"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert msg == "wip: Fix seed (provider didn't commit)"


def test_commits_untracked_files(tmp_path: Path) -> None:
    """Untracked files → helper commits them via ``git add -A``."""
    _init_repo(tmp_path)
    head_before = _head(tmp_path)
    (tmp_path / "new.txt").write_text("brand new\n")
    worker = _worker(tmp_path)
    new_head = worker._commit_provider_leftovers_if_any("Add new file", head_before)
    assert new_head != head_before
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "new.txt" in tracked


def test_returns_unchanged_head_when_status_fails(tmp_path: Path, monkeypatch) -> None:
    """If ``git status --porcelain`` itself fails (transient VFS issue etc.),
    helper returns head_after without attempting a commit — don't crash on
    a filesystem hiccup; the resume loop's next iteration tries again."""
    _init_repo(tmp_path)
    head_before = _head(tmp_path)
    worker = _worker(tmp_path)

    real_git = worker._git

    def fake_git(args, check=True, **kwargs):
        if args[:2] == ["status", "--porcelain"]:
            result = subprocess.CompletedProcess(args, 1, "", "vfs down")
            return result
        return real_git(args, check=check, **kwargs)

    monkeypatch.setattr(worker, "_git", fake_git)
    new_head = worker._commit_provider_leftovers_if_any("Retry", head_before)
    assert new_head == head_before
