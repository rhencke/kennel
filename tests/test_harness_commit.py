"""Tests for fido.harness_commit — harness-owned commit logic."""

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fido.harness_commit import (
    MAX_HOOK_RETRIES,
    CommitHookFailure,
    CommitNothingStaged,
    CommitResult,
    CommitSkipped,
    CommitSuccess,
    HarnessCommitter,
    hook_failure_nudge,
)
from fido.turn_outcome import (
    CommitTaskComplete,
    CommitTaskInProgress,
    SkipTaskWithReason,
)
from fido.types import GitIdentity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr)


def _fail(
    returncode: int = 1, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class _FakeRunner:
    """ProcessRunner fake that returns pre-programmed results in order."""

    def __init__(self, results: list[subprocess.CompletedProcess[str]]) -> None:
        self._results = list(results)
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def run(
        self, cmd: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((cmd, kwargs))
        return self._results.pop(0)


def _committer(
    work_dir: Path, results: list[subprocess.CompletedProcess[str]]
) -> tuple[HarnessCommitter, _FakeRunner]:
    runner = _FakeRunner(results)
    return HarnessCommitter(work_dir, runner), runner


# ---------------------------------------------------------------------------
# Tests: SkipTaskWithReason → CommitSkipped
# ---------------------------------------------------------------------------


class TestApplySkip:
    def test_returns_commit_skipped(self, tmp_path: Path) -> None:
        hc, runner = _committer(tmp_path, [])
        result = hc.apply(SkipTaskWithReason(reason="already done"))
        assert result == CommitSkipped(reason="already done")
        assert runner.calls == []

    def test_skip_ignores_committer(self, tmp_path: Path) -> None:
        """Committer arg is irrelevant for skip outcomes."""
        hc, runner = _committer(tmp_path, [])
        result = hc.apply(
            SkipTaskWithReason(reason="nope"),
            committer=GitIdentity(name="X", email="x@y"),
        )
        assert isinstance(result, CommitSkipped)
        assert runner.calls == []


# ---------------------------------------------------------------------------
# Tests: nothing staged → CommitNothingStaged
# ---------------------------------------------------------------------------


class TestApplyNothingStaged:
    def test_commit_complete_nothing_staged(self, tmp_path: Path) -> None:
        hc, runner = _committer(
            tmp_path,
            [
                _ok(),  # git add -u
                _ok(),  # git diff --cached --quiet → returncode 0 = clean
            ],
        )
        result = hc.apply(CommitTaskComplete(summary="Add feature"))
        assert result == CommitNothingStaged()
        assert len(runner.calls) == 2
        assert runner.calls[0][0] == ["git", "add", "-u"]
        assert runner.calls[1][0] == ["git", "diff", "--cached", "--quiet"]

    def test_commit_in_progress_nothing_staged(self, tmp_path: Path) -> None:
        hc, _ = _committer(
            tmp_path,
            [
                _ok(),  # git add -u
                _ok(),  # git diff --cached --quiet
            ],
        )
        result = hc.apply(CommitTaskInProgress(summary="wip"))
        assert result == CommitNothingStaged()


# ---------------------------------------------------------------------------
# Tests: successful commit → CommitSuccess
# ---------------------------------------------------------------------------


class TestApplyCommitSuccess:
    def test_commit_complete_success(self, tmp_path: Path) -> None:
        hc, runner = _committer(
            tmp_path,
            [
                _ok(),  # git add -u
                _fail(),  # git diff --cached --quiet → non-zero = staged
                _ok(),  # git commit -m
                _ok(stdout="abc123def\n"),  # git rev-parse HEAD
            ],
        )
        result = hc.apply(CommitTaskComplete(summary="Add feature"))
        assert result == CommitSuccess(sha="abc123def")
        # Verify commit message was passed correctly
        commit_call = runner.calls[2]
        assert commit_call[0] == ["git", "commit", "-m", "Add feature"]

    def test_commit_in_progress_success(self, tmp_path: Path) -> None:
        hc, _ = _committer(
            tmp_path,
            [
                _ok(),  # git add -u
                _fail(),  # diff --cached → staged
                _ok(),  # git commit
                _ok(stdout="def456\n"),  # rev-parse
            ],
        )
        result = hc.apply(CommitTaskInProgress(summary="wip: part 1"))
        assert result == CommitSuccess(sha="def456")


# ---------------------------------------------------------------------------
# Tests: hook failure → CommitHookFailure
# ---------------------------------------------------------------------------


class TestApplyHookFailure:
    def test_hook_failure_captures_output(self, tmp_path: Path) -> None:
        hc, _ = _committer(
            tmp_path,
            [
                _ok(),  # git add -u
                _fail(),  # diff --cached → staged
                _fail(
                    returncode=1,
                    stdout="ruff check failed\n",
                    stderr="error: hook returned non-zero\n",
                ),  # git commit fails
            ],
        )
        result = hc.apply(CommitTaskComplete(summary="Add feature"))
        assert isinstance(result, CommitHookFailure)
        assert "ruff check failed" in result.output
        assert "hook returned non-zero" in result.output

    def test_hook_failure_strips_whitespace(self, tmp_path: Path) -> None:
        hc, _ = _committer(
            tmp_path,
            [
                _ok(),
                _fail(),
                _fail(stdout="", stderr="  oops  "),
            ],
        )
        result = hc.apply(CommitTaskComplete(summary="x"))
        assert isinstance(result, CommitHookFailure)
        assert result.output == "oops"

    def test_hook_failure_empty_output(self, tmp_path: Path) -> None:
        hc, _ = _committer(
            tmp_path,
            [
                _ok(),
                _fail(),
                _fail(stdout="", stderr=""),
            ],
        )
        result = hc.apply(CommitTaskComplete(summary="x"))
        assert isinstance(result, CommitHookFailure)
        assert result.output == ""


# ---------------------------------------------------------------------------
# Tests: committer attribution
# ---------------------------------------------------------------------------


class TestCommitterAttribution:
    def test_committer_env_set_on_commit(self, tmp_path: Path) -> None:
        hc, runner = _committer(
            tmp_path,
            [
                _ok(),  # git add -u
                _fail(),  # diff --cached → staged
                _ok(),  # git commit
                _ok(stdout="sha1\n"),  # rev-parse
            ],
        )
        identity = GitIdentity(name="Alice", email="alice@example.com")
        with patch.dict("os.environ", {"PATH": "/usr/bin", "HOME": "/home/test"}):
            result = hc.apply(
                CommitTaskComplete(summary="Fix review feedback"),
                committer=identity,
            )
        assert result == CommitSuccess(sha="sha1")
        # The commit call (index 2) should have env with committer vars
        commit_kwargs = runner.calls[2][1]
        env = commit_kwargs["env"]
        assert env["GIT_COMMITTER_NAME"] == "Alice"
        assert env["GIT_COMMITTER_EMAIL"] == "alice@example.com"
        # Should also have inherited env
        assert env["PATH"] == "/usr/bin"

    def test_no_committer_no_env_override(self, tmp_path: Path) -> None:
        hc, runner = _committer(
            tmp_path,
            [
                _ok(),  # git add -u
                _fail(),  # diff --cached → staged
                _ok(),  # git commit
                _ok(stdout="sha2\n"),  # rev-parse
            ],
        )
        hc.apply(CommitTaskComplete(summary="Regular commit"))
        # The commit call should NOT have an 'env' key in kwargs
        commit_kwargs = runner.calls[2][1]
        assert "env" not in commit_kwargs

    def test_committer_on_hook_failure(self, tmp_path: Path) -> None:
        """Committer env is passed even when the commit fails."""
        hc, runner = _committer(
            tmp_path,
            [
                _ok(),
                _fail(),
                _fail(stdout="hook fail"),
            ],
        )
        identity = GitIdentity(name="Bob", email="bob@x.com")
        with patch.dict("os.environ", {"PATH": "/bin"}):
            result = hc.apply(
                CommitTaskInProgress(summary="wip"),
                committer=identity,
            )
        assert isinstance(result, CommitHookFailure)
        commit_kwargs = runner.calls[2][1]
        assert commit_kwargs["env"]["GIT_COMMITTER_NAME"] == "Bob"


# ---------------------------------------------------------------------------
# Tests: git command plumbing (verified through apply)
# ---------------------------------------------------------------------------


class TestGitPlumbing:
    def test_cwd_is_work_dir(self, tmp_path: Path) -> None:
        """apply passes work_dir as cwd to every git command."""
        hc, runner = _committer(tmp_path, [_ok(), _ok()])
        hc.apply(CommitTaskComplete(summary="x"))
        for _cmd, kwargs in runner.calls:
            assert kwargs["cwd"] == tmp_path

    def test_capture_output_and_text(self, tmp_path: Path) -> None:
        """apply passes capture_output=True and text=True."""
        hc, runner = _committer(tmp_path, [_ok(), _ok()])
        hc.apply(CommitTaskComplete(summary="x"))
        for _, kwargs in runner.calls:
            assert kwargs["capture_output"] is True
            assert kwargs["text"] is True

    def test_add_uses_check_true(self, tmp_path: Path) -> None:
        """git add -u should use check=True (fail-fast on errors)."""
        hc, runner = _committer(tmp_path, [_ok(), _ok()])
        hc.apply(CommitTaskComplete(summary="x"))
        add_kwargs = runner.calls[0][1]
        assert add_kwargs["check"] is True

    def test_diff_cached_uses_check_false(self, tmp_path: Path) -> None:
        """git diff --cached --quiet uses check=False (non-zero is normal)."""
        hc, runner = _committer(tmp_path, [_ok(), _ok()])
        hc.apply(CommitTaskComplete(summary="x"))
        diff_kwargs = runner.calls[1][1]
        assert diff_kwargs["check"] is False

    def test_commit_uses_check_false(self, tmp_path: Path) -> None:
        """git commit uses check=False so hook failures don't raise."""
        hc, runner = _committer(
            tmp_path,
            [_ok(), _fail(), _ok(), _ok(stdout="sha\n")],
        )
        hc.apply(CommitTaskComplete(summary="x"))
        commit_kwargs = runner.calls[2][1]
        assert commit_kwargs["check"] is False


# ---------------------------------------------------------------------------
# Tests: hook_failure_nudge
# ---------------------------------------------------------------------------


class TestHookFailureNudge:
    def test_contains_output(self) -> None:
        failure = CommitHookFailure(output="ruff found 3 errors")
        nudge = hook_failure_nudge(failure)
        assert "ruff found 3 errors" in nudge
        assert "turn_outcome" in nudge

    def test_contains_instruction(self) -> None:
        nudge = hook_failure_nudge(CommitHookFailure(output="x"))
        assert "pre-commit hook rejected" in nudge.lower()
        assert "fix" in nudge.lower()


# ---------------------------------------------------------------------------
# Tests: MAX_HOOK_RETRIES constant
# ---------------------------------------------------------------------------


class TestMaxHookRetries:
    def test_is_positive_int(self) -> None:
        assert isinstance(MAX_HOOK_RETRIES, int)
        assert MAX_HOOK_RETRIES > 0


# ---------------------------------------------------------------------------
# Tests: CommitResult union exhaustiveness
# ---------------------------------------------------------------------------


class TestCommitResultTypes:
    """Verify the union type covers all result dataclasses."""

    def test_all_result_types_in_union(self) -> None:
        # CommitResult is a type alias; verify each concrete type is assignable.
        results: list[CommitResult] = [
            CommitSuccess(sha="abc"),
            CommitHookFailure(output="err"),
            CommitNothingStaged(),
            CommitSkipped(reason="no-op"),
        ]
        assert len(results) == 4
