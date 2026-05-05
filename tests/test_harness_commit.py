"""Tests for fido.harness_commit — harness-owned commit logic."""

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fido.harness_commit import HarnessCommitter
from fido.rocq import harness_commit_decision as _hcd_mod
from fido.rocq.commit_result import (
    CommitHookFailure,
    CommitNothingStaged,
    CommitResult,
    CommitSkipped,
    CommitSuccess,
)
from fido.rocq.turn_outcome import (
    CommitTaskComplete,
    CommitTaskInProgress,
    SkipTaskWithReason,
    StuckOnTask,
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


class TestCommitSkip:
    def test_returns_commit_skipped(self, tmp_path: Path) -> None:
        hc, runner = _committer(tmp_path, [])
        result = hc.commit(SkipTaskWithReason(reason="already done"))
        assert result == CommitSkipped(reason="already done")
        assert runner.calls == []

    def test_skip_ignores_committer(self, tmp_path: Path) -> None:
        """Committer arg is irrelevant for skip outcomes."""
        hc, runner = _committer(tmp_path, [])
        result = hc.commit(
            SkipTaskWithReason(reason="nope"),
            helped_by=[GitIdentity(name="X", email="x@y")],
        )
        assert isinstance(result, CommitSkipped)
        assert runner.calls == []

    def test_stuck_on_task_returns_commit_skipped(self, tmp_path: Path) -> None:
        """StuckOnTask is treated like SkipTaskWithReason in harness_commit."""
        hc, runner = _committer(tmp_path, [])
        result = hc.commit(StuckOnTask(reason="need human input"))
        assert result == CommitSkipped(reason="need human input")
        assert runner.calls == []


# ---------------------------------------------------------------------------
# Tests: nothing staged → CommitNothingStaged
# ---------------------------------------------------------------------------


class TestCommitNothingStaged:
    def test_commit_complete_nothing_staged(self, tmp_path: Path) -> None:
        hc, runner = _committer(
            tmp_path,
            [
                _ok(),  # git add -u
                _ok(),  # git diff --cached --quiet → returncode 0 = clean
            ],
        )
        result = hc.commit(CommitTaskComplete(summary="Add feature"))
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
        result = hc.commit(CommitTaskInProgress(summary="wip"))
        assert result == CommitNothingStaged()


# ---------------------------------------------------------------------------
# Tests: successful commit → CommitSuccess
# ---------------------------------------------------------------------------


class TestCommitSuccess:
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
        result = hc.commit(CommitTaskComplete(summary="Add feature"))
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
        result = hc.commit(CommitTaskInProgress(summary="wip: part 1"))
        assert result == CommitSuccess(sha="def456")


# ---------------------------------------------------------------------------
# Tests: hook failure → CommitHookFailure
# ---------------------------------------------------------------------------


class TestCommitHookFailure:
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
        result = hc.commit(CommitTaskComplete(summary="Add feature"))
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
        result = hc.commit(CommitTaskComplete(summary="x"))
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
        result = hc.commit(CommitTaskComplete(summary="x"))
        assert isinstance(result, CommitHookFailure)
        assert result.output == ""


# ---------------------------------------------------------------------------
# Tests: committer attribution
# ---------------------------------------------------------------------------


class TestHelpedByAttribution:
    def test_helped_by_trailer_in_commit_message(self, tmp_path: Path) -> None:
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
        result = hc.commit(
            CommitTaskComplete(summary="Fix review feedback"),
            helped_by=[identity],
        )
        assert result == CommitSuccess(sha="sha1")
        # The commit message should include the Helped-by trailer
        commit_cmd = runner.calls[2][0]
        assert commit_cmd == [
            "git",
            "commit",
            "-m",
            "Fix review feedback\n\nHelped-by: Alice <alice@example.com>",
        ]

    def test_multiple_helped_by_trailers(self, tmp_path: Path) -> None:
        hc, runner = _committer(
            tmp_path,
            [
                _ok(),  # git add -u
                _fail(),  # diff --cached → staged
                _ok(),  # git commit
                _ok(stdout="sha3\n"),  # rev-parse
            ],
        )
        identities = [
            GitIdentity(name="Alice", email="alice@example.com"),
            GitIdentity(name="Bob", email="bob@example.com"),
        ]
        result = hc.commit(
            CommitTaskComplete(summary="Address review feedback"),
            helped_by=identities,
        )
        assert result == CommitSuccess(sha="sha3")
        commit_cmd = runner.calls[2][0]
        assert commit_cmd == [
            "git",
            "commit",
            "-m",
            "Address review feedback\n\n"
            "Helped-by: Alice <alice@example.com>\n"
            "Helped-by: Bob <bob@example.com>",
        ]

    def test_no_helped_by_plain_message(self, tmp_path: Path) -> None:
        hc, runner = _committer(
            tmp_path,
            [
                _ok(),  # git add -u
                _fail(),  # diff --cached → staged
                _ok(),  # git commit
                _ok(stdout="sha2\n"),  # rev-parse
            ],
        )
        hc.commit(CommitTaskComplete(summary="Regular commit"))
        # The commit message should just be the summary
        commit_cmd = runner.calls[2][0]
        assert commit_cmd == ["git", "commit", "-m", "Regular commit"]

    def test_helped_by_on_hook_failure(self, tmp_path: Path) -> None:
        """Helped-by trailer is in the message even when the commit fails."""
        hc, runner = _committer(
            tmp_path,
            [
                _ok(),
                _fail(),
                _fail(stdout="hook fail"),
            ],
        )
        identity = GitIdentity(name="Bob", email="bob@x.com")
        result = hc.commit(
            CommitTaskInProgress(summary="wip"),
            helped_by=[identity],
        )
        assert isinstance(result, CommitHookFailure)
        commit_cmd = runner.calls[2][0]
        assert commit_cmd == [
            "git",
            "commit",
            "-m",
            "wip\n\nHelped-by: Bob <bob@x.com>",
        ]


# ---------------------------------------------------------------------------
# Tests: git command plumbing (verified through commit)
# ---------------------------------------------------------------------------


class TestGitPlumbing:
    def test_cwd_is_work_dir(self, tmp_path: Path) -> None:
        """commit passes work_dir as cwd to every git command."""
        hc, runner = _committer(tmp_path, [_ok(), _ok()])
        hc.commit(CommitTaskComplete(summary="x"))
        for _cmd, kwargs in runner.calls:
            assert kwargs["cwd"] == tmp_path

    def test_capture_output_and_text(self, tmp_path: Path) -> None:
        """commit passes capture_output=True and text=True."""
        hc, runner = _committer(tmp_path, [_ok(), _ok()])
        hc.commit(CommitTaskComplete(summary="x"))
        for _, kwargs in runner.calls:
            assert kwargs["capture_output"] is True
            assert kwargs["text"] is True

    def test_add_uses_check_true(self, tmp_path: Path) -> None:
        """git add -u should use check=True (fail-fast on errors)."""
        hc, runner = _committer(tmp_path, [_ok(), _ok()])
        hc.commit(CommitTaskComplete(summary="x"))
        add_kwargs = runner.calls[0][1]
        assert add_kwargs["check"] is True

    def test_diff_cached_uses_check_false(self, tmp_path: Path) -> None:
        """git diff --cached --quiet uses check=False (non-zero is normal)."""
        hc, runner = _committer(tmp_path, [_ok(), _ok()])
        hc.commit(CommitTaskComplete(summary="x"))
        diff_kwargs = runner.calls[1][1]
        assert diff_kwargs["check"] is False

    def test_commit_uses_check_false(self, tmp_path: Path) -> None:
        """git commit uses check=False so hook failures don't raise."""
        hc, runner = _committer(
            tmp_path,
            [_ok(), _fail(), _ok(), _ok(stdout="sha\n")],
        )
        hc.commit(CommitTaskComplete(summary="x"))
        commit_kwargs = runner.calls[2][1]
        assert commit_kwargs["check"] is False


# ---------------------------------------------------------------------------
# Tests: git add -u failure handling
# ---------------------------------------------------------------------------


class _RaisingRunner:
    """ProcessRunner fake that raises CalledProcessError on the first call."""

    def __init__(self, stderr: str = "", stdout: str = "") -> None:
        self._stderr = stderr
        self._stdout = stdout

    def run(
        self, cmd: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            128, cmd, output=self._stdout, stderr=self._stderr
        )


class TestGitAddFailure:
    """git add -u CalledProcessError is caught and returned as CommitHookFailure."""

    def test_returns_hook_failure_with_stderr(self, tmp_path: Path) -> None:
        runner = _RaisingRunner(stderr="fatal: unable to read tree")
        hc = HarnessCommitter(tmp_path, runner)
        result = hc.commit(CommitTaskComplete(summary="whatever"))
        assert isinstance(result, CommitHookFailure)
        assert "git add -u failed" in result.output
        assert "fatal: unable to read tree" in result.output

    def test_returns_hook_failure_with_stdout_fallback(self, tmp_path: Path) -> None:
        runner = _RaisingRunner(stdout="error: some output")
        hc = HarnessCommitter(tmp_path, runner)
        result = hc.commit(CommitTaskComplete(summary="whatever"))
        assert isinstance(result, CommitHookFailure)
        assert "git add -u failed" in result.output
        assert "error: some output" in result.output

    def test_returns_hook_failure_with_no_output(self, tmp_path: Path) -> None:
        runner = _RaisingRunner()
        hc = HarnessCommitter(tmp_path, runner)
        result = hc.commit(CommitTaskInProgress(summary="wip"))
        assert isinstance(result, CommitHookFailure)
        assert "git add -u failed (exit 128)" in result.output

    def test_includes_exit_code(self, tmp_path: Path) -> None:
        runner = _RaisingRunner(stderr="bad")
        hc = HarnessCommitter(tmp_path, runner)
        result = hc.commit(CommitTaskComplete(summary="x"))
        assert isinstance(result, CommitHookFailure)
        assert "exit 128" in result.output


# ---------------------------------------------------------------------------
# Tests: hook_failure_nudge
# ---------------------------------------------------------------------------


class TestHookFailureNudge:
    def test_contains_output(self, tmp_path: Path) -> None:
        hc, _ = _committer(tmp_path, [])
        failure = CommitHookFailure(output="ruff found 3 errors")
        nudge = hc.hook_failure_nudge(failure)
        assert "ruff found 3 errors" in nudge
        assert "turn_outcome" in nudge

    def test_contains_instruction(self, tmp_path: Path) -> None:
        hc, _ = _committer(tmp_path, [])
        nudge = hc.hook_failure_nudge(CommitHookFailure(output="x"))
        assert "pre-commit hook rejected" in nudge.lower()
        assert "fix" in nudge.lower()


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


# ---------------------------------------------------------------------------
# Tests: Rocq oracle assertion
# ---------------------------------------------------------------------------


class TestDecisionOracle:
    """Verify the oracle assertion fires on mismatch."""

    def test_oracle_mismatch_raises(self, tmp_path: Path) -> None:
        """If harness_commit_decision returns a different result, AssertionError fires."""
        hc, _ = _committer(tmp_path, [])

        # Monkeypatch the oracle to return the wrong thing
        def bad_oracle(_o: object, _env: object) -> object:
            return _hcd_mod.CommitSuccess("wrong_sha")

        with patch.object(_hcd_mod, "harness_commit_decision", bad_oracle):
            with pytest.raises(AssertionError, match="oracle mismatch"):
                hc.commit(SkipTaskWithReason(reason="done"))
