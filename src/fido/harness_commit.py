"""Harness-owned commit logic for the turn-outcome protocol.

The LLM declares intent via a ``turn_outcome`` sentinel on its final
non-empty output line.  This module stages tracked changes and commits
on the LLM's behalf.  Pre-commit hook failures are captured and returned
as structured results so the caller can nudge the LLM to fix them.

The LLM declares intent; Python acts on it.  Git operations are never
the LLM's responsibility.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fido.infra import ProcessRunner
from fido.turn_outcome import (
    CommitTaskComplete,
    CommitTaskInProgress,
    SkipTaskWithReason,
    TurnOutcome,
)
from fido.types import GitIdentity

log = logging.getLogger(__name__)

# Maximum pre-commit hook retry attempts before the caller should give up
# and treat the task as stuck.  Exported so execute_task can reference it.
MAX_HOOK_RETRIES = 3


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitSuccess:
    """Commit was created successfully.

    *sha* is the full SHA of the new commit.
    """

    sha: str


@dataclass(frozen=True)
class CommitHookFailure:
    """Pre-commit hook (or commit itself) rejected the commit.

    *output* contains the combined stdout + stderr from the failed
    ``git commit`` so the caller can feed it back to the LLM.
    """

    output: str


@dataclass(frozen=True)
class CommitNothingStaged:
    """``git add -u`` staged nothing — worktree is clean or only has untracked files."""


@dataclass(frozen=True)
class CommitSkipped:
    """Turn outcome was ``skip-task-with-reason`` — no commit needed."""

    reason: str


CommitResult = CommitSuccess | CommitHookFailure | CommitNothingStaged | CommitSkipped


# ---------------------------------------------------------------------------
# Nudge prompt helper
# ---------------------------------------------------------------------------


def hook_failure_nudge(failure: CommitHookFailure) -> str:
    """Format a nudge prompt for the LLM after a hook failure.

    Pure formatting helper — no collaborators, no side effects.
    """
    return (
        "The pre-commit hook rejected the commit. Fix the issues below "
        "and emit a new turn_outcome sentinel.\n\n"
        f"Hook output:\n{failure.output}"
    )


# ---------------------------------------------------------------------------
# Committer
# ---------------------------------------------------------------------------


class HarnessCommitter:
    """Stages tracked changes and commits based on a parsed TurnOutcome.

    Dependencies are injected through the constructor so tests can supply
    a fake :class:`~fido.infra.ProcessRunner` without patching
    :mod:`subprocess`.
    """

    def __init__(self, work_dir: Path, runner: ProcessRunner) -> None:
        self._work_dir = work_dir
        self._runner = runner

    def _git(
        self,
        args: list[str],
        check: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command in the workspace.

        *extra_env* is merged on top of the current environment so ``git``
        still has access to ``PATH``, ``HOME``, etc.  When *extra_env* is
        ``None``, the subprocess inherits the parent environment as usual.
        """
        kwargs: dict[str, object] = {
            "capture_output": True,
            "text": True,
        }
        if extra_env is not None:
            kwargs["env"] = {**os.environ, **extra_env}
        return self._runner.run(
            ["git", *args], check=check, cwd=self._work_dir, **kwargs
        )

    def apply(
        self,
        outcome: TurnOutcome,
        *,
        committer: GitIdentity | None = None,
    ) -> CommitResult:
        """Stage tracked changes and commit based on *outcome*.

        For :class:`~fido.turn_outcome.SkipTaskWithReason`, returns
        :class:`CommitSkipped` immediately — no git operations are run.

        For :class:`~fido.turn_outcome.CommitTaskComplete` and
        :class:`~fido.turn_outcome.CommitTaskInProgress`, runs
        ``git add -u`` to stage tracked files, then ``git commit -m``.

        Returns :class:`CommitHookFailure` if the pre-commit hook (or
        ``git commit`` itself) exits non-zero.

        *committer* overrides ``GIT_COMMITTER_NAME`` / ``GIT_COMMITTER_EMAIL``
        for attribution on thread tasks — the reviewer who requested the
        change gets committer credit.
        """
        if isinstance(outcome, SkipTaskWithReason):
            return CommitSkipped(reason=outcome.reason)

        assert isinstance(outcome, CommitTaskComplete | CommitTaskInProgress)
        summary = outcome.summary

        # Stage tracked files only — never sweep untracked files like
        # .coverage artifacts, editor scratch files, or build outputs.
        self._git(["add", "-u"])

        # If nothing was staged, report it rather than creating an empty commit.
        diff_cached = self._git(["diff", "--cached", "--quiet"], check=False)
        if diff_cached.returncode == 0:
            return CommitNothingStaged()

        # Build committer attribution env when the task came from a thread.
        extra_env: dict[str, str] | None = None
        if committer is not None:
            extra_env = {
                "GIT_COMMITTER_NAME": committer.name,
                "GIT_COMMITTER_EMAIL": committer.email,
            }

        result = self._git(
            ["commit", "-m", summary],
            check=False,
            extra_env=extra_env,
        )
        if result.returncode != 0:
            output = (result.stdout + "\n" + result.stderr).strip()
            log.warning("commit rejected (hook or error): %s", output[:200])
            return CommitHookFailure(output=output)

        sha = self._git(["rev-parse", "HEAD"]).stdout.strip()
        log.info("harness commit: %s (%s)", sha[:12], summary[:60])
        return CommitSuccess(sha=sha)
