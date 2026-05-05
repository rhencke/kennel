"""Harness-owned commit logic for the turn-outcome protocol.

The LLM declares intent via a ``turn_outcome`` sentinel on its final
non-empty output line.  This module stages tracked changes and commits
on the LLM's behalf.  Pre-commit hook failures are captured and returned
as structured results so the caller can nudge the LLM to fix them.

The LLM declares intent; Python acts on it.  Git operations are never
the LLM's responsibility.

Type definitions live in Rocq-extracted modules — importers should get
``CommitResult`` types from :mod:`fido.rocq.commit_result` and
``TurnOutcome`` types from :mod:`fido.rocq.turn_outcome` directly.
"""

import logging
import subprocess
from pathlib import Path

from fido.infra import ProcessRunner
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
    TurnOutcome,
)
from fido.types import GitIdentity

log = logging.getLogger(__name__)

__all__ = [
    "HarnessCommitter",
]


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
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command in the workspace."""
        return self._runner.run(
            ["git", *args],
            check=check,
            cwd=self._work_dir,
            capture_output=True,
            text=True,
        )

    def _build_message(
        self, summary: str, *, helped_by: GitIdentity | None = None
    ) -> str:
        """Build the full commit message with optional Helped-by trailer."""
        if helped_by is None:
            return summary
        trailer = f"Helped-by: {helped_by.name} <{helped_by.email}>"
        return f"{summary}\n\n{trailer}"

    def hook_failure_nudge(self, failure: CommitHookFailure) -> str:
        """Format a nudge prompt for the LLM after a hook failure."""
        return (
            "The pre-commit hook rejected the commit. Fix the issues below "
            "and emit a new turn_outcome sentinel.\n\n"
            f"Hook output:\n{failure.output}"
        )

    def apply(
        self,
        outcome: TurnOutcome,
        *,
        helped_by: GitIdentity | None = None,
    ) -> CommitResult:
        """Stage tracked changes and commit based on *outcome*.

        For :class:`~fido.turn_outcome.SkipTaskWithReason`, returns
        :class:`CommitSkipped` immediately — no git operations are run.

        For :class:`~fido.turn_outcome.CommitTaskComplete` and
        :class:`~fido.turn_outcome.CommitTaskInProgress`, runs
        ``git add -u`` to stage tracked files, then ``git commit -m``.

        Returns :class:`CommitHookFailure` if the pre-commit hook (or
        ``git commit`` itself) exits non-zero.

        *helped_by* appends a ``Helped-by: Name <email>`` trailer to the
        commit message for thread tasks — the reviewer who requested the
        change gets attribution in the git log.
        """
        if isinstance(outcome, SkipTaskWithReason):
            return CommitSkipped(reason=outcome.reason)

        assert isinstance(outcome, CommitTaskComplete | CommitTaskInProgress)

        # Stage tracked files only — never sweep untracked files like
        # .coverage artifacts, editor scratch files, or build outputs.
        self._git(["add", "-u"])

        # If nothing was staged, report it rather than creating an empty commit.
        diff_cached = self._git(["diff", "--cached", "--quiet"], check=False)
        if diff_cached.returncode == 0:
            return CommitNothingStaged()

        message = self._build_message(outcome.summary, helped_by=helped_by)
        result = self._git(["commit", "-m", message], check=False)
        if result.returncode != 0:
            output = (result.stdout + "\n" + result.stderr).strip()
            log.warning("commit rejected (hook or error): %s", output[:200])
            return CommitHookFailure(output=output)

        sha = self._git(["rev-parse", "HEAD"]).stdout.strip()
        log.info("harness commit: %s (%s)", sha[:12], outcome.summary[:60])
        return CommitSuccess(sha=sha)
