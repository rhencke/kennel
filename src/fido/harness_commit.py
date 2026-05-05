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
from collections.abc import Sequence
from pathlib import Path

from fido.infra import ProcessRunner
from fido.rocq import harness_commit_decision as _hcd_mod
from fido.rocq import turn_outcome as _to_mod
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
        self, summary: str, *, helped_by: Sequence[GitIdentity] = ()
    ) -> str:
        """Build the full commit message with optional Helped-by trailers."""
        if not helped_by:
            return summary
        trailers = "\n".join(
            f"Helped-by: {identity.name} <{identity.email}>" for identity in helped_by
        )
        return f"{summary}\n\n{trailers}".rstrip()

    def _assert_decision_oracle(
        self,
        outcome: TurnOutcome,
        result: CommitResult,
        *,
        has_staged: bool = False,
        commit_ok: bool = False,
        commit_sha: str = "",
        commit_output: str = "",
    ) -> None:
        """Assert that the Rocq-proven harness_commit_decision agrees."""
        env = _hcd_mod.MkGitEnv(
            has_staged=has_staged,
            commit_ok=commit_ok,
            commit_sha=commit_sha,
            commit_output=commit_output,
        )
        oracle = _hcd_mod.harness_commit_decision(outcome, env)
        if result != oracle:
            raise AssertionError(
                f"harness_commit_decision oracle mismatch: "
                f"oracle={oracle!r}, actual={result!r}"
            )

    def _assert_commit_dispatch_oracle(
        self, outcome: TurnOutcome, *, dispatched_to_commit: bool
    ) -> None:
        """Assert that the Rocq-proven outcome_is_commit agrees with our dispatch."""
        oracle = _to_mod.outcome_is_commit(outcome)
        if oracle != dispatched_to_commit:
            raise AssertionError(
                f"outcome_is_commit oracle mismatch: "
                f"oracle={oracle!r}, dispatched_to_commit={dispatched_to_commit!r}"
            )

    def hook_failure_nudge(self, failure: CommitHookFailure) -> str:
        """Format a nudge prompt for the LLM after a hook failure."""
        return (
            "The pre-commit hook rejected the commit. Fix the issues below "
            "and emit a new turn_outcome sentinel.\n\n"
            f"Hook output:\n{failure.output}"
        )

    def commit(
        self,
        outcome: TurnOutcome,
        *,
        helped_by: Sequence[GitIdentity] = (),
    ) -> CommitResult:
        """Stage tracked changes and commit based on *outcome*.

        For :class:`~fido.turn_outcome.SkipTaskWithReason`, returns
        :class:`CommitSkipped` immediately — no git operations are run.

        For :class:`~fido.turn_outcome.CommitTaskComplete` and
        :class:`~fido.turn_outcome.CommitTaskInProgress`, runs
        ``git add -u`` to stage tracked files, then ``git commit -m``.

        Returns :class:`CommitHookFailure` if the pre-commit hook (or
        ``git commit`` itself) exits non-zero.

        *helped_by* appends one ``Helped-by: Name <email>`` trailer per
        identity to the commit message for thread tasks — reviewers who
        requested the change get attribution in the git log.
        """
        match outcome:
            case SkipTaskWithReason(reason=reason):
                self._assert_commit_dispatch_oracle(outcome, dispatched_to_commit=False)
                result = CommitSkipped(reason=reason)
                self._assert_decision_oracle(outcome, result)
                return result
            case StuckOnTask(reason=reason):
                self._assert_commit_dispatch_oracle(outcome, dispatched_to_commit=False)
                result = CommitSkipped(reason=reason)
                self._assert_decision_oracle(outcome, result)
                return result
            case (
                CommitTaskComplete(summary=summary)
                | CommitTaskInProgress(summary=summary)
            ):
                self._assert_commit_dispatch_oracle(outcome, dispatched_to_commit=True)
                return self._attempt_commit(outcome, summary, helped_by=helped_by)
            case _:  # pragma: no cover — unreachable; exhaustive above
                raise ValueError(f"unexpected TurnOutcome variant: {outcome!r}")

    def _attempt_commit(
        self,
        outcome: TurnOutcome,
        summary: str,
        *,
        helped_by: Sequence[GitIdentity] = (),
    ) -> CommitResult:
        """Stage tracked files and commit.  Pure I/O — the decision to call
        this has already been made by the match dispatch above."""
        # Stage tracked files only — never sweep untracked files like
        # .coverage artifacts, editor scratch files, or build outputs.
        try:
            self._git(["add", "-u"])
        except subprocess.CalledProcessError as exc:
            output = f"git add -u failed (exit {exc.returncode})"
            if exc.stderr:
                output += f"\n{exc.stderr.strip()}"
            elif exc.stdout:
                output += f"\n{exc.stdout.strip()}"
            log.warning("git add -u failed: %s", output[:200])
            result = CommitHookFailure(output=output)
            self._assert_decision_oracle(
                outcome, result, has_staged=True, commit_ok=False, commit_output=output
            )
            return result

        # If nothing was staged, report it rather than creating an empty commit.
        diff_cached = self._git(["diff", "--cached", "--quiet"], check=False)
        if diff_cached.returncode == 0:
            result = CommitNothingStaged()
            self._assert_decision_oracle(outcome, result, has_staged=False)
            return result

        message = self._build_message(summary, helped_by=helped_by)
        git_result = self._git(["commit", "-m", message], check=False)
        if git_result.returncode != 0:
            output = (git_result.stdout + "\n" + git_result.stderr).strip()
            log.warning("commit rejected (hook or error): %s", output[:200])
            result = CommitHookFailure(output=output)
            self._assert_decision_oracle(
                outcome, result, has_staged=True, commit_ok=False, commit_output=output
            )
            return result

        sha = self._git(["rev-parse", "HEAD"]).stdout.strip()
        log.info("harness commit: %s (%s)", sha[:12], summary[:60])
        result = CommitSuccess(sha=sha)
        self._assert_decision_oracle(
            outcome, result, has_staged=True, commit_ok=True, commit_sha=sha
        )
        return result
