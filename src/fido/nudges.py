"""Nudge prompt builders for the sentinel loop.

When the LLM does not emit a valid turn_outcome or the harness cannot commit,
the worker writes a nudge prompt that re-enters the session with targeted
instructions.  All three nudge kinds share a common context header (task
title, task id, work dir, PR number).  This module owns that composition.

NudgeKind dispatch uses the Rocq-extracted constructors from
:mod:`fido.rocq.nudge_kind`.
"""

from fido.rocq import nudge_kind as _nudge_oracle
from fido.rocq.commit_result import CommitHookFailure
from fido.rocq.nudge_kind import NudgeKindT


class Nudges:
    """Builds nudge prompts for the sentinel loop.

    Owns the shared context-header composition so callers never re-prepend it
    themselves.  Dispatches on the Rocq-extracted
    :data:`~fido.rocq.nudge_kind.NudgeKindT` constructors via
    :meth:`for_kind`:

    - :meth:`missing_sentinel` → :class:`~fido.rocq.nudge_kind.NudgeMissingSentinel`
    - :meth:`nothing_staged`  → :class:`~fido.rocq.nudge_kind.NudgeNothingStaged`
    - :meth:`hook_failure`    → :class:`~fido.rocq.nudge_kind.NudgeHookFailure`

    No mutable state — all methods are pure text transformations.  Accepted
    via constructor-DI on :class:`~fido.worker.Worker` so tests can substitute
    a fake without patching module-level names.
    """

    def _context_header(
        self,
        task_title: str,
        task_id: str,
        work_dir: str,
        pr_number: int,
    ) -> str:
        """Return the task-context block that opens every nudge prompt."""
        return (
            f"Task: {task_title} (id: {task_id})\nPR: {pr_number}\nWork dir: {work_dir}"
        )

    def for_kind(
        self,
        kind: NudgeKindT,
        task_title: str,
        task_id: str,
        work_dir: str,
        pr_number: int,
        *,
        parse_error: str = "",
        failure: CommitHookFailure | None = None,
    ) -> str:
        """Dispatch on *kind* and return the appropriate nudge prompt.

        Extra keyword args are required only for the matching kind:

        - :class:`~fido.rocq.nudge_kind.NudgeMissingSentinel` — *parse_error*
        - :class:`~fido.rocq.nudge_kind.NudgeNothingStaged` — (none)
        - :class:`~fido.rocq.nudge_kind.NudgeHookFailure` — *failure*
        """
        match kind:
            case _nudge_oracle.NudgeMissingSentinel():
                return self.missing_sentinel(
                    task_title, task_id, work_dir, pr_number, parse_error
                )
            case _nudge_oracle.NudgeNothingStaged():
                return self.nothing_staged(task_title, task_id, work_dir, pr_number)
            case _nudge_oracle.NudgeHookFailure():
                assert failure is not None, "hook_failure nudge requires a failure arg"
                return self.hook_failure(
                    task_title, task_id, work_dir, pr_number, failure
                )

    def missing_sentinel(
        self,
        task_title: str,
        task_id: str,
        work_dir: str,
        pr_number: int,
        parse_error: str,
    ) -> str:
        """Return a nudge prompt when the LLM output has no valid turn_outcome.

        Corresponds to :class:`~fido.rocq.nudge_kind.NudgeMissingSentinel`.
        The *parse_error* explains why the sentinel could not be parsed so the
        LLM knows exactly what it did wrong and can fix it.
        """
        header = self._context_header(task_title, task_id, work_dir, pr_number)
        return (
            f"{header}\n\n"
            f"Parse error: {parse_error}\n\n"
            "Every turn must end with exactly one JSON object on the final "
            "non-empty line.  If the task is complete, emit "
            '{"turn_outcome":"commit-task-complete","summary":"<commit message>"}. '
            "If you need another turn, emit "
            '{"turn_outcome":"commit-task-in-progress","summary":"<wip message>"}. '
            "If no code change is needed, emit "
            '{"turn_outcome":"skip-task-with-reason","reason":"<explanation>"}. '
            "If you are blocked and need human guidance, emit "
            '{"turn_outcome":"stuck-on-task","reason":"<what you need>"}.'
        )

    def nothing_staged(
        self,
        task_title: str,
        task_id: str,
        work_dir: str,
        pr_number: int,
    ) -> str:
        """Return a nudge prompt when a commit sentinel was emitted but nothing staged.

        Corresponds to :class:`~fido.rocq.nudge_kind.NudgeNothingStaged`.
        The harness ran ``git add -u`` after the sentinel but found nothing to
        commit — the working tree has no tracked changes.
        """
        header = self._context_header(task_title, task_id, work_dir, pr_number)
        return (
            f"{header}\n\n"
            "Your turn_outcome sentinel declared a commit, but ``git add -u`` "
            "found nothing staged — the working tree has no tracked changes.\n\n"
            "Either make the necessary file changes and emit a commit sentinel, "
            "or emit "
            '{"turn_outcome":"skip-task-with-reason","reason":"<explanation>"} '
            "if no code change is actually needed."
        )

    def hook_failure(
        self,
        task_title: str,
        task_id: str,
        work_dir: str,
        pr_number: int,
        failure: CommitHookFailure,
    ) -> str:
        """Return a nudge prompt after a pre-commit hook rejects the commit.

        Corresponds to :class:`~fido.rocq.nudge_kind.NudgeHookFailure`.
        Composes the context header with the hook-failure body so callers do
        not re-prepend the header themselves.
        """
        header = self._context_header(task_title, task_id, work_dir, pr_number)
        body = (
            "The pre-commit hook rejected the commit. Fix the issues below "
            "and emit a new turn_outcome sentinel.\n\n"
            f"Hook output:\n{failure.output}"
        )
        return f"{header}\n\n{body}"


__all__ = ["Nudges"]
