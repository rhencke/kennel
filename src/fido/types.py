"""Shared type definitions for fido."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal


class TaskType(StrEnum):
    CI = "ci"
    THREAD = "thread"
    SPEC = "spec"


class TaskStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ActiveIssue:
    """Snapshot of the GitHub issue being worked on.

    Injected into the system prompt at every LLM call site so the agent
    always has the spec in front of it, regardless of which turn it is on.
    """

    number: int
    title: str
    body: str


@dataclass(frozen=True)
class ActivePR:
    """Snapshot of the pull request associated with the current work session."""

    number: int
    title: str
    url: str
    body: str


@dataclass(frozen=True)
class ClosedPR:
    """A prior closed (not merged) PR that referenced the same issue.

    Surfaced in the system prompt so the agent can learn from earlier
    attempts and avoid repeating the same mistakes.
    """

    number: int
    title: str
    body: str
    close_reason: str


@dataclass(frozen=True)
class ClosedSubIssue:
    """A closed direct sub-issue of the parent issue being worked on.

    Fetched during setup so the agent can subtract already-covered scope
    from the parent's task list (or declare ``NoTasksNeeded`` when all
    scope is covered).

    Attributes
    ----------
    number:
        GitHub issue number of the sub-issue.
    title:
        Title of the sub-issue.
    body:
        Body text of the sub-issue.
    close_state:
        One of ``"merged"`` (a linked PR was merged), ``"closed_unmerged"``
        (a linked PR was closed without merging), or ``"closed_no_pr"``
        (no linked PR — cancelled, won't-fix, or deferred).
    state_reason:
        GitHub's ``state_reason`` for the sub-issue (``"completed"``,
        ``"not_planned"``, ``"reopened"``, or ``None`` when absent).  Only
        meaningful when ``close_state`` is ``"closed_no_pr"`` — used to
        distinguish "completed without a PR" from "won't fix / deferred".
    pr_number:
        GitHub PR number of the linked pull request, or ``None`` when
        ``close_state`` is ``"closed_no_pr"``.
    pr_repo:
        ``"owner/repo"`` of the linked pull request, or ``None`` when
        ``close_state`` is ``"closed_no_pr"``.  May differ from the parent
        issue's repo for cross-repo sub-issue PRs.
    pr_body:
        Body text of the linked PR, or ``""`` when no PR exists.
    """

    number: int
    title: str
    body: str
    close_state: str
    state_reason: str | None = None
    pr_number: int | None = None
    pr_repo: str | None = None
    pr_body: str = ""


@dataclass(frozen=True)
class TaskSnapshot:
    """Projection of a task dict for LLM context rendering.

    Captures only the fields needed by :func:`~fido.prompts.render_active_context`
    so the renderer does not depend on the full task dict shape from
    ``tasks.json``.
    """

    title: str
    type: str
    status: str
    description: str = ""


IntentOutcome = Literal["honored", "reshaped", "superseded", "no_op"]
"""Per-intent verdict outcome (per #1798 / INV-D).

  * ``honored`` — Opus produced ops that fulfill the intent as asked.
  * ``reshaped`` — ops fulfill *part* of the intent; the original
    framing changed (e.g. user asked for a feature, Opus split it
    into prereq + feature).  Material change — INV-E (#1803) emits a
    reply-back unless self-supersedence applies.
  * ``superseded`` — another intent in the same batch overrode this
    one (in whole or in part).  ``by_intent_comment_id`` names the
    superseding intent.  Material when authors differ; self-
    correction (no reply) when authors match.
  * ``no_op`` — intent was acknowledged but produced no task changes
    (e.g. commenter chatter, ack, or request already satisfied by an
    existing pending task with no edits needed).
"""


@dataclass(frozen=True)
class IntentVerdict:
    """Per-intent verdict in a rescope batch result (#1798 / INV-D).

    Replaces the older ``IntentDisposition`` material/aggregation
    classifier.  Each :class:`RescopeIntent` in a rescope batch maps
    to exactly one :class:`IntentVerdict`; the batch's full verdict
    list is Opus's structured answer for "what did I do with each
    ask?".

    INV-E (#1803) downstream rule: post a reply-back for ``V`` iff
    ``V.outcome`` materially changes the original ask (``reshaped``
    or cross-author ``superseded``) AND not self-supersedence
    (``V.by_intent_comment_id``'s intent has the same author as
    ``V``).  ``honored`` and ``no_op`` never warrant reply-back —
    they're either "got it as asked" or "nothing to do" and the
    triage reply already covered the user-facing acknowledgement.

    Attributes
    ----------
    intent_comment_id:
        :attr:`RescopeIntent.comment_id` this verdict pertains to.
        Each intent in the batch produces exactly one verdict.
    outcome:
        See :data:`IntentOutcome` for the four allowed values and
        their reply-back implications.
    ops:
        Op records this intent contributed to the batch.  May be
        empty when ``outcome`` is ``superseded`` (fully superseded —
        the winning intent owns the resulting ops) or ``no_op``.
        Two verdicts in the same batch MAY both name the same task
        id in :attr:`affected_task_ids` when they're jointly honored
        by the same task (the canonical 3+1 reviewer-pattern case:
        three review comments asking the same fix plus a fourth
        "just fix all of these" all attribute to one consolidated
        task).
    affected_task_ids:
        Task ids this verdict touched, in result-list order.  MAY
        overlap with other verdicts' :attr:`affected_task_ids` under
        joint-honoring.
    by_intent_comment_id:
        When set, names another intent in the same batch that
        (partially or fully) superseded this one.  Must reference a
        :attr:`RescopeIntent.comment_id` from the same batch.
        Required to be acyclic across the batch (INV-F / #1804 will
        prove this in Rocq; runtime asserts here).
    narrative:
        Opus's prose explanation of what happened.  Verbatim source
        for the optional reply-back posted by INV-E (#1803) —
        per the project memory ``voice_text_opus_not_templated``,
        Fido-voice text is per-call Opus prose, not templated.
        Required when ``outcome`` is ``reshaped`` or ``superseded``
        (reply-back needs prose); optional otherwise.
    """

    intent_comment_id: int
    outcome: IntentOutcome
    ops: tuple[Mapping[str, Any], ...] = ()
    """Op records this intent contributed.  Coerced to a tuple of
    :class:`~frozendict.frozendict` mappings in ``__post_init__`` so
    both the outer sequence and each per-op mapping are deeply
    immutable — callers cannot mutate ``verdict.ops`` or
    ``verdict.ops[0]['op']`` after construction (codex P1 / P2 on
    #1802)."""
    affected_task_ids: tuple[str, ...] = ()
    by_intent_comment_id: int | None = None
    narrative: str | None = None

    def __post_init__(self) -> None:
        # Scalar type checks first (codex P2 on #1802: dataclasses
        # don't enforce annotations; a parser typo like
        # ``intent_comment_id="123"`` would silently flow downstream).
        # ``bool`` rejected separately — it's an ``int`` subclass.
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.intent_comment_id, int
        ) or isinstance(self.intent_comment_id, bool):
            raise TypeError(
                "IntentVerdict.intent_comment_id must be int, got "
                f"{type(self.intent_comment_id).__name__}"
            )
        if self.by_intent_comment_id is not None and (
            not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                self.by_intent_comment_id, int
            )
            or isinstance(self.by_intent_comment_id, bool)
        ):
            raise TypeError(
                "IntentVerdict.by_intent_comment_id must be int or None, got "
                f"{type(self.by_intent_comment_id).__name__}"
            )
        if self.outcome not in ("honored", "reshaped", "superseded", "no_op"):
            raise ValueError(
                f"IntentVerdict.outcome must be one of "
                "'honored', 'reshaped', 'superseded', 'no_op'; "
                f"got {self.outcome!r}"
            )
        if self.narrative is not None and not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.narrative, str
        ):
            raise TypeError(
                "IntentVerdict.narrative must be str or None, got "
                f"{type(self.narrative).__name__}"
            )

        # Container-shape checks BEFORE materialization (codex P2
        # round 4 on #1802: reject bare ``str`` / ``set`` / ``dict``
        # that would silently mis-coerce).
        if isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.ops, Mapping
        ):
            raise TypeError(
                "IntentVerdict.ops must be a sequence of op mappings, "
                "not a single mapping (did you mean to wrap in a tuple?)"
            )
        if isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.affected_task_ids, str
        ):
            raise TypeError(
                "IntentVerdict.affected_task_ids must be a sequence of str, "
                "not a bare str (a string iterates as single chars and would "
                f"be silently mis-stored as {tuple(self.affected_task_ids)!r})"
            )
        if isinstance(self.affected_task_ids, (set, frozenset)):
            raise TypeError(
                "IntentVerdict.affected_task_ids must be ordered "
                "(docstring contract: 'in result-list order'); got "
                f"{type(self.affected_task_ids).__name__}, which has "
                "nondeterministic iteration order"
            )
        if isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.affected_task_ids, Mapping
        ):
            # codex P2 on PR #1809: ``{"T1": ...}`` would iterate as
            # its keys and pass the per-entry str check, silently
            # turning ``{"T1": ..., "T2": ...}`` into
            # ``("T1", "T2")`` — losing the values and accepting an
            # invalid shape.  Reject mappings up front.
            raise TypeError(
                "IntentVerdict.affected_task_ids must be a sequence of str, "
                "not a mapping (iterating a mapping yields keys and would "
                "silently mis-store the contents)"
            )

        # Materialize collections ONCE before per-entry validation
        # (codex P2 round 4 on #1802: a generator passed as ops /
        # affected_task_ids would be exhausted by the validation
        # loop, and the frozen tuple would silently end up empty —
        # dropping all op/task attribution).
        ops_seq = tuple(self.ops)
        ids_seq = tuple(self.affected_task_ids)

        # Per-entry type checks.
        for op in ops_seq:
            if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                op, Mapping
            ):
                raise TypeError(
                    "IntentVerdict.ops entries must be Mapping, got "
                    f"{type(op).__name__}"
                )
        for tid in ids_seq:
            if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                tid, str
            ):
                raise TypeError(
                    "IntentVerdict.affected_task_ids entries must be str, "
                    f"got {type(tid).__name__}"
                )

        # Deep-freeze the materialized collections so the "frozen
        # snapshot" promise holds at the value level too — callers
        # can pass plain lists/dicts; ``deep_freeze`` (#1748) coerces
        # to ``tuple`` of ``frozendict``.
        from fido.frozen import deep_freeze

        object.__setattr__(self, "ops", deep_freeze(list(ops_seq)))
        object.__setattr__(self, "affected_task_ids", ids_seq)

        # codex P2 on #1802: enforce supersedence + outcome invariants
        # at construction so a malformed verdict crashes at the
        # boundary instead of leaking into the reply-back classifier
        # (INV-E #1803) or future graph traversals.
        if self.by_intent_comment_id == self.intent_comment_id:
            raise ValueError(
                f"IntentVerdict.by_intent_comment_id ({self.by_intent_comment_id}) "
                "must not reference the verdict's own intent (self-supersedence "
                "is meaningless); full-batch acyclicity is verified at apply time"
            )
        if self.outcome == "superseded" and self.by_intent_comment_id is None:
            raise ValueError(
                "IntentVerdict outcome='superseded' requires by_intent_comment_id "
                "to name the superseding intent (INV-E #1803 cannot determine "
                "self-vs-cross-author supersedence without it)"
            )
        if self.outcome != "superseded" and self.by_intent_comment_id is not None:
            raise ValueError(
                f"IntentVerdict outcome={self.outcome!r} must not carry "
                "by_intent_comment_id — only 'superseded' verdicts carry a "
                "supersedence pointer (codex P2 on #1802: reject contradictory "
                "metadata at the boundary)"
            )
        if (
            self.outcome in ("reshaped", "superseded")
            and not (self.narrative or "").strip()
        ):
            raise ValueError(
                f"IntentVerdict outcome={self.outcome!r} requires non-empty "
                "narrative — reply-back posts narrative verbatim per "
                "the voice-text-opus-not-templated convention"
            )
        if self.outcome == "no_op" and (self.ops or self.affected_task_ids):
            # codex P2 round 5 on #1802: ``no_op`` docstring promises
            # "produced no task changes".  A ``no_op`` verdict
            # carrying ops or affected_task_ids is contradictory —
            # downstream INV-E would classify the intent as
            # non-material (skipping reply-back) while the verdict
            # still claims task-change attribution.  Reject at the
            # boundary.
            raise ValueError(
                "IntentVerdict outcome='no_op' must have empty ops and "
                "affected_task_ids — got "
                f"ops={list(self.ops)!r}, "
                f"affected_task_ids={list(self.affected_task_ids)!r}"
            )


@dataclass(frozen=True)
class RescopeIntent:
    """Origin metadata for a rescope trigger from comment synthesis.

    Carries the plain-English change request alongside the originating
    comment identity and a stable ordering timestamp so the rescoper can
    process intents in arrival order and reply back to the right commenter.

    Attributes
    ----------
    change_request:
        Plain-English description of the requested scope change, as written
        by the synthesis LLM from the comment's content.
    comment_id:
        GitHub ID of the comment that triggered the rescope.
    timestamp:
        ISO-8601 UTC timestamp of when the rescope was triggered — used for
        ordering when multiple intents are coalesced into a single Opus call.
    """

    change_request: str
    comment_id: int
    timestamp: str
    comment_type: str = "pulls"
    """GitHub comment namespace: ``"pulls"`` for review-thread comments
    (the rescope reply path posts via ``reply_to_review_comment``);
    ``"issues"`` for top-level PR/issue comments where the webhook
    handler already posted a triage reply (rescope notifier silently
    skips per #1724 codex P2 — same policy as ``_notify_thread_change``)."""
    author: str = ""
    """GitHub login of the commenter.  Rendered alongside the change
    request in the rescope prompt so Opus can apply per-author
    supersedence semantics (later #1803 / INV-E: suppress reply-back
    when an intent was superseded by another from the same author —
    the commenter already knows they overrode themselves).  Empty
    string when the source path didn't populate it (legacy / synthetic
    test fixtures pre-INV-C)."""
    repo: str = ""
    """``owner/repo`` slug of the PR this intent was filed against.
    Carried through to rescope so any new task the rescope creates
    can populate its ``thread`` anchor (#1843) — without this, reply-
    back has no idea where to post follow-ups.  Empty string for
    synthetic test fixtures that don't simulate a real PR context."""
    pr_number: int = 0
    """GitHub PR number this intent was filed against.  Same use as
    :attr:`repo` — anchors the new task's ``thread`` field.  ``0`` for
    synthetic fixtures."""


@dataclass(frozen=True)
class GitIdentity:
    """GitHub-derived git commit identity.

    Name comes from the authenticated user's display name (falling back to
    login).  Email is always the GitHub noreply form
    ``{id}+{login}@users.noreply.github.com`` — the user's real email is
    never used, even if exposed via the API.
    """

    name: str
    email: str

    def __str__(self) -> str:
        return f"{self.name} <{self.email}>"
