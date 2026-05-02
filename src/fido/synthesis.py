"""Synthesis output types for the unified comment-handling turn.

The action vocabulary is constrained and enumerated (Constraint A from
#1230): Fido may only perform operations from this closed set when
responding to a PR comment.  No verb outside this list exists; no
free-form action escape hatch.

Reply text is a required top-level field (Constraint B): every synthesis
response must include a non-empty reply.  The invariant is enforced by
the type rather than buried as an optional list element — a
``CommentResponse`` without prose simply cannot be constructed.
"""

from dataclasses import dataclass

# Valid GitHub reaction shortcodes.
VALID_REACTIONS: frozenset[str] = frozenset(
    {"+1", "-1", "laugh", "confused", "heart", "hooray", "rocket", "eyes"}
)


def validate_reaction(emoji: str) -> str:
    """Return *emoji* unchanged, or raise ``ValueError`` if not a valid GitHub reaction."""
    if emoji not in VALID_REACTIONS:
        raise ValueError(
            f"Invalid reaction {emoji!r}. Valid reactions: {sorted(VALID_REACTIONS)}"
        )
    return emoji


# ---------------------------------------------------------------------------
# Individual action types (Constraint A — closed vocabulary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AddReaction:
    """Add an emoji reaction to the triggering comment.

    *emoji* must be one of the GitHub reaction shortcodes in
    :data:`VALID_REACTIONS`.
    """

    emoji: str


@dataclass(frozen=True)
class RescopeIntent:
    """A plain-English statement of scope-change intent.

    The synthesis call does not decide task mutations directly — it
    describes what should change and the existing rescope machinery
    (``reorder_tasks``) decides how to mutate the task list.

    *description* is a single plain-English sentence describing the
    requested scope change.
    """

    description: str

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("RescopeIntent.description must be non-empty")


@dataclass(frozen=True)
class Preempt:
    """Signal whether the current in-progress worker task should be preempted.

    When *preempt* is ``True``, the handler requests that the worker abort
    its current task and re-evaluate its queue immediately after actions
    are applied.
    """

    preempt: bool


@dataclass(frozen=True)
class NoOp:
    """Explicitly take no additional action beyond posting the required reply."""


# The closed vocabulary of additional effects Fido may produce from a single
# synthesis call.  Constraint A: no operations outside this set exist.
SynthesisAction = AddReaction | RescopeIntent | Preempt | NoOp


# ---------------------------------------------------------------------------
# Synthesis response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommentResponse:
    """Structured output from the unified comment-handling synthesis LLM call.

    Attributes
    ----------
    reasoning:
        Private chain-of-thought — logged for traceability, never posted
        to GitHub.
    reply_text:
        The reply to post to the PR comment thread.  Always required and
        always non-empty (Constraint B).  Always freshly synthesised from
        the actual comment context — never a template, never a canned
        phrase, never absent.
    actions:
        Ordered sequence of additional effects from the closed action
        vocabulary.  Executed in order after the reply is posted.
    """

    reasoning: str
    reply_text: str
    actions: tuple[SynthesisAction, ...] = ()

    def __post_init__(self) -> None:
        if not self.reply_text.strip():
            raise ValueError(
                "CommentResponse.reply_text must be non-empty (Constraint B: "
                "reply prose is always required and always freshly synthesised)"
            )
