"""Synthesis output types for the unified comment-handling turn.

Flat response schema (Constraint A from #1230): the synthesis call
returns a single flat JSON object with top-level ``emoji`` and
``change_request`` fields instead of an actions list.  The closed
vocabulary is enforced structurally — the only effects the model can
express are a single emoji reaction and/or a single scope-change
request.

Reply text is a required top-level field (Constraint B): every synthesis
response must include a non-empty reply.  The invariant is enforced by
the type rather than buried as an optional field — a
``CommentResponse`` without prose simply cannot be constructed.
"""

from dataclasses import dataclass, field

from fido.rocq.replied_comment_claims import (
    ReviewAct,
    ReviewAnswer,
    ReviewReplyOutcome,
)

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
# Insight — noteworthy observation captured from a comment interaction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Insight:
    """A noteworthy observation from a comment interaction.

    Populated by the synthesis LLM when the comment teaches something
    worth remembering about Rob, the work, or the collaboration pattern.
    The bar is the same as the persona: if it felt worth pausing over, it
    belongs here.

    Attributes
    ----------
    title:
        Short label for the insight (used as a GitHub issue title).
    hook:
        One sentence stating the observation — the lede.
    why:
        Two to three sentences explaining why it matters or what broader
        lesson it carries.
    """

    title: str
    hook: str
    why: str


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
    emoji:
        Optional GitHub reaction shortcode to add to the triggering
        comment.  Must be one of :data:`VALID_REACTIONS` or ``None``.
    change_request:
        Optional plain-English description of a requested scope change.
        When present, the action executor preempts and the rescope
        machinery decides the actual task mutations.
    insights:
        Noteworthy observations from this comment interaction.  Populated
        by the synthesis LLM when the comment teaches something worth
        remembering about Rob, the work, or the collaboration pattern.
        Empty list when nothing stood out.
    """

    reasoning: str
    reply_text: str
    emoji: str | None = None
    change_request: str | None = None
    insights: list[Insight] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.reply_text.strip():
            raise ValueError(
                "CommentResponse.reply_text must be non-empty (Constraint B: "
                "reply prose is always required and always freshly synthesised)"
            )
        if self.emoji is not None:
            validate_reaction(self.emoji)
        if self.change_request is not None and not self.change_request.strip():
            raise ValueError(
                "CommentResponse.change_request must be non-empty when present"
            )


# ---------------------------------------------------------------------------
# Outcome bridge — maps flat response to Rocq oracle ReviewReplyOutcome
# ---------------------------------------------------------------------------


def outcome_for_response(response: CommentResponse) -> ReviewReplyOutcome:
    """Map a :class:`CommentResponse` to a :class:`ReviewReplyOutcome`.

    The mapping is intentionally simple:

    - ``change_request`` present → :class:`ReviewAct` (the executor will
      preempt and hand the description to the rescope machinery).
    - ``change_request`` absent → :class:`ReviewAnswer` (a plain reply
      with no scope effect).

    This keeps the Rocq oracle's ``review_outcome_creates_tasks`` and
    ``review_outcome_resolves_thread`` predicates working correctly
    through the schema transition.
    """
    if response.change_request is not None:
        return ReviewAct()
    return ReviewAnswer()
