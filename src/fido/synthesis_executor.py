"""Synthesis executor — dispatches a CommentResponse to GitHub effects.

Given a :class:`~fido.synthesis.CommentResponse` from the synthesis LLM
call, the executor posts the reply, manages the emoji reaction lifecycle
(eyes → final emoji), triggers rescope when a ``change_request`` is
present, and returns the :class:`ReviewReplyOutcome` for the Rocq
oracle.
"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from fido.rocq.replied_comment_claims import ReviewReplyOutcome
from fido.synthesis import CommentResponse, outcome_for_response

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommentTarget:
    """Identifies the comment being replied to and reacted on.

    Attributes
    ----------
    repo:
        Full repo slug (e.g. ``"owner/repo"``).
    pr:
        PR number.
    comment_id:
        ID of the triggering comment.
    comment_type:
        GitHub comment namespace: ``"pulls"`` for review comments,
        ``"issues"`` for top-level PR/issue comments.
    """

    repo: str
    pr: int
    comment_id: int
    comment_type: str


class ReplyPoster(Protocol):
    """Posts a reply to a comment and returns the API response."""

    def reply_to_review_comment(
        self,
        repo: str,
        pr: int | str,
        body: str,
        in_reply_to: int | str,
    ) -> dict[str, Any]: ...

    def comment_issue(
        self,
        repo: str,
        number: int | str,
        body: str,
    ) -> dict[str, Any]: ...

    def add_reaction(
        self,
        repo: str,
        comment_type: str,
        comment_id: int | str,
        content: str,
    ) -> None: ...


class RescopeTrigger(Protocol):
    """Triggers a background rescope with a plain-English change request."""

    def trigger_rescope(self, change_request: str) -> None: ...


class SynthesisExecutor:
    """Dispatches a :class:`CommentResponse` to its side effects.

    Effects, in order:

    1. **Post reply** — always (Constraint B guarantees non-empty text).
    2. **Add emoji reaction** — if ``response.emoji`` is set, add it to
       the triggering comment.
    3. **Trigger rescope** — if ``response.change_request`` is set, hand
       it to the rescope trigger (which preempts the current task and
       passes the intent to the rescope machinery).
    4. **Return outcome** — maps the response to a
       :class:`ReviewReplyOutcome` for the Rocq oracle.

    Dependencies are injected through the constructor per the
    constructor-DI convention.
    """

    def __init__(
        self,
        gh: ReplyPoster,
        rescope: RescopeTrigger | None = None,
    ) -> None:
        self._gh = gh
        self._rescope = rescope

    def execute(
        self,
        response: CommentResponse,
        target: CommentTarget,
    ) -> ReviewReplyOutcome:
        """Execute all effects for *response* against *target*.

        Returns the :class:`ReviewReplyOutcome` so the caller can feed
        it to the Rocq oracle for reply-claim bookkeeping.
        """
        # 1. Post reply
        self._post_reply(response.reply_text, target)

        # 2. Add emoji reaction if present
        if response.emoji is not None:
            log.info(
                "adding %s reaction to comment %s",
                response.emoji,
                target.comment_id,
            )
            self._gh.add_reaction(
                target.repo,
                target.comment_type,
                target.comment_id,
                response.emoji,
            )

        # 3. Trigger rescope if change_request present
        if response.change_request is not None:
            if self._rescope is not None:
                log.info(
                    "triggering rescope: %s",
                    response.change_request[:80],
                )
                self._rescope.trigger_rescope(response.change_request)
            else:
                log.warning(
                    "change_request present but no rescope trigger configured — "
                    "skipping rescope for: %s",
                    response.change_request[:80],
                )

        # 4. Return outcome for Rocq oracle
        return outcome_for_response(response)

    def _post_reply(self, body: str, target: CommentTarget) -> None:
        """Post the reply to the correct endpoint based on comment type."""
        if target.comment_type == "pulls":
            log.info(
                "posting review reply to %s PR #%d comment %d",
                target.repo,
                target.pr,
                target.comment_id,
            )
            self._gh.reply_to_review_comment(
                target.repo, target.pr, body, target.comment_id
            )
        else:
            log.info(
                "posting issue comment to %s #%d",
                target.repo,
                target.pr,
            )
            self._gh.comment_issue(target.repo, target.pr, body)
