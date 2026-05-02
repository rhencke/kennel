"""Synthesis executor — dispatches a CommentResponse to GitHub effects.

Given a :class:`~fido.synthesis.CommentResponse` from the synthesis LLM
call, the executor posts the reply, manages the emoji reaction lifecycle
(eyes → final emoji), triggers rescope when a ``change_request`` is
present, and returns the :class:`ReviewReplyOutcome` for the Rocq
oracle.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from fido.rocq.replied_comment_claims import ReviewReplyOutcome
from fido.synthesis import CommentResponse, outcome_for_response
from fido.types import RescоpeIntent

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

    def list_reactions(
        self,
        repo: str,
        comment_type: str,
        comment_id: int | str,
    ) -> list[dict[str, Any]]: ...

    def delete_reaction(
        self,
        repo: str,
        comment_type: str,
        comment_id: int | str,
        reaction_id: int | str,
    ) -> None: ...


class RescopeTrigger(Protocol):
    """Triggers a background rescope from a comment synthesis response.

    Receives a :class:`~fido.types.RescоpeIntent` carrying the plain-English
    change request alongside the originating comment identity and timestamp
    so the rescope machinery can track which comment triggered each intent
    and reply back on material outcomes.
    """

    def trigger_rescope(self, intent: RescоpeIntent) -> None: ...


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
        self._maybe_trigger_rescope(response, target)

        # 4. Return outcome for Rocq oracle
        return outcome_for_response(response)

    def execute_effects_only(
        self,
        response: CommentResponse,
        target: CommentTarget,
    ) -> ReviewReplyOutcome:
        """Execute post-reply effects for *response* against *target*.

        This method handles the emoji reaction lifecycle and rescope trigger,
        but does **not** post the reply itself.  Use this when the caller owns
        the reply-posting step (e.g. to thread the outbox protocol through
        it), and only needs the executor to manage the remaining side effects.

        Effects, in order:

        1. **Remove eyes reaction** — deletes any ``eyes`` reaction on the
           triggering comment that was posted by the authenticated user
           (best-effort; errors are logged but do not propagate).
        2. **Add emoji reaction** — if ``response.emoji`` is set, adds it to
           the triggering comment after removing eyes.
        3. **Trigger rescope** — if ``response.change_request`` is set, hands
           it to the rescope trigger.
        4. **Return outcome** — maps the response to a
           :class:`ReviewReplyOutcome` for the Rocq oracle.
        """
        # 1. Remove eyes reaction (best-effort)
        self._remove_eyes_reaction(target)

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
        self._maybe_trigger_rescope(response, target)

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

    def _maybe_trigger_rescope(
        self, response: CommentResponse, target: CommentTarget
    ) -> None:
        """Build a :class:`RescоpeIntent` and fire the rescope trigger if configured.

        No-op when *response.change_request* is ``None``.  Logs a warning if a
        change_request is present but no rescope trigger was injected.
        """
        if response.change_request is None:
            return
        if self._rescope is not None:
            intent = RescоpeIntent(
                change_request=response.change_request,
                comment_id=target.comment_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            log.info(
                "triggering rescope for comment %d: %s",
                intent.comment_id,
                intent.change_request[:80],
            )
            self._rescope.trigger_rescope(intent)
        else:
            log.warning(
                "change_request present but no rescope trigger configured — "
                "skipping rescope for: %s",
                response.change_request[:80],
            )

    def _remove_eyes_reaction(self, target: CommentTarget) -> None:
        """Remove the ``eyes`` reaction from *target* (best-effort).

        Lists all reactions on the comment and deletes any with
        ``content == "eyes"``.  Errors are logged but never propagated —
        a failed reaction cleanup must not abort an otherwise-successful
        reply.
        """
        try:
            reactions = self._gh.list_reactions(
                target.repo,
                target.comment_type,
                target.comment_id,
            )
            for reaction in reactions:
                if reaction.get("content") == "eyes":
                    reaction_id = reaction.get("id")
                    if reaction_id is not None:
                        log.info(
                            "removing eyes reaction %s from comment %s",
                            reaction_id,
                            target.comment_id,
                        )
                        self._gh.delete_reaction(
                            target.repo,
                            target.comment_type,
                            target.comment_id,
                            reaction_id,
                        )
        except Exception:
            log.exception(
                "failed to remove eyes reaction from comment %s — continuing",
                target.comment_id,
            )
