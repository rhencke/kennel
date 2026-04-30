import logging
import re
import subprocess
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fido.claude import ClaudeClient
from fido.config import Config, RepoConfig
from fido.github import GitHub
from fido.prompts import NO_TOOLS_CLAUSE, Prompts
from fido.provider import (
    ProviderAgent,
    safe_voice_turn,
    set_thread_kind,
    set_thread_repo,
)
from fido.provider_factory import DefaultProviderFactory
from fido.registry import WorkerRegistry
from fido.rocq import replied_comment_claims as oracle
from fido.rocq import thread_auto_resolve as thread_resolve_oracle
from fido.rocq import webhook_command_translation as wct_oracle
from fido.rocq import webhook_ingress_dedupe as ingress_fsm
from fido.state import State
from fido.store import FidoStore, append_reply_promise_markers
from fido.tasks import Tasks, thread_comment_author_for_auto_resolve_oracle
from fido.types import TaskType

log = logging.getLogger(__name__)

_FIDO_LOGINS = frozenset({"fidocancode", "fido-can-code"})

# Per-work_dir coalescing state for _reorder_tasks_background.
# Ensures at most one Opus call in-flight + one pending per repo.
_reorder_coalesce: dict[str, dict[str, Any]] = {}
_reorder_coalesce_lock = threading.Lock()


class WebhookIngressOracle:
    """Per-process at-least-once delivery vs exactly-once dispatch oracle.

    Tracks the ingress FSM state for every (repo, delivery_id) pair seen
    during this process lifetime.  Keyed by repo name → delivery ID string →
    :class:`~fido.rocq.webhook_ingress_dedupe.State`.

    All mutations are guarded by :attr:`_lock` so concurrent webhook handler
    threads running in Python 3.14t's free-threaded runtime cannot race on
    the delivery-ID table.

    The oracle is crash-on-violation: :meth:`_transition` raises
    :exc:`AssertionError` (with the theorem name) whenever the extracted Rocq
    FSM rejects a transition.  Callers that receive ``None`` from
    :meth:`check_dispatch` should suppress the event (already dispatched).
    """

    def __init__(self) -> None:
        # repo_name → delivery_id → ingress FSM state
        self._states: dict[str, dict[str, ingress_fsm.State]] = {}
        self._lock = threading.Lock()

    def _transition(
        self,
        repo_name: str,
        delivery_id: str,
        event: ingress_fsm.Event,
        current: ingress_fsm.State,
    ) -> ingress_fsm.State:
        """Fire *event* for *delivery_id*, raising ``AssertionError`` if rejected.

        Single oracle for every FSM transition so a coordination bug surfaces
        as a crash rather than silent event duplication or suppression.
        """
        new_state = ingress_fsm.transition(current, event)
        if new_state is None:
            raise AssertionError(
                f"webhook_ingress_dedupe FSM: {type(event).__name__} rejected in "
                f"state {type(current).__name__} for repo {repo_name!r} "
                f"delivery {delivery_id!r}"
            )
        log.debug(
            "ingress[%s][%s]: FSM %s →%s via %s",
            repo_name,
            delivery_id,
            type(current).__name__,
            type(new_state).__name__,
            type(event).__name__,
        )
        return new_state

    def check_dispatch(
        self,
        repo_name: str,
        delivery_id: str,
        *,
        collapse_review: bool = False,
    ) -> ingress_fsm.State | None:
        """Record an incoming webhook and return the new FSM state, or None to suppress.

        Parameters
        ----------
        repo_name:
            The ``owner/repo`` string for the repo receiving the webhook.
        delivery_id:
            The ``X-GitHub-Delivery`` header value identifying this delivery.
        collapse_review:
            When True, fire ``CollapseReview`` instead of ``Arrive``.
            Used for ``pull_request_review / submitted`` events whose inline
            comments are handled individually — the review-level event is
            collapsed rather than dispatched.

        Returns
        -------
        ingress_fsm.State
            The new FSM state after recording the event.
        None
            The delivery should be suppressed — it was already dispatched or
            collapsed in an earlier call.  The caller must return ``None``
            from ``dispatch()`` without executing any side effects.
        """
        with self._lock:
            repo_states = self._states.setdefault(repo_name, {})
            current = repo_states.get(delivery_id, ingress_fsm.Fresh())
            if isinstance(current, ingress_fsm.Fresh):
                event: ingress_fsm.Event = (
                    ingress_fsm.CollapseReview()
                    if collapse_review
                    else ingress_fsm.Arrive()
                )
            elif isinstance(current, ingress_fsm.Dispatched):
                event = ingress_fsm.Redeliver()
            else:
                # Collapsed or any other terminal state — fire Arrive to let
                # the FSM reject it (returns None → suppress).
                event = ingress_fsm.Arrive()
            new_state = self._transition(repo_name, delivery_id, event, current)
            repo_states[delivery_id] = new_state
        # Arrive from Fresh → Dispatched: proceed normally.
        # CollapseReview from Fresh → Collapsed: suppress (collapsed away).
        # Redeliver from Dispatched → Dispatched: suppress (already handled).
        # Arrive from Dispatched → None (AssertionError above): never reached.
        if isinstance(new_state, ingress_fsm.Collapsed):
            log.info(
                "ingress[%s][%s]: CollapseReview — suppressing pull_request_review",
                repo_name,
                delivery_id,
            )
            return None
        if isinstance(new_state, ingress_fsm.Dispatched) and isinstance(
            event, ingress_fsm.Redeliver
        ):
            log.info(
                "ingress[%s][%s]: Redeliver — suppressing duplicate delivery",
                repo_name,
                delivery_id,
            )
            return None
        return new_state


def _configured_agent(config: Config, repo_cfg: RepoConfig) -> ProviderAgent:
    return DefaultProviderFactory(
        session_system_file=config.sub_dir / "persona.md"
    ).create_agent(
        repo_cfg,
        work_dir=repo_cfg.work_dir,
        repo_name=repo_cfg.name,
    )


@dataclass
class Action:
    prompt: str
    reply_to: dict[str, Any] | None = None  # {repo, pr, comment_id}
    review_comments: dict[str, Any] | None = None  # {repo, pr, review_id}
    comment_body: str | None = None
    preempts_worker: bool = False
    is_bot: bool = False
    context: dict[str, Any] | None = None  # {pr_title, file, diff_hunk, line, pr_body}
    thread: dict[str, Any] | None = (
        None  # {repo, pr, comment_id} for task prioritisation
    )


def _pr_number_from_api_url(url: str, kind: str) -> int:
    """Extract a PR/issue number from a GitHub API URL."""
    pattern = r"/issues/(\d+)$" if kind == "issues" else r"/pulls/(\d+)$"
    match = re.search(pattern, url)
    if match is None:
        raise ValueError(f"invalid GitHub API URL for {kind}: {url!r}")
    return int(match.group(1))


def build_review_comment_action(
    repo: str,
    pr_number: int,
    pr_title: str,
    pr_body: str,
    comment: dict[str, Any],
    *,
    comment_body: str | None = None,
    comment_author: str | None = None,
) -> Action:
    """Rebuild a review-comment Action from live GitHub state."""
    user = (
        comment_author
        if comment_author is not None
        else comment.get("user", {}).get("login", "")
    )
    body = comment_body if comment_body is not None else (comment["body"] or "")
    is_bot = user.endswith("[bot]")
    return Action(
        prompt=(
            f"Review comment on PR #{pr_number} by {user}"
            f" ({'bot' if is_bot else 'human/owner'}):\n\n{body}"
        ),
        reply_to={
            "repo": repo,
            "pr": pr_number,
            "comment_id": comment["id"],
            "url": comment["html_url"],
            "author": user,
            "comment_type": "pulls",
        },
        comment_body=body,
        is_bot=is_bot,
        context={
            "pr_title": pr_title,
            "pr_body": pr_body,
            "file": comment["path"],
            "line": comment["line"],
            "diff_hunk": comment["diff_hunk"],
        },
    )


def _build_issue_comment_action(
    repo: str,
    pr_number: int,
    pr_title: str,
    pr_body: str,
    comment: dict[str, Any],
    *,
    comment_body: str | None = None,
) -> Action:
    """Rebuild a top-level PR-comment Action from live GitHub state."""
    user = comment["user"]["login"]
    body = comment_body if comment_body is not None else (comment["body"] or "")
    is_bot = user.endswith("[bot]")
    comment_id = int(comment["id"])
    return Action(
        prompt=f"PR top-level comment on #{pr_number} by {user}:\n\n{body}",
        reply_to=None,
        comment_body=body,
        is_bot=is_bot,
        context={
            "pr_title": pr_title,
            "pr_body": pr_body,
            "comment_id": comment_id,
        },
        thread={
            "repo": repo,
            "pr": pr_number,
            "comment_id": comment_id,
            "url": comment["html_url"],
            "author": user,
            "comment_type": "issues",
        },
    )


def _reply_promise_ids(context: dict[str, Any] | None) -> tuple[str, ...]:
    """Return every covered promise id carried in *context*."""
    if not context:
        return ()
    single = context.get("reply_promise_id")
    many = context.get("reply_promise_ids")
    result: list[str] = []
    if isinstance(single, str) and single:
        result.append(single)
    if isinstance(many, list):
        result.extend(str(promise_id) for promise_id in many if promise_id)
    return tuple(dict.fromkeys(result))


def _record_reply_artifact(
    repo_cfg: RepoConfig,
    *,
    artifact_comment_id: int | None,
    comment_type: str,
    lane_key: str,
    promise_ids: tuple[str, ...],
) -> None:
    """Persist one visible reply artifact and mark its promises posted."""
    if artifact_comment_id is None or not promise_ids:
        return
    store = FidoStore(repo_cfg.work_dir)
    store.record_artifact(
        artifact_comment_id=artifact_comment_id,
        comment_type=comment_type,
        lane_key=lane_key,
        promise_ids=promise_ids,
    )
    for promise_id in promise_ids:
        store.mark_posted(promise_id)


def _posted_comment_id(posted: object) -> int | None:
    """Return the GitHub comment id from a post response, when available."""
    if not isinstance(posted, dict):
        return None
    comment_id = posted.get("id")
    if isinstance(comment_id, int):
        return comment_id
    return None


def _review_lane_key(repo: str, pr_number: int, root_comment_id: int) -> str:
    return f"pulls:{repo}:{pr_number}:thread:{root_comment_id}"


def _issue_lane_key(repo: str, pr_number: int) -> str:
    return f"issues:{repo}:{pr_number}"


def _review_outcome(category: str) -> oracle.ReviewReplyOutcome:
    return {
        "ACT": oracle.ReviewAct(),
        "DO": oracle.ReviewDo(),
        "ASK": oracle.ReviewAsk(),
        "ANSWER": oracle.ReviewAnswer(),
        "DEFER": oracle.ReviewDefer(),
        "DUMP": oracle.ReviewDump(),
    }[category]


def _bot_feedback_outcome(
    category: str,
) -> thread_resolve_oracle.BotFeedbackOutcome | None:
    return {
        "DO": thread_resolve_oracle.BotFeedbackDo(),
        "DUMP": thread_resolve_oracle.BotFeedbackDump(),
    }.get(category)


def bot_feedback_creates_tasks(category: str) -> bool:
    """Return whether a bot feedback outcome should create task objects."""
    outcome = _bot_feedback_outcome(category)
    if outcome is None:
        return False
    return isinstance(
        thread_resolve_oracle.bot_feedback_decision(outcome),
        thread_resolve_oracle.TakeBotSuggestion,
    )


def bot_feedback_resolves_thread(category: str) -> bool:
    """Return whether a bot feedback outcome should resolve the thread."""
    outcome = _bot_feedback_outcome(category)
    if outcome is None:
        return False
    return isinstance(
        thread_resolve_oracle.bot_feedback_decision(outcome),
        thread_resolve_oracle.DumpBotSuggestionAndClose,
    )


def review_outcome_creates_tasks(category: str, *, is_bot: bool = False) -> bool:
    """Return whether a review reply outcome should create task objects."""
    if is_bot:
        return bot_feedback_creates_tasks(category)
    if category not in {"ACT", "DO", "ASK", "ANSWER", "DEFER", "DUMP"}:
        return False
    return bool(oracle.review_outcome_creates_tasks(_review_outcome(category)))


def review_outcome_resolves_thread(category: str, *, is_bot: bool = False) -> bool:
    """Return whether a review reply outcome should resolve the thread."""
    if is_bot:
        return bot_feedback_resolves_thread(category)
    if category not in {"ACT", "DO", "ASK", "ANSWER", "DEFER", "DUMP"}:
        return False
    return bool(oracle.review_outcome_resolves_thread(_review_outcome(category)))


def reply_outcome_creates_tasks(
    category: str,
    *,
    thread: dict[str, Any] | None,
    is_bot: bool = False,
) -> bool:
    """Return whether a reply outcome should create task objects."""
    if thread is not None:
        return review_outcome_creates_tasks(category, is_bot=is_bot)
    return category not in ("DUMP", "ANSWER", "ASK", "DEFER")


def queue_reply_tasks(
    category: str,
    titles: list[str],
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    *,
    thread: dict[str, Any] | None,
    is_bot: bool = False,
    registry: Any = None,
    create_task_fn: Callable[..., object] | None = None,
) -> int:
    """Create any tasks implied by a reply outcome.

    Returns the number of created task objects.
    """
    if not reply_outcome_creates_tasks(category, thread=thread, is_bot=is_bot):
        return 0
    task_fn = create_task if create_task_fn is None else create_task_fn
    created = 0
    for title in titles:
        task = task_fn(
            title,
            config,
            repo_cfg,
            gh,
            thread=thread,
            registry=registry,
        )
        if not (isinstance(task, dict) and task.get("status") == "skipped_resolved"):
            created += 1
    return created


def _apply_reply_result(
    category: str,
    titles: list[str],
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    *,
    thread: dict[str, Any] | None,
    registry: WorkerRegistry | None,
) -> None:
    """Apply task side effects from a recovered reply."""
    queue_reply_tasks(
        category,
        titles,
        config,
        repo_cfg,
        gh,
        thread=thread,
        registry=registry,
    )


def _mark_promises_failed(store: FidoStore, promise_ids: Iterable[str]) -> None:
    """Mark every promise in the group retryable after replay failure."""
    for promise_id in promise_ids:
        store.mark_failed(promise_id)


def _ack_promises(store: FidoStore, promise_ids: Iterable[str]) -> None:
    """Ack every promise in the group after replay success."""
    for promise_id in promise_ids:
        store.ack_promise(promise_id)


def recover_reply_promises(
    fido_dir: Path,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    pr_number: int,
    *,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
    registry: WorkerRegistry | None = None,
) -> bool:
    """Recover queued webhook replies for the current PR from SQLite promises."""
    store = FidoStore(repo_cfg.work_dir)
    promises = store.recoverable_promises()
    if not promises:
        return False

    pr_issue = gh.view_issue(repo_cfg.name, pr_number)
    pr_title = pr_issue["title"]
    pr_body = pr_issue["body"] or ""
    processed_any = False
    pull_entries: dict[str, tuple[dict[str, Any], int, int]] = {}
    issue_comments: list[dict[str, Any]] = []
    if any(p.comment_type == "issues" for p in promises):
        issue_comments = gh.get_issue_comments(repo_cfg.name, pr_number)
        if store.recover_from_bodies(c.get("body", "") for c in issue_comments):
            processed_any = True
        issue_comments_by_id = {int(c["id"]): c for c in issue_comments if c.get("id")}
    else:
        issue_comments_by_id = {}

    issue_groups: dict[int, list[tuple[str, dict[str, Any]]]] = {}

    for promise in promises:
        current = store.promise(promise.promise_id)
        if current is None or current.state == "acked":
            continue
        if promise.comment_type == "pulls":
            fetched = gh.fetch_comment_thread(
                repo_cfg.name, pr_number, promise.anchor_comment_id
            )
            if fetched and store.recover_from_bodies(
                c.get("body", "") for c in fetched
            ):
                processed_any = True
                continue
            comment = gh.get_pull_comment(repo_cfg.name, promise.anchor_comment_id)
            if comment is None:
                store.mark_failed(promise.promise_id)
                continue
            comment_pr = _pr_number_from_api_url(comment["pull_request_url"], "pulls")
            root_id = (
                int(comment["in_reply_to_id"])
                if "in_reply_to_id" in comment and comment["in_reply_to_id"] is not None
                else int(comment["id"])
            )
            pull_entries[promise.promise_id] = (comment, comment_pr, root_id)
        else:
            comment = issue_comments_by_id.get(promise.anchor_comment_id)
            if comment is None:
                comment = gh.get_issue_comment(repo_cfg.name, promise.anchor_comment_id)
            if comment is None:
                store.mark_failed(promise.promise_id)
                continue
            comment_pr = _pr_number_from_api_url(comment["issue_url"], "issues")
            if comment_pr != pr_number:
                continue
            issue_groups.setdefault(comment_pr, []).append(
                (promise.promise_id, comment)
            )

    for comment_pr, group in issue_groups.items():
        combined_parts: list[str] = []
        for _, group_comment in group:
            body = group_comment["body"] or ""
            if body and body not in combined_parts:
                combined_parts.append(body)
        representative = group[-1][1]
        action = _build_issue_comment_action(
            repo_cfg.name,
            pr_number,
            pr_title,
            pr_body,
            representative,
            comment_body="\n\n---\n\n".join(combined_parts) if combined_parts else None,
        )
        action.context = {
            **(action.context or {}),
            "reply_promise_id": group[0][0],
            "reply_promise_ids": [promise_id for promise_id, _ in group],
        }
        try:
            category, titles = reply_to_issue_comment(
                action,
                config,
                repo_cfg,
                gh,
                agent=agent,
                prompts=prompts,
            )
        except Exception:
            _mark_promises_failed(store, (promise_id for promise_id, _ in group))
            raise
        _ack_promises(store, (promise_id for promise_id, _ in group))
        _apply_reply_result(
            category,
            titles,
            config,
            repo_cfg,
            gh,
            thread=action.thread,
            registry=registry,
        )
        processed_any = True

    for promise in promises:
        if promise.comment_type != "pulls":
            continue
        current = store.promise(promise.promise_id)
        if current is None or current.state == "acked":
            continue
        if promise.promise_id not in pull_entries:
            continue
        comment, comment_pr, root_id = pull_entries[promise.promise_id]
        if comment_pr != pr_number:
            continue

        group: list[tuple[str, dict[str, Any]]] = []
        for candidate_promise_id, (
            candidate_comment,
            candidate_pr,
            candidate_root_id,
        ) in pull_entries.items():
            if candidate_pr == pr_number and candidate_root_id == root_id:
                group.append((candidate_promise_id, candidate_comment))

        combined_parts: list[str] = []
        for _, group_comment in group:
            body = group_comment["body"] or ""
            if body and body not in combined_parts:
                combined_parts.append(body)
        combined_body = "\n\n---\n\n".join(combined_parts) if combined_parts else None
        representative = group[-1][1]
        action = build_review_comment_action(
            repo_cfg.name,
            pr_number,
            pr_title,
            pr_body,
            representative,
            comment_body=combined_body,
        )
        action.context = {
            **(action.context or {}),
            "reply_promise_id": promise.promise_id,
            "reply_promise_ids": [group_promise_id for group_promise_id, _ in group],
        }
        try:
            category, titles = reply_to_comment(
                action,
                config,
                repo_cfg,
                gh,
                agent=agent,
                prompts=prompts,
            )
        except Exception:
            _mark_promises_failed(
                store, (group_promise_id for group_promise_id, _ in group)
            )
            raise
        _ack_promises(store, (group_promise_id for group_promise_id, _ in group))
        _apply_reply_result(
            category,
            titles,
            config,
            repo_cfg,
            gh,
            thread=action.reply_to,
            registry=registry,
        )
        processed_any = True

    return processed_any


def _is_allowed(user: str, repo_cfg: RepoConfig, config: Config) -> bool:
    """Check if user is a repo collaborator or an allowed bot.

    ``repo_cfg.membership.collaborators`` is populated at server startup
    (``server.populate_memberships``) and excludes the bot itself.
    """
    return user in repo_cfg.membership.collaborators or user in config.allowed_bots


def dispatch(
    event: str,
    payload: dict[str, Any],
    config: Config,
    repo_cfg: RepoConfig,
    *,
    delivery_id: str | None = None,
    oracle: WebhookIngressOracle | None = None,
) -> Action | None:
    """Map a GitHub webhook event to an action. Returns None if ignored.

    When *delivery_id* and *oracle* are provided the
    :class:`WebhookIngressOracle` is consulted before any routing logic runs.
    Duplicate deliveries (same delivery ID arriving a second time) and
    collapsed ``pull_request_review / submitted`` events are suppressed by
    returning ``None`` before any side effects execute.
    """
    action = payload.get("action", "")
    repo = payload.get("repository", {}).get("full_name", "")

    # Oracle check — deduplicate at the ingress boundary.
    if delivery_id is not None and oracle is not None:
        collapse_review = event == "pull_request_review" and action == "submitted"
        ingress_result = oracle.check_dispatch(
            repo_cfg.name,
            delivery_id,
            collapse_review=collapse_review,
        )
        if ingress_result is None:
            return None

    if event == "ping":
        log.info("ping received — hook_id=%s", payload.get("hook_id"))
        return None

    if event == "issues" and action == "assigned":
        assignee = payload.get("assignee", {}).get("login", "")
        issue = payload.get("issue", {})
        number = issue.get("number")
        title = issue.get("title", "")
        if not number:
            return None
        log.info("issue #%s assigned to %s: %s", number, assignee, title)
        wev = wct_oracle.EvtIssueAssigned(1, number, assignee)
        cmd = wct_oracle.translate(wev)
        assert isinstance(cmd, wct_oracle.CmdIssueAssigned), "translate_total"
        return Action(prompt=f"New issue #{number} assigned to {assignee}: {title}")

    if event == "pull_request_review" and action == "submitted":
        review = payload.get("review", {})
        pr = payload.get("pull_request", {})
        number = pr.get("number")
        state = review.get("state", "")
        user = review.get("user", {}).get("login", "")
        review_id = review.get("id")
        if not number:
            return None
        if not _is_allowed(user, repo_cfg, config):
            log.debug("ignoring review on PR #%s by %s (not allowed)", number, user)
            return None
        log.info("review on PR #%s: %s by %s", number, state, user)
        if review_id is not None:
            wev = wct_oracle.EvtReviewSubmitted(1, number, review_id, user)
            cmd = wct_oracle.translate(wev)
            assert isinstance(cmd, wct_oracle.CmdReviewSubmitted), "translate_total"
        return Action(
            prompt=f"Review on PR #{number}: {state} by {user}",
            review_comments={"repo": repo, "pr": number, "review_id": review_id}
            if review_id
            else None,
        )

    if event == "pull_request_review_comment" and action == "created":
        comment = payload.get("comment", {})
        pr = payload.get("pull_request", {})
        number = pr.get("number")
        user = comment.get("user", {}).get("login", "")
        comment_id = comment.get("id")
        if user.lower() in ("fidocancode", "fido-can-code"):
            log.debug("ignoring own comment on PR #%s", number)
            return None
        if not number:
            return None
        if not _is_allowed(user, repo_cfg, config):
            log.debug("ignoring comment on PR #%s by %s (not allowed)", number, user)
            return None
        comment_body = comment.get("body", "") or ""
        log.info("comment on PR #%s by %s: %s", number, user, comment_body[:80])
        is_bot = user.endswith("[bot]")
        if comment_id is not None:
            wev = wct_oracle.EvtReviewComment(1, number, comment_id, user, is_bot)
            cmd = wct_oracle.translate(wev)
            assert isinstance(cmd, wct_oracle.CmdComment), "translate_total"
            assert isinstance(cmd.cmd_kind, wct_oracle.ReviewLine), "translate_total"
        return Action(
            prompt=f"Review comment on PR #{number} by {user} ({'bot' if is_bot else 'human/owner'}):\n\n{comment_body}",
            reply_to={
                "repo": repo,
                "pr": number,
                "comment_id": comment_id,
                "url": comment.get("html_url", ""),
                "author": user,
                "comment_type": "pulls",
            },
            comment_body=comment_body,
            is_bot=is_bot,
            context={
                "pr_title": pr.get("title", ""),
                "pr_body": pr.get("body", "") or "",
                "file": comment.get("path", ""),
                "line": comment.get("line"),
                "diff_hunk": comment.get("diff_hunk", ""),
            },
        )

    if event == "issue_comment" and action == "created":
        comment = payload.get("comment", {})
        issue = payload.get("issue", {})
        user = comment.get("user", {}).get("login", "")
        pr = issue.get("pull_request")
        if not pr:
            log.debug("issue_comment on non-PR issue — ignoring")
            return None
        if user.lower() in ("fidocancode", "fido-can-code"):
            log.debug("ignoring own comment on PR")
            return None
        if not _is_allowed(user, repo_cfg, config):
            log.debug("ignoring comment by %s (not allowed)", user)
            return None
        number = issue.get("number")
        comment_body = comment.get("body", "") or ""
        comment_id = comment.get("id")
        is_bot = user.endswith("[bot]")
        log.info("PR comment on #%s by %s: %s", number, user, comment_body[:80])
        if number is not None and comment_id is not None:
            wev = wct_oracle.EvtIssueComment(1, number, comment_id, user, is_bot)
            cmd = wct_oracle.translate(wev)
            assert isinstance(cmd, wct_oracle.CmdComment), "translate_total"
            assert isinstance(cmd.cmd_kind, wct_oracle.TopLevelPR), "translate_total"
        return Action(
            prompt=f"PR top-level comment on #{number} by {user}:\n\n{comment_body}",
            reply_to=None,  # top-level comments use issues API, not pulls
            comment_body=comment_body,
            is_bot=is_bot,
            context={
                "pr_title": issue.get("title", ""),
                "pr_body": issue.get("body", "") or "",
                "comment_id": comment_id,
            },
            thread={
                "repo": repo,
                "pr": number,
                "comment_id": comment_id,
                "url": comment.get("html_url", ""),
                "author": user,
                "comment_type": "issues",
            }
            if number and comment_id
            else None,
        )

    if event == "check_run" and action == "completed":
        check = payload.get("check_run", {})
        conclusion = check.get("conclusion", "")
        if conclusion not in ("failure", "timed_out"):
            log.debug("check_run completed with %s — ignoring", conclusion)
            return None
        name = check.get("name", "")
        prs = check.get("pull_requests", [])
        pr_nums = [pr.get("number") for pr in prs if pr.get("number")]
        log.info("CI failure: %s (%s) on PRs %s", name, conclusion, pr_nums)
        _ci_conclusion = (
            wct_oracle.CIFailure()
            if conclusion == "failure"
            else wct_oracle.CITimedOut()
        )
        wev = wct_oracle.EvtCIFailure(1, name, _ci_conclusion, pr_nums)
        cmd = wct_oracle.translate(wev)
        assert isinstance(cmd, wct_oracle.CmdCIFailure), "translate_total"
        pr_str = ", ".join(f"#{n}" for n in pr_nums) if pr_nums else "unknown PR"
        return Action(
            prompt=f"CI failure on {pr_str}: {name} ({conclusion})",
            preempts_worker=True,
        )

    if event == "pull_request" and action == "closed":
        pr = payload.get("pull_request", {})
        if not pr.get("merged"):
            log.debug("PR #%s closed without merge — ignoring", pr.get("number"))
            return None
        number = pr.get("number")
        log.info("PR #%s merged", number)
        if number is not None:
            wev = wct_oracle.EvtPRMerged(1, number)
            cmd = wct_oracle.translate(wev)
            assert isinstance(cmd, wct_oracle.CmdPRMerged), "translate_total"
        return Action(prompt=f"PR #{number} merged — cleanup")

    log.debug("ignored event: %s (action=%s)", event, action)
    return None


def _load_persona(config: Config) -> str:
    """Read persona.md from sub_dir; return empty string if missing."""
    try:
        return (config.sub_dir / "persona.md").read_text()
    except FileNotFoundError:
        return ""


def _open_defer_issue(gh: Any, repo: str, pr_url: str, title: str, comment: str) -> str:
    """Create a tracking issue for a DEFER triage result.

    Returns the new issue URL.  Raises on any creation failure so the caller
    fails closed rather than crafting a reply that references a missing issue.
    """
    issue_body = f"Deferred from {pr_url}\n\n> {comment}" if pr_url else comment
    url = gh.create_issue(repo, title, issue_body)
    log.info("opened tracking issue for DEFER: %s", url)
    return url


def maybe_react(
    comment_body: str,
    comment_id: int | str,
    comment_type: str,
    repo: str,
    config: Config,
    gh: GitHub,
    *,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> None:
    """Let Fido decide whether to react to a comment with an emoji.

    comment_type: 'pulls' for review comments, 'issues' for issue comments.
    """
    if agent is None:
        agent = _configured_agent(config, config.repos[repo])
    if prompts is None:
        prompts = Prompts(_load_persona(config))
    reaction = (
        agent.run_turn(prompts.react_prompt(comment_body), model=agent.voice_model)
        .lower()
        .split("\n")[0]
        .strip()
    )

    valid = {"+1", "-1", "laugh", "confused", "heart", "hooray", "rocket", "eyes"}
    if reaction not in valid:
        log.debug("fido chose not to react (got: %s)", reaction)
        return

    log.info("fido reacts with %s to comment %s", reaction, comment_id)
    try:
        gh.add_reaction(repo, comment_type, comment_id, reaction)
    except Exception:
        log.exception("failed to post reaction")


def reply_to_comment(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    *,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> tuple[str, list[str]]:
    """Triage a comment via Opus, generate a reply via Opus, post it.

    Returns (triage_category, task_titles).
    task_titles is a list: one entry for non-task categories (used as reply
    context), or one or more entries for ACT/DO (each becomes a task).
    Uses the caller's SQLite reply claim to prevent concurrent replies.
    Raises on reply-post failure so callers fail closed.
    """
    if agent is None:
        agent = _configured_agent(config, repo_cfg)
    if prompts is None:
        prompts = Prompts(_load_persona(config))
    info = action.reply_to
    if not info or not action.comment_body:
        return ("ACT", [action.comment_body or action.prompt])

    comment = action.comment_body

    context: dict[str, Any] = dict(action.context) if action.context else {}
    direct_promise = None
    if context.get("reply_promise_id") is None and isinstance(
        info.get("comment_id"), int
    ):
        direct_promise = FidoStore(repo_cfg.work_dir).prepare_reply(
            owner="worker",
            comment_type=str(info.get("comment_type", "pulls")),
            anchor_comment_id=int(info["comment_id"]),
        )
        if direct_promise is None:
            log.info("comment %s already claimed — skipping reply", info["comment_id"])
            return ("ACT", [])
        context["reply_promise_id"] = direct_promise.promise_id

    # Always fetch the full thread for this comment.
    thread_comments: list[dict[str, Any]] = []
    if info.get("repo") and info.get("pr") and info.get("comment_id"):
        fetched = gh.fetch_comment_thread(info["repo"], info["pr"], info["comment_id"])
        if fetched:
            thread_comments = list(fetched)
            context["comment_thread"] = thread_comments
            log.info(
                "fetched %d comment(s) in thread for context", len(thread_comments)
            )

    # Capture Fido reply IDs from the initial snapshot so we can detect new
    # replies posted by concurrent handlers during the triage + generation
    # window (compared against the re-fetched thread below).
    initial_fido_ids: set[int] = {
        c["id"] for c in thread_comments if c.get("author", "").lower() in _FIDO_LOGINS
    }

    final_comment_id = info.get("comment_id")

    # Enrich context with sibling threads when the comment needs more context
    if needs_more_context(comment, agent=agent) and info.get("repo") and info.get("pr"):
        siblings = gh.fetch_sibling_threads(info["repo"], info["pr"])
        if siblings:
            context["sibling_threads"] = siblings
            log.info(
                "needs-more-context comment — fetched %d sibling thread(s) for context",
                len(siblings),
            )

    # Step 1: Haiku triage (on the triggering comment to determine category)
    category, titles = _triage(
        comment, action.is_bot, context, agent=agent, prompts=prompts
    )
    log.info("triage: %s — %s", category, titles)

    # Step 1b: Derive task titles from the full comment chain for action
    # categories.  The final triggering comment is the ACT source, while prior
    # comments provide the context needed for terse follow-ups.
    if category in ("ACT", "DO"):
        log.info("deriving task title from comment chain")
        titles = [
            _comment_chain_action_title(
                thread_comments,
                final_comment_id if isinstance(final_comment_id, int) else None,
                comment,
                titles,
                agent=agent,
            )
        ]

    # Step 2: For DEFER, open a tracking issue before crafting the reply.
    # Raises on failure so we don't craft a reply referencing a missing issue.
    issue_url: str | None = None
    if category == "DEFER" and info.get("repo"):
        pr_url = f"https://github.com/{info['repo']}/pull/{info['pr']}"
        issue_url = _open_defer_issue(gh, info["repo"], pr_url, titles[0], comment)

    # Step 3: Opus reply based on triage
    instr = prompts.reply_instruction(
        category, comment, ", ".join(titles), context, issue_url=issue_url
    )

    log.info(
        "reply generator: requesting %s reply for PR #%s comment %s",
        category,
        info["pr"],
        info["comment_id"],
    )
    body = safe_voice_turn(
        agent,
        prompts.persona_wrap(instr),
        model=agent.voice_model,
        system_prompt=prompts.reply_system_prompt(),
        log_prefix="reply_to_comment",
    )
    log.info(
        "reply generator: returned %d chars (preview=%r)",
        len(body),
        body[:80],
    )

    # Re-fetch the thread right before posting so the edit-vs-post decision
    # uses current GitHub state rather than the snapshot taken before triage.
    # Concurrent handlers may have posted replies during the triage + generation
    # window; without a re-fetch the stale snapshot leads to duplicate posts.
    if info.get("repo") and info.get("pr") and info.get("comment_id"):
        refreshed = gh.fetch_comment_thread(
            info["repo"], info["pr"], info["comment_id"]
        )
        if refreshed:
            thread_comments = list(refreshed)
            log.info(
                "re-fetched %d comment(s) in thread before posting",
                len(thread_comments),
            )

    # Skip posting if a concurrent handler already replied to *this specific*
    # comment during triage.  A Fido reply with ``in_reply_to_id`` matching
    # this comment id, that wasn't in the initial snapshot, means another
    # handler handled it — adding a second reply would be a duplicate.
    #
    # The check used to count *any* new Fido reply in the thread, but that
    # false-positived on multi-inline-comment reviews: sibling handlers
    # running in parallel posted to *their own* comments, and the slowest
    # handler saw those siblings in the re-fetched snapshot, mistakenly
    # concluded "someone replied to MY comment", and silently dropped its
    # own reply while still ack'ing the promise.  Closes #1004; mirrors
    # the original double-post fix in #518.
    target_id = info.get("comment_id")
    current_fido_target_ids: set[int] = {
        c["id"]
        for c in thread_comments
        if c.get("author", "").lower() in _FIDO_LOGINS
        and c.get("in_reply_to_id") == target_id
    }
    if current_fido_target_ids - initial_fido_ids:
        log.info(
            "concurrent handler already replied — skipping post for comment %s",
            target_id,
        )
        if direct_promise is not None:
            FidoStore(repo_cfg.work_dir).ack_promise(direct_promise.promise_id)
        return (category, titles)

    promise_ids = _reply_promise_ids(context)
    body = append_reply_promise_markers(body, promise_ids)
    log.info("posting reply to PR #%s: %s", info["pr"], body[:80])
    posted = gh.reply_to_review_comment(
        info["repo"], info["pr"], body, info["comment_id"]
    )
    log.info("reply posted")
    root_comment_id = (
        thread_comments[0]["id"] if thread_comments else info["comment_id"]
    )
    if root_comment_id is not None:
        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=_posted_comment_id(posted),
            comment_type="pulls",
            lane_key=_review_lane_key(
                info["repo"], int(info["pr"]), int(root_comment_id)
            ),
            promise_ids=promise_ids,
        )
    if direct_promise is not None:
        FidoStore(repo_cfg.work_dir).ack_promise(direct_promise.promise_id)

    # Maybe react
    maybe_react(
        comment,
        info["comment_id"],
        "pulls",
        info.get("repo", ""),
        config,
        gh,
        agent=agent,
        prompts=prompts,
    )

    if review_outcome_resolves_thread(category, is_bot=action.is_bot) and info.get(
        "comment_id"
    ):
        _try_resolve_thread(info, gh)

    return (category, titles)


def _try_resolve_thread(info: dict[str, Any], gh: GitHub) -> None:
    """Best-effort resolve the review thread containing a comment."""
    repo = info.get("repo")
    pr = info.get("pr")
    comment_id = info.get("comment_id")
    if not isinstance(repo, str) or not isinstance(comment_id, int):
        return
    if pr is None:
        return
    if not isinstance(pr, int):
        try:
            pr = int(pr)
        except TypeError, ValueError:
            return
    owner, repo_name = repo.split("/", 1)
    for node in gh.get_review_threads(owner, repo_name, pr):
        if node.get("isResolved"):
            continue
        comments = node.get("comments", {}).get("nodes", [])
        if any(c.get("databaseId") == comment_id for c in comments):
            gh.resolve_thread(node["id"])
            return


def reply_to_review(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    *,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> None:
    """No-op for inline comments — they are handled per-comment.

    GitHub fires both ``pull_request_review`` (this handler) and a separate
    ``pull_request_review_comment`` for each inline comment within the
    review.  Iterating the inline comments here too caused two independent
    handlers to triage and post against the same comment in parallel — the
    per-comment lock didn't serialize them because triage takes the lock
    only after a long pre-flight, so both posted clarification replies on
    the same thread.  Closes #518.

    Inline comments are now exclusively handled by the per-comment
    webhook.  This handler is left as a stub so the dispatcher can still
    register it for the event type (and so future top-level review-body
    handling has a place to live).

    Note: the review's *top-level* body text (the box at the bottom of
    \"Submit review\") still arrives only through this event and is not
    yet handled.  Tracked separately — out of scope for the dedup fix.
    """
    _ = (action, config, repo_cfg, gh, agent, prompts)
    log.debug(
        "reply_to_review: skipping inline comments — handled per-comment (closes #518)"
    )


def needs_more_context(
    comment_body: str, *, agent: ProviderAgent | None = None
) -> bool:
    """Ask Haiku whether this comment needs sibling thread context to act on.

    Returns True if Haiku thinks the comment is too vague or cross-referential
    to act on alone (e.g. "same", "ditto", "^"), False otherwise.
    Falls back to False on any error.
    """
    if agent is None:
        raise ValueError("needs_more_context requires agent")
    prompt = (
        f"{NO_TOOLS_CLAUSE}\n\n"
        "A reviewer left this comment on a pull request:\n\n"
        f"{comment_body!r}\n\n"
        "Does this comment need context from sibling review threads to be understood "
        "(e.g. it says 'same', 'ditto', '^', 'here too', or is otherwise too vague "
        "to act on alone)?\n\n"
        "Reply with exactly YES or NO."
    )
    log.info("needs-more-context check: requesting haiku")
    answer = agent.run_turn(prompt, model=agent.brief_model).upper()
    log.info(
        "needs-more-context check: returned %d chars (answer=%r)",
        len(answer),
        answer[:20],
    )
    return answer.startswith("YES")


_MAX_TITLE_LEN = 80


def _shorten_title_if_needed(
    title: str, *, agent: ProviderAgent, log_prefix: str
) -> str:
    """Shorten an overlong task title while preserving imperative wording."""
    result = title
    for _ in range(3):
        if len(result) <= _MAX_TITLE_LEN:
            break
        log.info(
            "%s: title too long (%d chars), requesting shorten",
            log_prefix,
            len(result),
        )
        shortened = safe_voice_turn(
            agent,
            f"{NO_TOOLS_CLAUSE}\n\n"
            f"Shorten this task title to under {_MAX_TITLE_LEN} characters while keeping it imperative. "
            f"Reply with ONLY the shortened title.\n\nTitle: {result}",
            model=agent.voice_model,
            log_prefix=f"{log_prefix}/shorten",
        )
        result = shortened.strip()
        log.info(
            "%s: shorten returned %d chars (preview=%r)",
            log_prefix,
            len(result),
            result[:60],
        )
    return result[:_MAX_TITLE_LEN]


def _summarize_as_action_item(
    comment_body: str, *, agent: ProviderAgent | None = None
) -> str:
    """Ask Opus to convert a comment into a short imperative action-item title.

    If the result is too long, asks the provider agent to shorten it up to 3 times before
    falling back to hard truncation.
    """
    if agent is None:
        raise ValueError("_summarize_as_action_item requires agent")
    prompt = (
        f"{NO_TOOLS_CLAUSE}\n\n"
        "Convert this PR review comment into a short, imperative task title starting with a verb. "
        "Reply with ONLY the title — no category prefix, no punctuation at the end.\n\n"
        f"Comment: {comment_body}"
    )
    log.info("summarize-action-item: requesting initial title from opus")
    raw = safe_voice_turn(
        agent, prompt, model=agent.voice_model, log_prefix="_summarize_as_action_item"
    )
    result = raw.strip()
    log.info(
        "summarize-action-item: returned %d chars (preview=%r)",
        len(result),
        result[:60],
    )
    return _shorten_title_if_needed(
        result, agent=agent, log_prefix="_summarize_as_action_item"
    )


def _comment_chain_action_title(
    thread_comments: list[dict[str, Any]],
    final_comment_id: int | None,
    final_comment_body: str,
    triage_titles: list[str],
    *,
    agent: ProviderAgent | None = None,
) -> str:
    """Ask Opus for a task title using the whole comment chain."""
    if agent is None:
        raise ValueError("_comment_chain_action_title requires agent")
    chain = list(thread_comments)
    if final_comment_id is not None and not any(
        c.get("id") == final_comment_id for c in chain
    ):
        chain.append(
            {
                "id": final_comment_id,
                "author": "commenter",
                "body": final_comment_body,
            }
        )
    if not chain:
        chain.append(
            {
                "id": final_comment_id,
                "author": "commenter",
                "body": final_comment_body,
            }
        )
    lines: list[str] = []
    for index, item in enumerate(chain, start=1):
        marker = " (FINAL ACT COMMENT)" if item.get("id") == final_comment_id else ""
        user = item.get("user")
        user_login = user.get("login") if isinstance(user, dict) else None
        author = item.get("author") or user_login or "unknown"
        body = str(item.get("body", "") or "")
        lines.append(f"{index}. {author}{marker}: {body}")
    suggested = "\n".join(f"- {title}" for title in triage_titles if title.strip())
    prompt = (
        f"{NO_TOOLS_CLAUSE}\n\n"
        "Convert this PR review comment chain into a short, imperative task title "
        "starting with a verb. The comment marked FINAL ACT COMMENT is the comment "
        "that produced the ACT decision; use the earlier comments only as context. "
        "Preserve the concrete action from the final comment. Reply with ONLY the "
        "title — no category prefix, no punctuation at the end.\n\n"
        f"Suggested ACT title(s) from triage:\n{suggested or '- none'}\n\n"
        "Comment chain:\n" + "\n".join(lines)
    )
    log.info("comment-chain-action-title: requesting title from opus")
    raw = safe_voice_turn(
        agent, prompt, model=agent.voice_model, log_prefix="_comment_chain_action_title"
    )
    result = raw.strip()
    log.info(
        "comment-chain-action-title: returned %d chars (preview=%r)",
        len(result),
        result[:60],
    )
    return _shorten_title_if_needed(
        result, agent=agent, log_prefix="_comment_chain_action_title"
    )


def _triage(
    comment_body: str,
    is_bot: bool,
    context: dict[str, Any] | None = None,
    *,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> tuple[str, list[str]]:
    """Ask Opus to triage a comment. Returns (category, titles).

    A comment may produce zero or many tasks: titles is a list with one entry
    for ANSWER/ASK/DEFER/DUMP (used as reply context), or one or more entries
    for ACT/DO (each becomes a separate work-queue task).
    """
    if agent is None:
        raise ValueError("_triage requires agent")
    if prompts is None:
        prompts = Prompts("")
    prompt = prompts.triage_prompt(comment_body, is_bot, context)
    log.info("triage classifier: requesting category from opus")
    text = agent.run_turn(prompt, model=agent.voice_model)
    log.info(
        "triage classifier: returned %d chars (preview=%r)",
        len(text or ""),
        (text or "")[:80],
    )
    category: str | None = None
    titles: list[str] = []
    for line in text.splitlines() if text else []:
        if ":" not in line:
            continue
        prefix, title = line.split(":", 1)
        prefix = prefix.strip().upper()
        title = title.strip()
        if prefix not in ("ACT", "ASK", "ANSWER", "DO", "DEFER", "DUMP"):
            continue
        if category is None:
            category = prefix
        if prefix == category and title:
            titles.append(title)
    if category is not None and titles:
        return category, titles
    log.warning(
        "triage classifier: unparseable response, falling back to %s + summarize",
        "DO" if is_bot else "ACT",
    )
    # Fallback: ACT for humans, DO for bots; summarize comment into action item
    category = "DO" if is_bot else "ACT"
    title = _summarize_as_action_item(comment_body, agent=agent)
    return category, [title]


def reply_to_issue_comment(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    *,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> tuple[str, list[str]]:
    """Triage and reply to a top-level PR comment (issue_comment event).

    Raises on reply-post failure so callers fail closed.
    """
    if agent is None:
        agent = _configured_agent(config, repo_cfg)
    if prompts is None:
        prompts = Prompts(_load_persona(config))
    comment = action.comment_body or ""
    context = dict(action.context) if action.context else {}
    direct_promise = None
    comment_id = context.get("comment_id")
    if context.get("reply_promise_id") is None and isinstance(comment_id, int):
        direct_promise = FidoStore(repo_cfg.work_dir).prepare_reply(
            owner="worker",
            comment_type="issues",
            anchor_comment_id=comment_id,
        )
        if direct_promise is None:
            log.info("comment %s already claimed — skipping issue reply", comment_id)
            return ("ACT", [])
        context["reply_promise_id"] = direct_promise.promise_id

    # Extract PR number from prompt
    m = re.search(r"#(\d+)", action.prompt)
    number = m.group(1) if m else ""

    repo_full = gh.get_repo_info(cwd=repo_cfg.work_dir)

    # Fetch full conversation history for context
    conversation_context = ""
    if number:
        try:
            all_comments = gh.get_issue_comments(repo_cfg.name, int(number))
            preceding = [c for c in all_comments if c.get("body", "") != comment]
            if preceding:
                lines = [
                    f"{c.get('user', {}).get('login', '?')}: {c.get('body', '')}"
                    for c in preceding
                ]
                conversation_context = (
                    "\n\nFull conversation on this issue/PR:\n" + "\n".join(lines)
                )
        except Exception:
            log.warning(
                "reply_to_issue_comment: failed to fetch conversation history for PR #%s"
                " — proceeding without it",
                number,
                exc_info=True,
            )

    # Merge conversation context into triage context
    if conversation_context:
        context["conversation"] = conversation_context

    category, titles = _triage(
        comment,
        action.is_bot,
        context or None,
        agent=agent,
        prompts=prompts,
    )
    log.info("issue comment triage: %s — %s", category, titles)

    # For DEFER, open a tracking issue before crafting the reply.
    # Raises on failure so we don't craft a reply referencing a missing issue.
    issue_url: str | None = None
    if category == "DEFER":
        pr_url = f"https://github.com/{repo_full}/pull/{number}" if number else ""
        issue_url = _open_defer_issue(gh, repo_full, pr_url, titles[0], comment)

    instr = prompts.issue_reply_instruction(
        category, comment, ", ".join(titles), action.context, issue_url=issue_url
    )

    log.info("generating %s reply for issue comment on PR #%s", category, number)
    body = safe_voice_turn(
        agent,
        prompts.persona_wrap(instr),
        model=agent.voice_model,
        system_prompt=prompts.reply_system_prompt(),
        log_prefix="reply_to_issue_comment",
    )
    log.info(
        "reply generation returned for PR #%s — body_len=%d preview=%r",
        number,
        len(body),
        body[:80],
    )

    promise_ids = _reply_promise_ids(context)
    body = append_reply_promise_markers(body, promise_ids)
    log.info("posting issue comment reply on PR #%s: %s", number, body[:80])
    posted = gh.comment_issue(repo_full, number, body)
    log.info("reply posted on PR #%s", number)
    _record_reply_artifact(
        repo_cfg,
        artifact_comment_id=_posted_comment_id(posted),
        comment_type="issues",
        lane_key=_issue_lane_key(repo_full, int(number)),
        promise_ids=promise_ids,
    )
    if direct_promise is not None:
        FidoStore(repo_cfg.work_dir).ack_promise(direct_promise.promise_id)

    _cid = (action.context or {}).get("comment_id")
    if _cid:
        log.info(
            "reply_to_issue_comment: adding reaction on PR #%s comment %s", number, _cid
        )
        maybe_react(
            comment,
            _cid,
            "issues",
            repo_full,
            config,
            gh,
            agent=agent,
            prompts=prompts,
        )
        log.info(
            "reply_to_issue_comment: reaction path done on PR #%s comment %s",
            number,
            _cid,
        )

    log.info(
        "reply_to_issue_comment: complete for PR #%s (category=%s)", number, category
    )
    return (category, titles)


_TYPE_PRIORITY = {TaskType.CI: 0, TaskType.THREAD: 1, TaskType.SPEC: 2}


def _maybe_abort_for_new_task(
    repo_cfg: RepoConfig,
    new_task: dict[str, Any],
    registry: WorkerRegistry,
    *,
    _state: Any = None,
    _tasks: Any = None,
) -> None:
    """Abort the current task if the new task has higher priority.

    Priority is deterministic by type: ci > thread > spec.
    A higher-priority task always preempts — the current task is kept
    pending for later (ABORT_KEEP).  Equal or lower priority does not
    preempt.
    """
    from fido.state import State
    from fido.tasks import Tasks

    if _state is None:
        _state = State(repo_cfg.work_dir / ".git" / "fido")
    if _tasks is None:
        _tasks = Tasks(repo_cfg.work_dir)

    state = _state.load()
    current_task_id = state.get("current_task_id")
    if not current_task_id:
        return

    task_list = _tasks.list()
    current_task = next((t for t in task_list if t["id"] == current_task_id), None)
    if current_task is None:
        return

    new_priority = _TYPE_PRIORITY.get(new_task.get("type", "spec"), 2)
    current_priority = _TYPE_PRIORITY.get(current_task.get("type", "spec"), 2)

    if new_priority < current_priority:
        log.info(
            "preempt: %s task interrupts %s task — aborting %s",
            new_task.get("type", "?"),
            current_task.get("type", "?"),
            current_task.get("title", "")[:60],
        )
        registry.abort_task(repo_cfg.name)


def _get_commit_summary(work_dir: Path) -> str:
    """Return a short ``git log --oneline`` summary of recent commits.

    Best-effort enrichment: used to give Opus context about what has already
    been implemented when it reorders the pending task list.  Returns an empty
    string on nonzero exit, subprocess error, or missing git binary — callers
    must not treat the result as authoritative.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except subprocess.SubprocessError, OSError:
        return ""


def _notify_thread_change(
    change: dict[str, Any],
    config: Config,
    gh: GitHub,
    *,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> None:
    """Post a brief comment notifying a commenter that their task was rescoped.

    Called for each thread task that was dropped or modified during dependency
    analysis.  Uses Opus (in Fido's voice) to generate the message.

    Only fires for review comments (comment_type='pulls'), where it replies
    in-thread via the pull review comment API.  Issue comments
    (comment_type='issues') are skipped: the webhook handler already posted a
    triage reply to the original comment, and a second notification here would
    be a duplicate top-level issue comment.
    """
    task = change["task"]
    thread = task.get("thread") or {}
    comment_id = thread.get("comment_id")
    repo = thread.get("repo", "")
    pr = thread.get("pr")
    url = thread.get("url", "")
    author = thread.get("author", "")
    comment_type = thread.get("comment_type", "issues")
    if not (comment_id and repo and pr):
        return

    # Issue comments already received a triage reply from the webhook handler.
    # Posting again here would produce a duplicate top-level PR comment.
    if comment_type != "pulls":
        log.info(
            "skipping rescope notification for issue comment %s"
            " (webhook already replied)",
            comment_id,
        )
        return

    if agent is None:
        agent = _configured_agent(
            config, config.repos[change["task"]["thread"]["repo"]]
        )
    if prompts is None:
        prompts = Prompts(_load_persona(config))

    kind = change["kind"]
    original_title = task.get("title", "")

    if kind == "completed":
        instruction = (
            f"A task originating from a PR comment has been marked done — it was "
            f"covered by work already committed and is no longer in the active queue.\n\n"
            f"Original task: {original_title}\n"
            f"Comment author: {author or '(unknown)'}\n"
            f"Comment: {url}\n\n"
            "Write a very brief reply notifying the commenter that their task has been "
            "marked done because it was covered by recent commits. Reference the comment URL."
        )
    else:
        new_title = change.get("new_title", "")
        instruction = (
            f"The task you were planning from a PR comment has been updated to "
            f"reflect new requirements.\n\n"
            f"Original task: {original_title}\n"
            f"Updated task: {new_title}\n"
            f"Comment author: {author or '(unknown)'}\n"
            f"Comment: {url}\n\n"
            "Write a very brief reply notifying the commenter that their original task "
            "has been updated. Reference the comment URL."
        )

    body = safe_voice_turn(
        agent,
        prompts.persona_wrap(instruction),
        model=agent.voice_model,
        system_prompt=prompts.reply_system_prompt(),
        log_prefix="_notify_thread_change",
    )
    try:
        gh.reply_to_review_comment(repo, pr, body, comment_id)
        log.info("notified thread %s (%s)", comment_id, kind)
    except Exception:
        log.exception("failed to notify thread %s", comment_id)


def _task_snapshot(task_list: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Summarise task_list as an ordered list of (id, status, title) tuples.

    Used by :func:`_rewrite_pr_description` to detect whether the task list
    changed while Opus was generating the PR description.
    """
    return [(t["id"], t.get("status", ""), t.get("title", "")) for t in task_list]


def _rewrite_pr_description(
    work_dir: Path,
    gh: Any,
    *,
    agent: ProviderAgent | None = None,
    _state: Any = None,
    _tasks: Any = None,
    _max_retries: int = 3,
) -> None:
    """Rewrite the PR description summary after a successful rescope.

    Delegates to :func:`fido.worker._write_pr_description` so that initial
    PR creation and post-rescope rewrites share one code path.

    Silently skips when there is no active issue or no open PR for it.
    All other errors (missing ``---`` divider, empty Opus output, GitHub API
    failures) propagate so the caller's thread excepthook can surface them.

    Retries up to *_max_retries* times when the task list changes while Opus
    is generating the description, so the written description always reflects
    the state of the task list at the moment Opus returned.  The PR body is
    re-fetched on each retry so the work-queue section stays current.
    """
    from fido.worker import (
        _write_pr_description,  # pyright: ignore[reportPrivateUsage]
    )

    if _state is None or _tasks is None:
        from fido.state import State as StateStore
        from fido.tasks import Tasks as TaskStore

        if _state is None:
            _state = StateStore(work_dir / ".git" / "fido")
        if _tasks is None:
            _tasks = TaskStore(work_dir)

    state = _state.load()
    issue = state.get("issue")
    if not issue:
        log.info("_rewrite_pr_description: no active issue in state — skipping")
        return

    repo = gh.get_repo_info(cwd=work_dir)
    user = gh.get_user()

    pr_data = gh.find_pr(repo, issue, user)
    if pr_data is None or pr_data.get("state") != "OPEN":
        log.info("_rewrite_pr_description: no open PR for issue #%s — skipping", issue)
        return

    pr_number = pr_data["number"]

    for attempt in range(_max_retries):
        task_list = _tasks.list()
        snapshot_before = _task_snapshot(task_list)

        body = gh.get_pr_body(repo, pr_number)
        _write_pr_description(
            work_dir,
            gh,
            repo,
            pr_number,
            issue,
            task_list,
            body,
            agent=agent,
        )

        snapshot_after = _task_snapshot(_tasks.list())
        if snapshot_after == snapshot_before:
            return

        log.info(
            "_rewrite_pr_description: task list changed during rewrite — retrying"
            " (attempt %d/%d)",
            attempt + 1,
            _max_retries,
        )

    log.warning(
        "_rewrite_pr_description: task list still changing after %d attempts"
        " — description may be slightly stale",
        _max_retries,
    )


def _make_reorder_kwargs(
    work_dir: Path,
    config: Config,
    repo_cfg: RepoConfig | None,
    registry: WorkerRegistry | None,
    gh: Any,
    agent: ProviderAgent,
    prompts: Prompts,
    rewrite_fn: Callable[..., None],
    sync_fn: Callable[[Path, Any], None] | None = None,
) -> dict[str, Any]:
    """Build the kwargs dict for a :func:`~fido.tasks.reorder_tasks` call."""

    def on_changes(changes: list[dict[str, Any]]) -> None:
        for change in changes:
            _notify_thread_change(change, config, gh, agent=None, prompts=None)

    def on_done() -> None:
        if sync_fn is None:
            from fido.tasks import sync_tasks

            sync_tasks(work_dir, gh)
        else:
            sync_fn(work_dir, gh)
        rewrite_fn(
            work_dir,
            gh,
            agent=agent,
            _state=State(work_dir / ".git" / "fido"),
            _tasks=Tasks(work_dir),
        )

    kwargs: dict[str, Any] = {
        "_on_changes": on_changes,
        "_on_done": on_done,
        "agent": agent,
        "prompts": prompts,
    }
    if registry is not None and repo_cfg is not None:

        def on_inprogress_affected() -> None:
            log.info(
                "reorder_tasks_background: in-progress task affected — aborting %s",
                repo_cfg.name,
            )
            registry.abort_task(repo_cfg.name)

        kwargs["_on_inprogress_affected"] = on_inprogress_affected
    return kwargs


def _reorder_tasks_background(
    work_dir: Path,
    commit_summary: str,
    config: Config,
    gh: GitHub,
    repo_cfg: RepoConfig | None = None,
    registry: WorkerRegistry | None = None,
    *,
    _start: Callable[[threading.Thread], None] = threading.Thread.start,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
    _rewrite_fn: Callable[..., None] | None = None,
    _reorder_fn: Callable[..., None] | None = None,
    _sync_fn: Callable[[Path, Any], None] | None = None,
    _coalesce_state: dict[str, Any] | None = None,
    _release_untriaged_on_finish: bool = False,
) -> None:
    """Run :func:`~fido.tasks.reorder_tasks` in a daemon background thread.

    Coalesces concurrent calls: if a reorder thread is already running for
    *work_dir*, the new trigger is recorded as pending rather than spawning
    another thread.  When the running thread finishes it checks for a pending
    run and, if one exists, executes it before exiting — so at most one Opus
    call is in-flight plus one queued per repo.

    Passes an ``_on_changes`` callback so that any thread tasks dropped or
    modified during rescoping trigger a notification reply to the original
    comment.

    If *repo_cfg* and *registry* are provided, also passes an
    ``_on_inprogress_affected`` callback that aborts the running worker whenever
    the in-progress task is dropped or modified by the rescope, so the worker
    loop restarts on the new next task.

    Passes an ``_on_done`` callback that rewrites the PR description after a
    successful reorder, so the human-facing summary stays in sync with the
    updated plan.
    """
    from fido.tasks import reorder_tasks as _reorder_tasks

    reorder = _reorder_fn if _reorder_fn is not None else _reorder_tasks
    rewrite_fn = _rewrite_fn if _rewrite_fn is not None else _rewrite_pr_description
    state = _coalesce_state if _coalesce_state is not None else _reorder_coalesce
    if agent is None:
        agent = (
            _configured_agent(config, repo_cfg)
            if repo_cfg is not None
            else ClaudeClient()
        )
    if prompts is None:
        prompts = Prompts(_load_persona(config))

    key = str(work_dir)
    kwargs = _make_reorder_kwargs(
        work_dir, config, repo_cfg, registry, gh, agent, prompts, rewrite_fn, _sync_fn
    )

    with _reorder_coalesce_lock:
        entry = state.setdefault(
            key, {"running": False, "pending": None, "untriaged_holds": 0}
        )
        if _release_untriaged_on_finish:
            entry["untriaged_holds"] = int(entry.get("untriaged_holds", 0)) + 1
        if entry["running"]:
            # Coalesce: latest call wins; the running thread will do one more pass.
            entry["pending"] = (commit_summary, kwargs)
            return
        entry["running"] = True
        entry["pending"] = None

    def run_loop() -> None:
        cs = commit_summary
        kw = kwargs
        release_untriaged = 0
        # Register as "webhook" so the session talker reflects the true nature of
        # this thread: it is triggered by webhooks and should not be treated as the
        # worker for preemption purposes.  Without this, current_thread_kind()
        # defaults to "worker", causing real webhooks to fire _fire_worker_cancel
        # against the reorder thread and misidentify it as the running worker (#955).
        set_thread_kind("webhook")
        if repo_cfg is not None:
            set_thread_repo(repo_cfg.name)
        if registry is not None and repo_cfg is not None:
            registry.set_rescoping(repo_cfg.name, True)
        try:
            while True:
                reorder(work_dir, cs, **kw)
                with _reorder_coalesce_lock:
                    pending = state[key].get("pending")
                    if pending is None:
                        break
                    state[key]["pending"] = None
                    cs, kw = pending
        finally:
            with _reorder_coalesce_lock:
                entry = state.get(key)
                if entry is not None:
                    release_untriaged += int(entry.get("untriaged_holds", 0))
                    entry["untriaged_holds"] = 0
                    entry["running"] = False
            if registry is not None and repo_cfg is not None:
                registry.set_rescoping(repo_cfg.name, False)
                for _ in range(release_untriaged):
                    registry.exit_untriaged(repo_cfg.name)
            if repo_cfg is not None:
                set_thread_repo(None)
            set_thread_kind(None)

    t = threading.Thread(
        target=run_loop,
        name=f"reorder-{work_dir.name}",
        daemon=True,
    )
    try:
        _start(t)
    except Exception:
        release_untriaged = 0
        with _reorder_coalesce_lock:
            entry = state.get(key)
            if entry is not None:
                release_untriaged = int(entry.get("untriaged_holds", 0))
                entry["untriaged_holds"] = 0
                entry["running"] = False
                entry["pending"] = None
        if registry is not None and repo_cfg is not None:
            registry.set_rescoping(repo_cfg.name, False)
            for _ in range(release_untriaged):
                registry.exit_untriaged(repo_cfg.name)
        raise


def _resolved_thread_comment_author_for_oracle(
    login: str,
    owner: str,
    collaborators: frozenset[str],
    allowed_bots: frozenset[str],
) -> thread_resolve_oracle.ThreadCommentAuthor:
    return thread_comment_author_for_auto_resolve_oracle(
        login,
        fido_logins=_FIDO_LOGINS,
        owner=owner,
        collaborators=collaborators,
        allowed_bots=allowed_bots,
    )


def _thread_task_is_stale_resolved(
    gh: GitHub,
    thread: dict[str, Any],
    *,
    collaborators: frozenset[str] = frozenset(),
    allowed_bots: frozenset[str] = frozenset(),
) -> bool:
    """Return whether a resolved-thread task request is stale duplicate work.

    Resolved-thread suppression exists for late handlers racing with Fido's
    auto-resolve path. A new human reply on that same resolved thread is not
    stale: it is fresh input that should queue work without requiring the human
    to manually unresolve the GitHub thread.
    """
    comment_id = thread.get("comment_id")
    if comment_id is None:
        return True
    owner, _repo_name = str(thread["repo"]).split("/", 1)
    comments = gh.fetch_comment_thread(thread["repo"], thread["pr"], int(comment_id))
    if not comments:
        return True
    oracle_comments = [
        thread_resolve_oracle.ThreadComment(
            thread_comment_id=int(comment["id"]),
            thread_comment_author=_resolved_thread_comment_author_for_oracle(
                str(comment.get("author", "")),
                owner,
                collaborators,
                allowed_bots,
            ),
        )
        for comment in comments
    ]
    decision = thread_resolve_oracle.resolved_thread_queue_decision(
        thread_resolve_oracle.ReviewThread(
            review_thread_resolved=True,
            review_thread_comments=oracle_comments,
        ),
        int(comment_id),
    )
    return isinstance(decision, thread_resolve_oracle.DismissStaleResolvedThread)


def create_task(
    prompt: str,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    thread: dict[str, Any] | None = None,
    registry: WorkerRegistry | None = None,
    *,
    _get_commit_summary_fn: Callable[[Path], str] = _get_commit_summary,
    _reorder_background_fn: Callable[..., None] = _reorder_tasks_background,
    _tasks: Tasks | None = None,
) -> dict[str, Any]:
    """Write a task to the shared task file, then trigger sync.

    PR comment tasks (those with a thread) are added to the task list at the
    position determined by the rescoping reorder — they receive spec-level
    priority (first in list wins among non-CI tasks).

    When *thread* is set (a PR comment task), also triggers a background
    dependency-analysis reorder via Opus so that remaining spec tasks are
    resequenced to account for the new requirement.  Spec tasks created during
    initial setup are not rescoped — the planner already ordered them.

    If *registry* is given, checks whether the new task has higher priority
    than the current in-progress task; if so, signals abort so the worker
    picks up the higher-priority task.

    Returns the new task dict.
    """
    if _tasks is None:
        _tasks = Tasks(repo_cfg.work_dir)
    # Race guard for thread tasks (#520): if the originating review thread
    # has already been resolved on GitHub (most often because fido completed
    # an earlier task in the same thread and auto-resolved it before this
    # late-arriving triage queued the new task), don't queue.  Without this
    # check the worker re-does work that's already shipped (#521), or
    # rejects the resolved-thread state and reopens it.
    if thread and thread.get("repo") and thread.get("pr") and thread.get("comment_id"):
        try:
            already = gh.is_thread_resolved_for_comment(
                thread["repo"], int(thread["pr"]), int(thread["comment_id"])
            )
        except Exception:
            log.exception(
                "create_task: thread-resolved check failed for comment %s; queuing anyway",
                thread.get("comment_id"),
            )
            already = False
        # Strict ``is True`` so MagicMock test gh stubs (whose method calls
        # return another MagicMock — truthy by default) don't cause this
        # guard to swallow every test-level task creation.  Real GitHub
        # always returns a real bool from ``is_thread_resolved_for_comment``.
        if already is True and _thread_task_is_stale_resolved(
            gh,
            thread,
            collaborators=repo_cfg.membership.collaborators,
            allowed_bots=config.allowed_bots,
        ):
            log.info(
                "create_task: thread for comment %s already resolved on GitHub — "
                "skipping queue (closes #520)",
                thread["comment_id"],
            )
            # Return a synthetic task-shaped dict so callers that don't check
            # status don't choke; callers that DO walk tasks.json won't see it.
            return {
                "title": prompt,
                "type": (TaskType.THREAD if thread else TaskType.SPEC).value,
                "status": "skipped_resolved",
                "thread": thread,
            }
        if already is True:
            log.info(
                "create_task: thread for comment %s is resolved, but comment is "
                "fresh human input — queueing task",
                thread["comment_id"],
            )
    task_type = TaskType.THREAD if thread else TaskType.SPEC
    log.info("creating task: %s", prompt[:100])
    new_task = _tasks.add(title=prompt, task_type=task_type, thread=thread)
    launch_sync(config, repo_cfg, gh)
    if thread:
        commit_summary = _get_commit_summary_fn(repo_cfg.work_dir)
        if registry is not None and _reorder_background_fn is _reorder_tasks_background:
            registry.enter_untriaged(repo_cfg.name)
            try:
                _reorder_background_fn(
                    repo_cfg.work_dir,
                    commit_summary,
                    config,
                    gh,
                    repo_cfg,
                    registry,
                    _release_untriaged_on_finish=True,
                )
            except Exception:
                registry.exit_untriaged(repo_cfg.name)
                raise
        else:
            _reorder_background_fn(
                repo_cfg.work_dir, commit_summary, config, gh, repo_cfg, registry
            )
    if registry is not None:
        _maybe_abort_for_new_task(repo_cfg, new_task, registry)
    return new_task


def backfill_missed_pr_comments(
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    pr_number: int,
    *,
    gh_user: str,
) -> int:
    """Replay ``issue_comment`` webhooks we may have missed while fido was
    down (fix for #794).  Returns the number of comments inspected.

    Scope is narrow by design: only **top-level PR comments** are replayed.
    Inline review comments and review threads are already scanned each
    iteration by ``Worker.handle_threads``, so the worker loop backfills
    those on its own — only issue-comments are invisible to the loop.

    Idempotent: comments with a completed SQLite claim are skipped — Fido
    already replied to them.  Comments handled with a task (ACT/DO) are additionally
    deduped by :func:`create_task` via ``comment_id`` in ``tasks.json``.
    This function is intended to run **once per WorkerThread lifetime** (at
    startup) — not every iteration.
    """
    log.info("backfill: scanning PR #%s for missed top-level comments", pr_number)
    comments = gh.get_issue_comments(repo_cfg.name, pr_number)
    for c in comments:
        user = (c.get("user") or {}).get("login", "")
        if not user:
            continue
        if user.lower() == gh_user.lower():
            continue
        if user.lower() in ("fidocancode", "fido-can-code"):
            continue
        if not _is_allowed(user, repo_cfg, config):
            continue
        comment_id = c.get("id")
        if comment_id is None:
            continue
        # Skip comments Fido already claimed or completed in the SQLite store.
        if FidoStore(repo_cfg.work_dir).is_claimed_or_completed(int(comment_id)):
            log.info("backfill: comment %s already claimed — skipping", comment_id)
            continue
        body = c.get("body", "") or ""
        is_bot = user.endswith("[bot]")
        thread = {
            "repo": repo_cfg.name,
            "pr": pr_number,
            "comment_id": comment_id,
            "url": c.get("html_url", ""),
            "author": user,
            "comment_type": "issues",
        }
        prompt = (
            f"PR top-level comment on #{pr_number} by {user} "
            f"({'bot' if is_bot else 'human/owner'}):\n\n{body}"
        )
        create_task(prompt, config, repo_cfg, gh, thread=thread)
    log.info("backfill: PR #%s — inspected %d comments", pr_number, len(comments))
    return len(comments)


def launch_sync(config: Config, repo_cfg: RepoConfig, gh: GitHub) -> None:
    """Sync tasks.json → PR body in a background thread."""
    from fido.tasks import sync_tasks_background

    sync_tasks_background(repo_cfg.work_dir, gh)
    log.info("sync-tasks launched")


def launch_worker(repo_cfg: RepoConfig, registry: WorkerRegistry) -> None:
    """Wake the per-repo WorkerThread via the registry."""
    log.info("waking worker thread for %s", repo_cfg.name)
    registry.wake(repo_cfg.name)
