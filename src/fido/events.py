import json
import logging
import re
import subprocess
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from fido.rocq import reply_outbox_protocol as reply_outbox_oracle
from fido.rocq import thread_auto_resolve as thread_resolve_oracle
from fido.rocq import webhook_command_translation as wct_oracle
from fido.rocq import webhook_ingress_dedupe as ingress_fsm
from fido.state import State
from fido.store import (
    FidoStore,
    ReplyOutboxEffectRecord,
    ReplyPromiseRecord,
    append_reply_promise_markers,
)
from fido.synthesis import Insight
from fido.synthesis_call import (
    SynthesisExhaustedError,
    call_failure_explanation,
    call_synthesis,
)
from fido.synthesis_executor import CommentTarget, SynthesisExecutor
from fido.tasks import Tasks, thread_comment_author_for_auto_resolve_oracle
from fido.types import ActiveIssue, ActivePR, RescopeIntent, TaskType
from fido.worker import ActivityReporter

log = logging.getLogger(__name__)

_FIDO_LOGINS = frozenset({"fidocancode", "fido-can-code"})


class _BackgroundRescopeTrigger:
    """RescopeTrigger implementation backed by :func:`_reorder_tasks_background`.

    Constructed in :func:`reply_to_comment` and :func:`reply_to_issue_comment`
    so :class:`~fido.synthesis_executor.SynthesisExecutor` can trigger a
    background rescope without needing direct access to the reorder machinery.

    Preempt-always semantics (#1230): a comment never directly preempts the
    current task — it always registers its intent via the rescope trigger,
    and the rescope decides whether the in-progress task needs to be
    preempted, removed, modified, or left alone.  Triage is fast and
    informed but not work-heavy; the worker thread owns the actual task
    state mutations after the rescope reducer commits.
    """

    def __init__(
        self,
        work_dir: Path,
        config: Config,
        gh: GitHub,
        repo_cfg: RepoConfig,
        registry: ActivityReporter,
        agent: ProviderAgent | None = None,
        prompts: Prompts | None = None,
    ) -> None:
        self._work_dir = work_dir
        self._config = config
        self._gh = gh
        self._repo_cfg = repo_cfg
        self._registry = registry
        self._agent = agent
        self._prompts = prompts

    def trigger_rescope(self, intent: RescopeIntent) -> None:
        """Trigger :func:`_reorder_tasks_background` with the given rescope intent.

        The intent's *change_request* text is logged for traceability and used
        as the commit summary passed to the reorder call so Opus can see what
        changed.  The full intent (including *comment_id* and *timestamp*) is
        forwarded so the rescoper can reference the originating comment and
        accumulate multiple concurrent intents correctly.
        """
        log.info(
            "RescopeTrigger: triggering background rescope for comment %d: %s",
            intent.comment_id,
            intent.change_request[:80],
        )
        _reorder_tasks_background(
            self._work_dir,
            intent.change_request,
            self._config,
            self._gh,
            self._repo_cfg,
            self._registry,
            agent=self._agent,
            prompts=self._prompts,
            intents=[intent],
        )


_INSIGHT_REPO = "FidoCanCode/home"
_INSIGHT_LABEL = "Insight"


class _GitHubInsightFiler:
    """InsightFiler implementation that creates GitHub issues on :data:`_INSIGHT_REPO`.

    Idempotent: before filing, searches for an existing issue carrying the
    ``<!-- insight-source: {comment_id} -->`` marker in its body.  If one is
    found the filing is skipped so replaying the same comment never creates
    duplicate insights.
    """

    def __init__(self, gh: GitHub) -> None:
        self._gh = gh

    def file_insight(self, insight: Insight, target: CommentTarget) -> None:
        """File *insight* as a GitHub issue against :data:`_INSIGHT_REPO`.

        Skips creation if an issue with the idempotency marker for
        *target.comment_id* already exists.
        """
        marker = f"<!-- insight-source: {target.comment_id} -->"
        existing = self._gh.search_issues(_INSIGHT_REPO, f'"{marker}" in:body is:issue')
        if existing:
            log.info(
                "insight already filed for comment %d — skipping: %s",
                target.comment_id,
                existing[0].get("html_url", ""),
            )
            return
        source_link = _insight_source_link(target)
        title = f"Insight: {insight.title}"
        body = f"{insight.hook}\n\n{insight.why}\n\nSource: {source_link}\n\n{marker}"
        url = self._gh.create_issue(_INSIGHT_REPO, title, body, labels=[_INSIGHT_LABEL])
        log.info("filed insight issue for comment %d: %s", target.comment_id, url)


def _insight_source_link(target: CommentTarget) -> str:
    """Return a GitHub URL pointing to the originating comment.

    Uses the ``discussion_r{comment_id}`` anchor for review comments and
    ``issuecomment-{comment_id}`` for top-level issue/PR comments.
    """
    if target.comment_type == "pulls":
        return (
            f"https://github.com/{target.repo}/pull/{target.pr}"
            f"#discussion_r{target.comment_id}"
        )
    return (
        f"https://github.com/{target.repo}/issues/{target.pr}"
        f"#issuecomment-{target.comment_id}"
    )


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
            Used for ``pull_request_review / submitted`` events with
            ``review.state == "commented"`` — inline comments are handled
            individually per ``pull_request_review_comment`` event, so the
            review-level event is collapsed rather than dispatched.  Decisive
            states (``approved``, ``changes_requested``, ``dismissed``) are
            **not** collapsed so the worker wakes immediately.

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
    lineage_key, lineage_comment_ids = _review_lineage(repo, pr_number, comment)
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
            "lineage_key": lineage_key,
            "lineage_comment_ids": list(lineage_comment_ids),
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
    lineage_key = _issue_lane_key(repo, pr_number)
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
            "lineage_key": lineage_key,
            "lineage_comment_ids": [comment_id],
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
    store.record_reply_delivery(
        artifact_comment_id=artifact_comment_id,
        comment_type=comment_type,
        lane_key=lane_key,
        promise_ids=promise_ids,
    )
    _assert_reply_outbox_posts_visible_artifact(
        store,
        artifact_comment_id=artifact_comment_id,
        promise_ids=promise_ids,
    )


def _claim_reply_outbox_effects(
    repo_cfg: RepoConfig,
    *,
    delivery_id: str,
    promise_ids: Iterable[str],
) -> None:
    """Durably claim every visible-reply outbox effect before posting."""
    store = FidoStore(repo_cfg.work_dir)
    for promise_id in promise_ids:
        promise = store.promise(promise_id)
        assert promise is not None, (
            f"reply_outbox_protocol: missing promise {promise_id!r} for visible reply"
        )
        existing = store.reply_outbox_effect(promise.promise_id)
        if existing is not None and existing.state == "claimed":
            raise RuntimeError(
                "reply_outbox_protocol: visible reply outbox effect already claimed"
            )
        if existing is not None and existing.state == "delivered":
            raise RuntimeError(
                "reply_outbox_protocol: delivered visible reply missing artifact row"
            )
        effect = store.claim_reply_outbox_effect(
            promise_id=promise.promise_id,
            delivery_id=delivery_id,
            origin_id=promise.anchor_comment_id,
        )
        _assert_reply_outbox_claimed_visible_effect(promise, effect)


def _reply_outbox_delivery_key(
    context: dict[str, Any], *, comment_type: str, comment_id: object
) -> str:
    delivery_id = context.get("delivery_id")
    if isinstance(delivery_id, str) and delivery_id:
        return delivery_id
    return f"{comment_type}:{comment_id}"


def _existing_reply_artifact(
    repo_cfg: RepoConfig, promise_ids: Iterable[str]
) -> int | None:
    """Return a durable reply artifact when every promise is already covered."""
    store = FidoStore(repo_cfg.work_dir)
    normalized = tuple(
        dict.fromkeys(promise_id for promise_id in promise_ids if promise_id)
    )
    if not normalized:
        return None
    artifacts = [store.artifact_for_promise(promise_id) for promise_id in normalized]
    if any(artifact is None for artifact in artifacts):
        return None
    artifact_ids = {artifact.artifact_comment_id for artifact in artifacts if artifact}
    if len(artifact_ids) != 1:
        return None
    artifact_id = next(iter(artifact_ids))
    _assert_reply_outbox_reuses_visible_artifact(
        store,
        artifact_comment_id=artifact_id,
        promise_ids=normalized,
    )
    for promise_id in normalized:
        store.mark_posted(promise_id)
    return artifact_id


def _reply_outbox_positive_id(value: str) -> int:
    """Map a durable UUID row id into the positive domain used by D14."""
    return uuid.UUID(value).int + 1


def _reply_outbox_text_positive_id(value: str) -> int:
    """Map a stable text key into the positive domain used by D14."""
    return uuid.uuid5(uuid.NAMESPACE_URL, value).int + 1


def _reply_outbox_delivery_id(value: str) -> int:
    """Map the persisted GitHub delivery id into the positive D14 domain."""
    return _reply_outbox_text_positive_id(value)


def _reply_outbox_origin_kind(comment_type: str) -> reply_outbox_oracle.OriginKind:
    origin_kinds = {
        "issues": reply_outbox_oracle.IssueCommentOrigin,
        "pulls": reply_outbox_oracle.ReviewThreadOrigin,
    }
    return origin_kinds[comment_type]()


def _reply_outbox_prepared_state(
    promise: ReplyPromiseRecord, effect: ReplyOutboxEffectRecord | None = None
) -> tuple[reply_outbox_oracle.ProtocolState, int, int, int]:
    promise_key = _reply_outbox_positive_id(promise.promise_id)
    origin = int(promise.anchor_comment_id)
    delivery = (
        origin if effect is None else _reply_outbox_delivery_id(effect.delivery_id)
    )
    effect_id = (
        promise_key if effect is None else _reply_outbox_positive_id(effect.effect_id)
    )
    state = reply_outbox_oracle.prepare_reply(
        delivery,
        origin,
        promise_key,
        effect_id,
        _reply_outbox_origin_kind(promise.comment_type),
        reply_outbox_oracle.empty_protocol_state,
    )
    assert state is not None, (
        "reply_outbox_protocol: prepare_reply rejected existing durable promise"
    )
    return state, origin, promise_key, effect_id


def _assert_reply_outbox_posts_visible_artifact(
    store: FidoStore,
    *,
    artifact_comment_id: int,
    promise_ids: Iterable[str],
) -> None:
    """Crash if Python records a visible reply outside the D14 protocol."""
    for promise_id in promise_ids:
        promise = store.promise(promise_id)
        effect = store.reply_outbox_effect(promise_id)
        assert promise is not None, (
            f"reply_outbox_protocol: missing promise {promise_id!r} for visible reply"
        )
        assert effect is not None and effect.external_id == artifact_comment_id, (
            "reply_outbox_protocol: missing delivered visible-reply outbox effect"
        )
        posted, origin, _effect_id = _reply_outbox_recorded_visible_state(
            promise,
            effect,
            label="visible reply",
        )
        assert (
            reply_outbox_oracle.live_reply_for_origin(posted, origin)
            == artifact_comment_id
        ), "reply_outbox_protocol: reply_post_is_idempotent violated"


def _assert_reply_outbox_reuses_visible_artifact(
    store: FidoStore,
    *,
    artifact_comment_id: int,
    promise_ids: Iterable[str],
) -> None:
    """Crash if Python reuses a visible reply not accepted by the D14 oracle."""
    for promise_id in promise_ids:
        promise = store.promise(promise_id)
        artifact = store.artifact_for_promise(promise_id)
        effect = store.reply_outbox_effect(promise_id)
        assert promise is not None and artifact is not None and effect is not None, (
            f"reply_outbox_protocol: missing durable rows for promise {promise_id!r}"
        )
        assert effect.external_id == artifact_comment_id, (
            "reply_outbox_protocol: visible reply outbox split from artifact"
        )
        assert artifact.artifact_comment_id == artifact_comment_id, (
            "reply_outbox_protocol: visible reply artifact split across promises"
        )
        posted, _origin, effect_id = _reply_outbox_recorded_visible_state(
            promise,
            effect,
            label="artifact reuse",
        )
        replayed = reply_outbox_oracle.record_reply_posted(
            effect_id,
            int(artifact_comment_id) + 1,
            posted,
        )
        assert replayed == posted and isinstance(
            reply_outbox_oracle.outbox_decision(posted, effect_id),
            reply_outbox_oracle.ReuseDeliveredEffect,
        ), "reply_outbox_protocol: reply_post_is_idempotent violated"


def _reply_outbox_recorded_visible_state(
    promise: ReplyPromiseRecord,
    effect: ReplyOutboxEffectRecord,
    *,
    label: str,
) -> tuple[reply_outbox_oracle.ProtocolState, int, int]:
    state, origin, promise_key, effect_id = _reply_outbox_prepared_state(
        promise, effect
    )
    assert reply_outbox_oracle.can_generate_reply(state, promise_key, origin), (
        "reply_outbox_protocol: claim_before_generate violated"
    )
    claimed = reply_outbox_oracle.claim_outbox_effect(effect_id, state)
    assert claimed is not None, (
        f"reply_outbox_protocol: claim_outbox_effect rejected {label}"
    )
    if effect.state == "claimed":
        return claimed, origin, effect_id
    assert effect.state == "delivered" and effect.external_id is not None, (
        "reply_outbox_protocol: visible reply effect must be claimed or delivered"
    )
    posted = reply_outbox_oracle.record_reply_posted(
        effect_id,
        int(effect.external_id),
        claimed,
    )
    assert posted is not None, (
        f"reply_outbox_protocol: record_reply_posted rejected {label}"
    )
    return posted, origin, effect_id


def _assert_reply_outbox_claimed_visible_effect(
    promise: ReplyPromiseRecord, effect: ReplyOutboxEffectRecord
) -> None:
    """Crash if Python would post without a durable D14 outbox claim."""
    claimed, _origin, effect_id = _reply_outbox_recorded_visible_state(
        promise,
        effect,
        label="visible reply claim",
    )
    assert effect.state == "claimed", (
        "reply_outbox_protocol: visible reply effect must be durably claimed"
    )
    assert isinstance(
        reply_outbox_oracle.outbox_decision(claimed, effect_id),
        reply_outbox_oracle.WaitForInFlightEffect,
    ), "reply_outbox_protocol: claim_before_post violated"


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


def _normalize_comment_ids(comment_ids: Iterable[object]) -> tuple[int, ...]:
    """Return positive integer comment ids in stable first-seen order."""
    normalized: list[int] = []
    for comment_id in comment_ids:
        if not isinstance(comment_id, int | str):
            continue
        try:
            value = int(comment_id)
        except TypeError, ValueError:
            continue
        if value > 0 and value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _review_lineage(
    repo: str,
    pr_number: int,
    comment: dict[str, Any],
    thread_comments: Iterable[dict[str, Any]] = (),
) -> tuple[str, tuple[int, ...]]:
    """Return the durable lineage key and covered ids for a review thread."""
    comment_id = int(comment["id"])
    root_id = int(comment.get("in_reply_to_id") or comment_id)
    lineage_ids = _normalize_comment_ids(
        (
            root_id,
            *(thread_comment.get("id") for thread_comment in thread_comments),
            comment_id,
        )
    )
    return _review_lane_key(repo, pr_number, root_id), lineage_ids


def thread_lineage_comment_ids(thread: dict[str, Any] | None) -> tuple[int, ...]:
    """Return comment ids covered by a task/reply thread lineage."""
    if not thread:
        return ()
    lineage = thread.get("lineage_comment_ids")
    if isinstance(lineage, list):
        return _normalize_comment_ids(lineage)
    comment_id = thread.get("comment_id")
    if comment_id is None:
        return ()
    return _normalize_comment_ids((comment_id,))


def _comment_created_at(comment: dict[str, Any]) -> str:
    """Return a stable GitHub timestamp for FIFO ordering."""
    created_at = comment.get("created_at") or comment.get("updated_at")
    if created_at:
        return str(created_at)
    return datetime.now(tz=UTC).isoformat()


def _enqueue_pr_comment_webhook(
    *,
    repo_cfg: RepoConfig,
    repo: str,
    pr_number: int,
    comment_type: str,
    comment: dict[str, Any],
    author: str,
    is_bot: bool,
    body: str,
    delivery_id: str | None,
    payload: dict[str, Any],
) -> None:
    """Persist one normalized PR comment webhook before acting on it."""
    comment_id = comment.get("id")
    if delivery_id is None or comment_id is None:
        return
    FidoStore(repo_cfg.work_dir).enqueue_pr_comment(
        delivery_id=delivery_id,
        repo=repo,
        pr_number=pr_number,
        comment_type=comment_type,
        comment_id=int(comment_id),
        author=author,
        is_bot=is_bot,
        body=body,
        github_created_at=_comment_created_at(comment),
        payload_json=json.dumps(
            {
                "event": (
                    "pull_request_review_comment"
                    if comment_type == "pulls"
                    else "issue_comment"
                ),
                "delivery_id": delivery_id,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _queued_pr_comment_action(
    *,
    prompt: str,
    repo: str,
    pr_number: int,
    comment_type: str,
    comment_id: int,
    html_url: str,
    author: str,
    is_bot: bool,
    context: dict[str, Any],
) -> Action:
    """Wake the worker for a durably queued PR comment without using provider."""
    return Action(
        prompt=prompt,
        preempts_worker=True,
        is_bot=is_bot,
        context=context,
        thread={
            "repo": repo,
            "pr": pr_number,
            "comment_id": comment_id,
            "url": html_url,
            "author": author,
            "comment_type": comment_type,
        },
    )


def reply_outcome_creates_tasks(
    category: str,
    *,
    thread: dict[str, Any] | None = None,
    is_bot: bool = False,
) -> bool:
    """Return whether a reply outcome should create task objects.

    Synthesis produces ``"ACT"`` (change_request present) or ``"ANSWER"``
    (no scope change).  Only ``"ACT"`` creates tasks.
    """
    return category == "ACT"


def queue_reply_tasks(
    category: str,
    titles: list[str],
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    *,
    thread: dict[str, Any] | None,
    is_bot: bool = False,
    registry: Any = None,  # noqa: ANN401  # WorkerRegistry-or-ActivityReporter; either works
    create_task_fn: Callable[..., object] | None = None,
    dispatcher: "Dispatcher | None" = None,
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
            dispatcher=dispatcher,
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
    thread: dict[str, Any] | None,
    registry: ActivityReporter,
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
    registry: ActivityReporter,
    *,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
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

    for group in issue_groups.values():
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
                registry=registry,
            )
        except Exception:
            _mark_promises_failed(store, (promise_id for promise_id, _ in group))
            raise
        _apply_reply_result(
            category,
            titles,
            config,
            repo_cfg,
            gh,
            action.thread,
            registry,
        )
        _ack_promises(store, (promise_id for promise_id, _ in group))
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
                registry=registry,
            )
        except Exception:
            _mark_promises_failed(
                store, (group_promise_id for group_promise_id, _ in group)
            )
            raise
        _apply_reply_result(
            category,
            titles,
            config,
            repo_cfg,
            gh,
            thread=action.reply_to,
            registry=registry,
        )
        _ack_promises(store, (group_promise_id for group_promise_id, _ in group))
        processed_any = True

    return processed_any


def _is_allowed(user: str, repo_cfg: RepoConfig, config: Config) -> bool:
    """Check if user is a repo collaborator or an allowed bot.

    ``repo_cfg.membership.collaborators`` is populated at server startup
    (``server.populate_memberships``) and excludes the bot itself.
    """
    return user in repo_cfg.membership.collaborators or user in config.allowed_bots


class Dispatcher:
    """Typed collaborator that owns the dispatch, backfill, and sync logic.

    One ``Dispatcher`` is constructed per repo at startup — accepting
    ``config``, ``repo_cfg``, and optionally ``gh`` — so every method call
    uses the repo context already baked into the instance.

    This is the replacement for the ``_fn_dispatch`` callable-slot pattern on
    :class:`~fido.server.WebhookHandler`.  Callers construct one instance per
    repo and hold them in a ``dict[str, Dispatcher]`` collaborator rather than
    reaching into a class-level function slot.

    ``gh`` is optional because :meth:`dispatch` does not use it.  Pass ``gh``
    when constructing a ``Dispatcher`` that will call
    :meth:`backfill_missed_pr_comments` or :meth:`launch_sync`.
    """

    def __init__(
        self, config: Config, repo_cfg: RepoConfig, gh: GitHub | None = None
    ) -> None:
        self._config = config
        self._repo_cfg = repo_cfg
        self._gh = gh

    def dispatch(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        delivery_id: str | None = None,
        oracle: WebhookIngressOracle | None = None,
    ) -> Action | None:
        """Map a GitHub webhook event to an action. Returns None if ignored.

        When *delivery_id* and *oracle* are provided the
        :class:`WebhookIngressOracle` is consulted before any routing logic
        runs.  Duplicate deliveries (same delivery ID arriving a second time)
        and ``pull_request_review / submitted`` events with ``review.state ==
        "commented"`` are suppressed by returning ``None`` before any side
        effects execute.  Decisive review states (``approved``,
        ``changes_requested``, ``dismissed``) are dispatched normally so the
        worker wakes up without waiting for the next poll cycle.
        """
        action = payload.get("action")
        repo_cfg = self._repo_cfg
        repo = repo_cfg.name  # validated at routing time

        # Oracle check — deduplicate at the ingress boundary.
        if delivery_id is not None and oracle is not None:
            review_state = payload.get("review", {}).get("state", "")
            collapse_review = (
                event == "pull_request_review"
                and action == "submitted"
                and review_state == "commented"
            )
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
            assignee = payload["assignee"]["login"]
            issue = payload["issue"]
            number = issue["number"]
            title = issue["title"]
            if not number:
                return None
            log.info("issue #%s assigned to %s: %s", number, assignee, title)
            wev = wct_oracle.EvtIssueAssigned(1, number, assignee)
            cmd = wct_oracle.translate(wev)
            assert isinstance(cmd, wct_oracle.CmdIssueAssigned), "translate_total"
            return Action(prompt=f"New issue #{number} assigned to {assignee}: {title}")

        if event == "pull_request_review" and action == "submitted":
            review = payload["review"]
            pr = payload["pull_request"]
            number = pr.get("number")
            state = review["state"]
            user = review["user"]["login"]
            review_id = review.get("id")
            if not number:
                return None
            if not _is_allowed(user, repo_cfg, self._config):
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

        if event == "pull_request_review_comment" and action in {"created", "edited"}:
            comment = payload["comment"]
            pr = payload["pull_request"]
            number = pr.get("number")
            user = comment["user"]["login"]
            comment_id = comment["id"]
            if user.lower() in ("fidocancode", "fido-can-code"):
                log.debug("ignoring own comment on PR #%s", number)
                return None
            if not number:
                return None
            if not _is_allowed(user, repo_cfg, self._config):
                log.debug(
                    "ignoring comment on PR #%s by %s (not allowed)", number, user
                )
                return None
            comment_body = comment.get("body", "") or ""
            log.info("comment on PR #%s by %s: %s", number, user, comment_body[:80])
            is_bot = user.endswith("[bot]")
            comment_id = int(comment_id)
            wev = wct_oracle.EvtReviewComment(1, number, comment_id, user, is_bot)
            cmd = wct_oracle.translate(wev)
            assert isinstance(cmd, wct_oracle.CmdComment), "translate_total"
            assert isinstance(cmd.cmd_kind, wct_oracle.ReviewLine), "translate_total"
            _enqueue_pr_comment_webhook(
                repo_cfg=repo_cfg,
                repo=repo,
                pr_number=number,
                comment_type="pulls",
                comment=comment,
                author=user,
                is_bot=is_bot,
                body=comment_body,
                delivery_id=delivery_id,
                payload=payload,
            )
            prefix = (
                "Queued edited review comment"
                if action == "edited"
                else "Queued review comment"
            )
            return _queued_pr_comment_action(
                prompt=(
                    f"{prefix} on PR #{number} by {user}"
                    f" ({'bot' if is_bot else 'human/owner'})"
                ),
                repo=repo,
                pr_number=number,
                comment_type="pulls",
                comment_id=comment_id,
                html_url=comment.get("html_url", ""),
                author=user,
                is_bot=is_bot,
                context={
                    "comment_body": comment_body,
                    "delivery_id": delivery_id,
                    "pr_title": pr.get("title", ""),
                    "pr_body": pr.get("body", "") or "",
                    "file": comment.get("path", ""),
                    "line": comment.get("line"),
                    "diff_hunk": comment.get("diff_hunk", ""),
                },
            )

        if event == "issue_comment" and action in {"created", "edited"}:
            comment = payload["comment"]
            issue = payload["issue"]
            user = comment["user"]["login"]
            pr = issue.get("pull_request")
            if not pr:
                log.debug("issue_comment on non-PR issue — ignoring")
                return None
            if user.lower() in ("fidocancode", "fido-can-code"):
                log.debug("ignoring own comment on PR")
                return None
            if not _is_allowed(user, repo_cfg, self._config):
                log.debug("ignoring comment by %s (not allowed)", user)
                return None
            number = issue["number"]
            comment_body = comment.get("body", "") or ""
            comment_id = int(comment["id"])
            is_bot = user.endswith("[bot]")
            log.info("PR comment on #%s by %s: %s", number, user, comment_body[:80])
            wev = wct_oracle.EvtIssueComment(1, number, comment_id, user, is_bot)
            cmd = wct_oracle.translate(wev)
            assert isinstance(cmd, wct_oracle.CmdComment), "translate_total"
            assert isinstance(cmd.cmd_kind, wct_oracle.TopLevelPR), "translate_total"
            _enqueue_pr_comment_webhook(
                repo_cfg=repo_cfg,
                repo=repo,
                pr_number=number,
                comment_type="issues",
                comment=comment,
                author=user,
                is_bot=is_bot,
                body=comment_body,
                delivery_id=delivery_id,
                payload=payload,
            )
            prefix = (
                "Queued edited PR top-level comment"
                if action == "edited"
                else "Queued PR top-level comment"
            )
            return _queued_pr_comment_action(
                prompt=f"{prefix} on #{number} by {user}",
                repo=repo,
                pr_number=number,
                comment_type="issues",
                comment_id=comment_id,
                html_url=comment.get("html_url", ""),
                author=user,
                is_bot=is_bot,
                context={
                    "comment_body": comment_body,
                    "delivery_id": delivery_id,
                    "pr_title": issue.get("title", ""),
                    "pr_body": issue.get("body", "") or "",
                },
            )

        if event == "check_run" and action == "completed":
            check = payload["check_run"]
            conclusion = check["conclusion"]
            if conclusion not in ("failure", "timed_out"):
                log.debug("check_run completed with %s — ignoring", conclusion)
                return None
            name = check["name"]
            prs = check["pull_requests"]
            pr_nums = [pr["number"] for pr in prs]
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
            pr = payload["pull_request"]
            number = pr["number"]
            removed = FidoStore(repo_cfg.work_dir).clear_pr_comment_queue(
                repo=repo,
                pr_number=int(number),
            )
            if removed:
                log.info(
                    "cleared %d queued comment(s) for closed PR #%s", removed, number
                )
            if not pr["merged"]:
                log.debug("PR #%s closed without merge — ignoring", number)
                return None
            log.info("PR #%s merged", number)
            wev = wct_oracle.EvtPRMerged(1, number)
            cmd = wct_oracle.translate(wev)
            assert isinstance(cmd, wct_oracle.CmdPRMerged), "translate_total"
            return Action(prompt=f"PR #{number} merged — cleanup")

        log.debug("ignored event: %s (action=%s)", event, action)
        return None

    def backfill_missed_pr_comments(
        self,
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
        already replied to them.  Comments handled with a task (ACT/DO) are
        additionally deduped by :func:`create_task` via ``comment_id`` in
        ``tasks.json``.  This method is intended to run **once per
        WorkerThread lifetime** (at startup) — not every iteration.
        """
        assert self._gh is not None, "gh required for backfill_missed_pr_comments"
        repo_cfg = self._repo_cfg
        log.info("backfill: scanning PR #%s for missed top-level comments", pr_number)
        comments = self._gh.get_issue_comments(repo_cfg.name, pr_number)
        for c in comments:
            user = (c.get("user") or {}).get("login", "")
            if not user:
                continue
            if user.lower() == gh_user.lower():
                continue
            if user.lower() in ("fidocancode", "fido-can-code"):
                continue
            if not _is_allowed(user, repo_cfg, self._config):
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
            create_task(
                prompt, self._config, repo_cfg, self._gh, thread=thread, dispatcher=self
            )
        log.info("backfill: PR #%s — inspected %d comments", pr_number, len(comments))
        return len(comments)

    def launch_sync(self) -> None:
        """Sync tasks.json → PR body in a background thread."""
        assert self._gh is not None, "gh required for launch_sync"
        from fido.tasks import sync_tasks_background

        sync_tasks_background(self._repo_cfg.work_dir, self._gh)
        log.info("sync-tasks launched")


def _load_persona(config: Config) -> str:
    """Read persona.md from sub_dir; return empty string if missing."""
    try:
        return (config.sub_dir / "persona.md").read_text()
    except FileNotFoundError:
        return ""


def reply_to_comment(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    registry: ActivityReporter,
    *,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> tuple[str, list[str]]:
    """Handle a review comment via the synthesis LLM call, post reply.

    Returns (category, task_titles) where category is derived from the
    synthesis response: ``"ACT"`` when a change_request is present,
    ``"ANSWER"`` otherwise.  task_titles is ``[change_request]`` for ACT
    (the caller passes this to :func:`queue_reply_tasks` to create a task)
    or ``[]`` for ANSWER.

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
        # Convert before the truthiness check — a generator or auto-mocked
        # iterable is truthy regardless of whether iterating it yields any
        # items, so ``if fetched:`` on the raw return is unsafe.
        thread_comments = list(
            gh.fetch_comment_thread(info["repo"], info["pr"], info["comment_id"])
        )
        if thread_comments:
            context["comment_thread"] = thread_comments
            root_comment = next(
                (
                    thread_comment
                    for thread_comment in thread_comments
                    if thread_comment.get("in_reply_to_id") is None
                ),
                thread_comments[0],
            )
            lineage_ids = _normalize_comment_ids(
                thread_comment.get("id") for thread_comment in thread_comments
            )
            info["lineage_key"] = _review_lane_key(
                info["repo"], int(info["pr"]), int(root_comment["id"])
            )
            info["lineage_comment_ids"] = list(lineage_ids)
            log.info(
                "fetched %d comment(s) in thread for context", len(thread_comments)
            )

    # Capture Fido reply IDs from the initial snapshot so we can detect new
    # replies posted by concurrent handlers during the synthesis window
    # (compared against the re-fetched thread below).
    initial_fido_ids: set[int] = {
        c["id"] for c in thread_comments if c.get("author", "").lower() in _FIDO_LOGINS
    }

    # Add eyes reaction immediately at pickup to signal work-in-progress.
    # Best-effort: never fail the reply if the reaction post fails.
    _eyes_posted = False
    if info.get("repo") and info.get("comment_id"):
        try:
            gh.add_reaction(
                info["repo"],
                "pulls",
                info["comment_id"],
                "eyes",
            )
            log.info("added eyes reaction to comment %s", info["comment_id"])
            _eyes_posted = True
        except Exception:
            log.exception(
                "failed to add eyes reaction to comment %s — continuing",
                info["comment_id"],
            )

    issue_ctx, pr_ctx = _load_active_context_for_rescope(
        repo_cfg.work_dir, repo_cfg.name, gh
    )

    # Build the executor + target up-front so the failure path can clean
    # up the eyes reaction without duplicating the cleanup logic.  The
    # success path uses the same instance below in execute_effects_only.
    rescope_trigger = _BackgroundRescopeTrigger(
        repo_cfg.work_dir,
        config,
        gh,
        repo_cfg,
        registry,
        agent=agent,
        prompts=prompts,
    )
    executor = SynthesisExecutor(
        gh,
        rescope=rescope_trigger,
        insight_filer=_GitHubInsightFiler(gh),
        fido_logins=_FIDO_LOGINS,
    )
    target: CommentTarget | None = None
    if isinstance(info.get("comment_id"), int):
        target = CommentTarget(
            repo=str(info.get("repo", "")),
            pr=int(info.get("pr", 0)),
            comment_id=int(info["comment_id"]),
            comment_type="pulls",
        )

    log.info(
        "synthesis: calling for PR #%s comment %s",
        info["pr"],
        info["comment_id"],
    )
    try:
        synthesis_response = call_synthesis(
            comment,
            is_bot=action.is_bot,
            context=context or None,
            issue=issue_ctx,
            pr=pr_ctx,
            agent=agent,
            prompts=prompts,
        )
    except SynthesisExhaustedError:
        log.warning(
            "synthesis exhausted retries for comment %s — falling back to "
            "LLM-generated failure explanation",
            info.get("comment_id"),
        )
        try:
            synthesis_response = call_failure_explanation(
                comment, agent=agent, prompts=prompts
            )
        except SynthesisExhaustedError:
            # Even the fallback explanation exhausted retries.  Best-effort
            # clear the eyes reaction so it does not sit forever as a false
            # "Fido is looking" signal, then re-raise.
            if _eyes_posted and target is not None:
                executor.remove_eyes_reaction(target)
            raise
    log.info(
        "synthesis: returned (emoji=%r change_request=%r preview=%r)",
        synthesis_response.emoji,
        synthesis_response.change_request,
        synthesis_response.reply_text[:80],
    )
    body = synthesis_response.reply_text

    # Derive (category, titles) from synthesis response for caller's
    # queue_reply_tasks call.
    if synthesis_response.change_request is not None:
        category: str = "ACT"
        titles: list[str] = [synthesis_response.change_request]
    else:
        category = "ANSWER"
        titles = []

    # Re-fetch the thread right before posting so the concurrent-reply check
    # uses current GitHub state rather than the snapshot taken before synthesis.
    if info.get("repo") and info.get("pr") and info.get("comment_id"):
        refreshed = gh.fetch_comment_thread(
            info["repo"], info["pr"], info["comment_id"]
        )
        if refreshed:
            thread_comments = refreshed
            log.info(
                "re-fetched %d comment(s) in thread before posting",
                len(thread_comments),
            )

    # Skip posting if a concurrent handler already replied to *this specific*
    # comment during synthesis.  A Fido reply with ``in_reply_to_id`` matching
    # this comment id, that wasn't in the initial snapshot, means another
    # handler handled it — adding a second reply would be a duplicate.
    #
    # Closes #1004; mirrors the original double-post fix in #518.
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
    root_comment_id = (
        thread_comments[0]["id"] if thread_comments else info["comment_id"]
    )
    existing_artifact_id = _existing_reply_artifact(repo_cfg, promise_ids)
    if existing_artifact_id is None:
        _claim_reply_outbox_effects(
            repo_cfg,
            delivery_id=_reply_outbox_delivery_key(
                context, comment_type="pulls", comment_id=info["comment_id"]
            ),
            promise_ids=promise_ids,
        )
        log.info("posting reply to PR #%s: %s", info["pr"], body[:80])
        posted = gh.reply_to_review_comment(
            info["repo"], info["pr"], body, info["comment_id"]
        )
        log.info("reply posted")
        if root_comment_id is not None:
            artifact_comment_id = _posted_comment_id(posted)
            assert artifact_comment_id is not None or not promise_ids, (
                "reply_outbox_protocol: GitHub reply post returned no comment id"
            )
            _record_reply_artifact(
                repo_cfg,
                artifact_comment_id=artifact_comment_id,
                comment_type="pulls",
                lane_key=_review_lane_key(
                    info["repo"], int(info["pr"]), int(root_comment_id)
                ),
                promise_ids=promise_ids,
            )
    else:
        log.info(
            "reply artifact %s already recorded — skipping post", existing_artifact_id
        )
    if direct_promise is not None:
        FidoStore(repo_cfg.work_dir).ack_promise(direct_promise.promise_id)

    # Execute post-reply effects: remove eyes, add final emoji, trigger
    # rescope.  Only runs if we have a real comment_id to react on (the
    # ``target`` was constructed up-front for the same reason).  Reuses the
    # ``executor`` instance built before the synthesis call so the failure
    # path could share its eyes-removal helper.
    if target is not None:
        executor.execute_effects_only(synthesis_response, target)

    return (category, titles)


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


def reply_to_issue_comment(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    registry: ActivityReporter,
    *,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> tuple[str, list[str]]:
    """Handle a top-level PR comment via the synthesis LLM call, post reply.

    Returns (category, task_titles) where category is derived from the
    synthesis response: ``"ACT"`` when a change_request is present,
    ``"ANSWER"`` otherwise.

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

    # Merge conversation context into synthesis context
    if conversation_context:
        context["conversation"] = conversation_context

    # Add eyes reaction immediately at pickup to signal work-in-progress.
    # Best-effort: never fail the reply if the reaction post fails.
    _eyes_posted_issue = False
    _cid = context.get("comment_id")
    if _cid and repo_full:
        try:
            gh.add_reaction(repo_full, "issues", _cid, "eyes")
            log.info("added eyes reaction to issue comment %s on PR #%s", _cid, number)
            _eyes_posted_issue = True
        except Exception:
            log.exception(
                "failed to add eyes reaction to issue comment %s — continuing", _cid
            )

    issue_ctx, pr_ctx = _load_active_context_for_rescope(
        repo_cfg.work_dir, repo_cfg.name, gh
    )

    # Build the executor + target up-front so the failure path can clean
    # up the eyes reaction without duplicating the cleanup logic.  The
    # success path uses the same instance below in execute_effects_only.
    rescope_trigger = _BackgroundRescopeTrigger(
        repo_cfg.work_dir,
        config,
        gh,
        repo_cfg,
        registry,
        agent=agent,
        prompts=prompts,
    )
    executor = SynthesisExecutor(
        gh,
        rescope=rescope_trigger,
        insight_filer=_GitHubInsightFiler(gh),
        fido_logins=_FIDO_LOGINS,
    )
    issue_target: CommentTarget | None = None
    if _cid and repo_full:
        issue_target = CommentTarget(
            repo=repo_full,
            pr=int(number) if number else 0,
            comment_id=int(_cid),
            comment_type="issues",
        )

    log.info("synthesis: calling for issue comment on PR #%s", number)
    try:
        synthesis_response = call_synthesis(
            comment,
            is_bot=action.is_bot,
            context=context or None,
            issue=issue_ctx,
            pr=pr_ctx,
            agent=agent,
            prompts=prompts,
        )
    except SynthesisExhaustedError:
        log.warning(
            "synthesis exhausted retries for issue comment %s on PR #%s — "
            "falling back to LLM-generated failure explanation",
            _cid,
            number,
        )
        try:
            synthesis_response = call_failure_explanation(
                comment, agent=agent, prompts=prompts
            )
        except SynthesisExhaustedError:
            # Even the fallback explanation exhausted retries.  Best-effort
            # clear the eyes reaction so it does not sit forever as a false
            # "Fido is looking" signal, then re-raise.
            if _eyes_posted_issue and issue_target is not None:
                executor.remove_eyes_reaction(issue_target)
            raise
    log.info(
        "synthesis: returned for PR #%s (emoji=%r change_request=%r preview=%r)",
        number,
        synthesis_response.emoji,
        synthesis_response.change_request,
        synthesis_response.reply_text[:80],
    )
    body = synthesis_response.reply_text

    # Derive (category, titles) from synthesis response.
    if synthesis_response.change_request is not None:
        category: str = "ACT"
        titles: list[str] = [synthesis_response.change_request]
    else:
        category = "ANSWER"
        titles = []

    promise_ids = _reply_promise_ids(context)
    body = append_reply_promise_markers(body, promise_ids)
    existing_artifact_id = _existing_reply_artifact(repo_cfg, promise_ids)
    if existing_artifact_id is None:
        _claim_reply_outbox_effects(
            repo_cfg,
            delivery_id=_reply_outbox_delivery_key(
                context, comment_type="issues", comment_id=comment_id
            ),
            promise_ids=promise_ids,
        )
        log.info("posting issue comment reply on PR #%s: %s", number, body[:80])
        posted = gh.comment_issue(repo_full, number, body)
        log.info("reply posted on PR #%s", number)
        artifact_comment_id = _posted_comment_id(posted)
        assert artifact_comment_id is not None or not promise_ids, (
            "reply_outbox_protocol: GitHub issue comment returned no comment id"
        )
        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=artifact_comment_id,
            comment_type="issues",
            lane_key=_issue_lane_key(repo_full, int(number) if number else 0),
            promise_ids=promise_ids,
        )
    else:
        log.info(
            "reply artifact %s already recorded — skipping post", existing_artifact_id
        )
    if direct_promise is not None:
        FidoStore(repo_cfg.work_dir).ack_promise(direct_promise.promise_id)

    # Execute post-reply effects: remove eyes, add final emoji, trigger
    # rescope.  Reuses the ``executor`` and ``issue_target`` constructed
    # before the synthesis call so the failure path could share its
    # eyes-removal helper.
    if issue_target is not None:
        executor.execute_effects_only(synthesis_response, issue_target)

    log.info(
        "reply_to_issue_comment: complete for PR #%s (category=%s)", number, category
    )
    return (category, titles)


_TYPE_PRIORITY = {TaskType.CI: 0, TaskType.THREAD: 1, TaskType.SPEC: 2}


def _maybe_abort_for_new_task(
    repo_cfg: RepoConfig,
    new_task: dict[str, Any],
    registry: ActivityReporter,
    *,
    _state: State | None = None,
    _tasks: Tasks | None = None,
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
        registry.abort_task(repo_cfg.name, task_id=current_task_id)


def _get_commit_summary(work_dir: Path) -> str:
    """Return a short ``git log --oneline`` summary of recent commits.

    Used to give Opus context about what has already been implemented when it
    reorders the pending task list.  Raises on subprocess error or nonzero git
    exit so callers see real failures rather than silently receiving an empty
    string.
    """
    result = subprocess.run(
        ["git", "log", "--oneline", "-20"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return result.stdout.strip()


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
    return [(t["id"], t["status"], t["title"]) for t in task_list]


def _rewrite_pr_description(
    work_dir: Path,
    gh: GitHub,
    *,
    agent: ProviderAgent | None = None,
    _state: State | None = None,
    _tasks: Tasks | None = None,
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


def _load_active_context_for_rescope(
    work_dir: Path,
    repo_name: str,
    gh: GitHub,
) -> tuple[ActiveIssue | None, ActivePR | None]:
    """Read issue/PR snapshots from state and GitHub for the rescope prompt.

    Returns ``(ActiveIssue, ActivePR)`` when both are resolvable, or
    ``(ActiveIssue, None)`` / ``(None, None)`` on partial / missing data.
    Silently returns ``(None, None)`` when state does not yet record an issue
    (e.g. rescope fired before the worker set state.json).
    """
    fido_dir = work_dir / ".git" / "fido"
    state_path = fido_dir / "state.json"
    if not state_path.exists():
        return None, None
    state_data = State(fido_dir).load()
    issue_number = state_data.get("issue")
    pr_number = state_data.get("pr_number")
    if not isinstance(issue_number, int):
        return None, None
    issue_data = gh.view_issue(repo_name, issue_number)
    issue_ctx = ActiveIssue(
        number=issue_number,
        title=issue_data["title"],
        body=issue_data.get("body") or "",
    )
    if not isinstance(pr_number, int):
        return issue_ctx, None
    pr_data = gh.get_pr(repo_name, pr_number)
    pr_ctx = ActivePR(
        number=pr_number,
        title=pr_data["title"],
        url=f"https://github.com/{repo_name}/pull/{pr_number}",
        body=pr_data["body"] or "",
    )
    return issue_ctx, pr_ctx


def _make_reorder_kwargs(
    work_dir: Path,
    config: Config,
    repo_cfg: RepoConfig,
    registry: ActivityReporter,
    gh: GitHub,
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

    def on_inprogress_affected(task_id: str) -> None:
        log.info(
            "reorder_tasks_background: in-progress task affected — aborting %s",
            repo_cfg.name,
        )
        registry.abort_task(repo_cfg.name, task_id=task_id)

    kwargs: dict[str, Any] = {
        "_on_changes": on_changes,
        "_on_done": on_done,
        "_on_inprogress_affected": on_inprogress_affected,
        "agent": agent,
        "prompts": prompts,
    }
    issue_ctx, pr_ctx = _load_active_context_for_rescope(work_dir, repo_cfg.name, gh)
    if issue_ctx is not None:
        kwargs["issue"] = issue_ctx
    if pr_ctx is not None:
        kwargs["pr"] = pr_ctx
    return kwargs


def _reorder_tasks_background(
    work_dir: Path,
    commit_summary: str,
    config: Config,
    gh: GitHub,
    repo_cfg: RepoConfig,
    registry: ActivityReporter,
    *,
    intents: list[RescopeIntent] | None = None,
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

    When *intents* is provided (comment-triggered rescope), they accumulate in
    the pending entry rather than being overwritten — so every originating
    comment is tracked and the rescoper can reference all of them.  The commit
    summary and kwargs from the latest call win (context should be fresh);
    intents from all coalesced calls are preserved.

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
        agent = _configured_agent(config, repo_cfg)
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
            # Coalesce: latest commit_summary and kwargs win; intents accumulate
            # so every originating comment is tracked even when multiple
            # comment-triggered rescopes arrive during the same Opus call.
            existing = entry["pending"]
            if existing is None:
                entry["pending"] = (commit_summary, kwargs, list(intents or []))
            else:
                accumulated = list(existing[2]) + list(intents or [])
                entry["pending"] = (commit_summary, kwargs, accumulated)
            return
        entry["running"] = True
        entry["pending"] = None

    def run_loop() -> None:
        cs = commit_summary
        kw = kwargs
        current_intents: list[RescopeIntent] = list(intents or [])
        release_untriaged = 0
        iteration = 0
        # Register as "webhook" so the session talker reflects the true nature of
        # this thread: it is triggered by webhooks and should not be treated as the
        # worker for preemption purposes.  Without this, current_thread_kind()
        # defaults to "worker", causing real webhooks to fire _fire_worker_cancel
        # against the reorder thread and misidentify it as the running worker (#955).
        #
        # IMPORTANT: every line from here through the end of the finally must be
        # inside the try/finally — if a prelude line (set_thread_kind,
        # set_rescoping, etc.) raises, the finally still has to release the
        # inbox holds; otherwise the worker parks forever (#1280).
        try:
            set_thread_kind("webhook")
            set_thread_repo(repo_cfg.name)
            registry.set_rescoping(repo_cfg.name, True)
            log.info("rescope BG: starting (work_dir=%s)", work_dir)
            while True:
                iteration += 1
                log.info("rescope BG: iteration %d starting", iteration)
                reorder(work_dir, cs, intents=current_intents or None, **kw)
                log.info("rescope BG: iteration %d complete", iteration)
                with _reorder_coalesce_lock:
                    pending = state[key].get("pending")
                    if pending is None:
                        break
                    state[key]["pending"] = None
                    cs, kw, current_intents = pending
        finally:
            log.info("rescope BG: entering finally (iterations=%d)", iteration)
            with _reorder_coalesce_lock:
                entry = state.get(key)
                if entry is not None:
                    release_untriaged += int(entry.get("untriaged_holds", 0))
                    entry["untriaged_holds"] = 0
                    entry["running"] = False
            # Release the inbox holds FIRST. A failure in any subsequent
            # cleanup step (set_rescoping, set_thread_repo, set_thread_kind)
            # must not skip the exit_untriaged calls — losing them leaves the
            # worker permanently blocked on a non-empty inbox counter (#1280).
            # registry and repo_cfg are required at construction time (#1336)
            # so the release path is unconditional — no Optional guards.
            for _ in range(release_untriaged):
                registry.exit_untriaged(repo_cfg.name)
            registry.set_rescoping(repo_cfg.name, False)
            set_thread_repo(None)
            set_thread_kind(None)
            log.info(
                "rescope BG: finally complete (released %d untriaged hold(s))",
                release_untriaged,
            )

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
        # Same ordering as run_loop's finally — release before set_rescoping.
        # registry and repo_cfg are required (#1336) so this path is
        # unconditional.
        for _ in range(release_untriaged):
            registry.exit_untriaged(repo_cfg.name)
        registry.set_rescoping(repo_cfg.name, False)
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
    registry: ActivityReporter | None = None,
    *,
    dispatcher: "Dispatcher | None" = None,
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
    if dispatcher is not None:
        dispatcher.launch_sync()
    if thread:
        commit_summary = _get_commit_summary_fn(repo_cfg.work_dir)
        if _reorder_background_fn is _reorder_tasks_background:
            # Production path — _reorder_tasks_background requires a real
            # registry to bookkeep the inbox.  Without one, skip the rescope
            # entirely and log loudly (#1336): silently dropping the rescope
            # is exactly the fail-soft pattern that produced the original
            # bug.
            if registry is None:
                log.warning(
                    "create_task: thread task created without registry — "
                    "skipping background rescope (#1336)",
                )
            else:
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


def launch_worker(repo_cfg: RepoConfig, registry: WorkerRegistry) -> None:
    """Wake the per-repo WorkerThread via the registry."""
    log.info("waking worker thread for %s", repo_cfg.name)
    registry.wake(repo_cfg.name)
