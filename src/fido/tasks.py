"""Shared task file operations with flock-based locking."""

import fcntl
import json
import logging
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fido.appstate import FidoState, TaskListSnapshot
    from fido.atomic import AtomicUpdater

from fido.claude import ClaudeClient
from fido.github import GitHub
from fido.prompts import Prompts
from fido.provider import READ_ONLY_ALLOWED_TOOLS, ProviderAgent
from fido.rocq import pr_body_task_store as task_store_oracle
from fido.rocq import task_queue_rescope as rescope_oracle
from fido.rocq import thread_auto_resolve as thread_resolve_oracle
from fido.state import (
    JsonFileStore,
    State,
    _resolve_git_dir,  # pyright: ignore[reportPrivateUsage]
)
from fido.types import (
    ActiveIssue,
    ActivePR,
    ClosedPR,
    IntentVerdict,
    RescopeIntent,
    TaskStatus,
    TaskType,
)

log = logging.getLogger(__name__)

# Maximum number of nudge retries when Opus proposes duplicate task titles.
_RESCOPE_MAX_NUDGES = 3


def _task_kind_for_oracle(task: dict[str, Any]) -> task_store_oracle.TaskKind:
    title_upper = task.get("title", "").upper()
    if title_upper.startswith("ASK:"):
        return task_store_oracle.TaskAsk()
    if title_upper.startswith("DEFER:"):
        return task_store_oracle.TaskDefer()
    if title_upper.startswith("CI FAILURE:") or task.get("type") == TaskType.CI:
        return task_store_oracle.TaskCI()
    if task.get("type") == TaskType.THREAD:
        return task_store_oracle.TaskThread()
    return task_store_oracle.TaskSpec()


def _task_status_for_oracle(task: dict[str, Any]) -> task_store_oracle.TaskStatus:
    match task.get("status", TaskStatus.PENDING):
        case TaskStatus.COMPLETED | "completed":
            return task_store_oracle.StatusCompleted()
        case TaskStatus.BLOCKED | "blocked":
            return task_store_oracle.StatusBlocked()
        case _:
            return task_store_oracle.StatusPending()


def _task_source_comment_for_oracle(task: dict[str, Any]) -> int | None:
    comment_id = (task.get("thread") or {}).get("comment_id")
    if comment_id is None:
        return None
    return int(comment_id)


def _thread_task_status_for_oracle(
    task: dict[str, Any],
) -> thread_resolve_oracle.TaskStatus:
    match task.get("status", TaskStatus.PENDING):
        case TaskStatus.COMPLETED | "completed":
            return thread_resolve_oracle.StatusCompleted()
        case TaskStatus.BLOCKED | "blocked":
            return thread_resolve_oracle.StatusBlocked()
        case _:
            return thread_resolve_oracle.StatusPending()


def _thread_task_for_auto_resolve_oracle(
    task: dict[str, Any],
) -> thread_resolve_oracle.ThreadTask | None:
    comment_id = (task.get("thread") or {}).get("comment_id")
    if comment_id is None:
        return None
    return thread_resolve_oracle.ThreadTask(
        thread_task_comment=int(comment_id),
        thread_task_status=_thread_task_status_for_oracle(task),
    )


def _thread_tasks_for_auto_resolve_oracle_one(
    task: dict[str, Any],
) -> list[thread_resolve_oracle.ThreadTask]:
    """Project one task into the auto-resolve oracle's per-comment view.

    A non-merged thread task contributes one ThreadTask: its primary
    anchor.  A merged thread task (lineage_comment_ids carries the
    folded-in source anchors after #1717) contributes one ThreadTask
    per lineage comment, all with the same status.  Without this
    expansion, a still-pending merged target's lineage comments would
    be invisible to the auto-resolve oracle and the source review
    threads would resolve before the merged target is actually done
    (codex on #1738).
    """
    primary = _thread_task_for_auto_resolve_oracle(task)
    if primary is None:
        return []
    status = primary.thread_task_status
    seen: set[int] = {primary.thread_task_comment}
    out: list[thread_resolve_oracle.ThreadTask] = [primary]
    for cid in _thread_lineage_comment_ids(task.get("thread")):
        if cid in seen:
            continue
        seen.add(cid)
        out.append(
            thread_resolve_oracle.ThreadTask(
                thread_task_comment=cid, thread_task_status=status
            )
        )
    return out


def thread_tasks_for_auto_resolve_oracle(
    task_list: list[dict[str, Any]],
) -> list[thread_resolve_oracle.ThreadTask]:
    tasks: list[thread_resolve_oracle.ThreadTask] = []
    for task in task_list:
        tasks.extend(_thread_tasks_for_auto_resolve_oracle_one(task))
    return tasks


def thread_comment_author_for_auto_resolve_oracle(
    login: str,
    *,
    fido_logins: frozenset[str],
    owner: str = "",
    collaborators: frozenset[str] = frozenset(),
    allowed_bots: frozenset[str] = frozenset(),
) -> thread_resolve_oracle.ThreadCommentAuthor:
    if login.lower() in fido_logins:
        return thread_resolve_oracle.CommentByFido()
    if login == owner or login in collaborators:
        return thread_resolve_oracle.CommentByActionable()
    if login in allowed_bots or login.endswith("[bot]"):
        return thread_resolve_oracle.CommentByBot()
    return thread_resolve_oracle.CommentIgnored()


def review_thread_for_auto_resolve_oracle(
    node: dict[str, Any],
    gh_user: str,
    *,
    owner: str = "",
    collaborators: frozenset[str] = frozenset(),
    allowed_bots: frozenset[str] = frozenset(),
) -> thread_resolve_oracle.ReviewThread:
    comments: list[thread_resolve_oracle.ThreadComment] = []
    for comment in node.get("comments", {}).get("nodes", []):
        database_id = comment.get("databaseId")
        if database_id is None:
            continue
        author = (comment.get("author") or {}).get("login", "")
        comments.append(
            thread_resolve_oracle.ThreadComment(
                thread_comment_id=int(database_id),
                thread_comment_author=thread_comment_author_for_auto_resolve_oracle(
                    str(author),
                    fido_logins=frozenset({gh_user.lower()}),
                    owner=owner,
                    collaborators=collaborators,
                    allowed_bots=allowed_bots,
                ),
            )
        )
    return thread_resolve_oracle.ReviewThread(
        review_thread_resolved=bool(node.get("isResolved", False)),
        review_thread_comments=comments,
    )


def _review_thread_contains_comment(
    node: dict[str, Any],
    comment_id: int,
) -> bool:
    for comment in node.get("comments", {}).get("nodes", []):
        if comment.get("databaseId") == comment_id:
            return True
    return False


def _normalize_title(title: str) -> str:
    """Collapse all whitespace runs in a task title to single spaces.

    Multiline / tabbed / multi-space input would break PR-body
    round-tripping (one task per markdown checkbox line, parsed by
    ``seed_tasks_from_pr_body``) and produce ugly work-queue rendering.
    """
    return " ".join(title.split())


def _existing_titles_by_id(current: list[dict[str, Any]]) -> dict[str, str]:
    return {t["id"]: t.get("title", "") for t in current if "id" in t}


def _merge_source_oracle_ids(
    item: dict[str, Any],
    ids_by_task_id: dict[str, int],
) -> list[int]:
    """Map a rescope item's ``merge_sources`` task-id list to oracle ids.

    The adapter's contract for #1717: ``item["merge_sources"]`` is a list
    of OTHER task ids whose lineage and primary anchor should fold into
    this item's task.  Empty list = no merge.  Garbage (non-list, target
    referencing itself, unknown id) is ignored — the validator already
    rejects malformed payloads atomically (#1715), so reaching here with
    invalid sources means we've been called under tests that bypass the
    validator; fall back to "no merge" rather than emitting a bogus op.
    """
    raw = item.get("merge_sources")
    if not isinstance(raw, list) or not raw:
        return []
    # _validate_rescope_batch rejects unknown ids, self-targeting,
    # non-string entries, and duplicates atomically before this runs in
    # production.  The isinstance(str) guard is a thin safety net for
    # tests that exercise _apply_reorder directly: dict membership on a
    # non-hashable value (e.g. a nested list) would raise TypeError.
    return [
        ids_by_task_id[src]
        for src in raw
        if isinstance(src, str) and src in ids_by_task_id
    ]


def _is_valid_anchor_id(value: object) -> bool:
    """A GitHub comment id is a positive 64-bit int.

    Reject ``bool`` explicitly even though it inherits from ``int``, so a
    ``"anchor_comment_id": true`` payload from the LLM doesn't become a
    real anchor write.  Zero and negative values are likewise rejected.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _effective_title(item: dict[str, Any], existing_by_id: dict[str, str]) -> str:
    """Compute the title that this rescope item will land as on disk.

    For an item targeting an existing task id, falls back to the existing
    title when the proposed value is missing, blank, or non-string — Opus
    didn't supply a rename, preserve identity.  For a null/unknown id (a
    new task), returns the normalized proposed title or empty string.

    Both the duplicate-title nudge check and the apply-time reducer must
    use this single function so they can't drift: a value that the nudge
    accepted as unique must persist as that same string, and vice versa.
    """
    proposed = item.get("title")
    normalized = _normalize_title(proposed) if isinstance(proposed, str) else ""
    if normalized:
        return normalized
    item_id = item.get("id")
    if item_id is not None and item_id in existing_by_id:
        return existing_by_id[item_id]
    return ""


def _thread_lineage_comment_ids(thread: dict[str, Any] | None) -> list[int]:
    if not thread:
        return []
    comment_ids = thread.get("lineage_comment_ids")
    raw_ids = (
        comment_ids if isinstance(comment_ids, list) else [thread.get("comment_id")]
    )
    lineage: list[int] = []
    for comment_id in raw_ids:
        if not isinstance(comment_id, int | str):
            continue
        try:
            value = int(comment_id)
        except TypeError, ValueError:
            continue
        if value > 0 and value not in lineage:
            lineage.append(value)
    return lineage


def _merge_thread_lineage(
    existing_thread: dict[str, Any], new_thread: dict[str, Any]
) -> bool:
    """Merge related source-comment ids into an existing task thread.

    ``lineage_comment_ids`` is the only lineage metadata that survives —
    it answers "which comments contributed to this task" so a future
    reply-back filter (#1256) can reach every commenter whose intent
    materially shaped the task.  The legacy ``lineage_key`` join field
    is dropped (#1665) — lineage is no longer a dedup key, so the
    canonical chain identifier serves no purpose.
    """
    merged = _thread_lineage_comment_ids(existing_thread)
    changed = False
    for comment_id in _thread_lineage_comment_ids(new_thread):
        if comment_id not in merged:
            merged.append(comment_id)
            changed = True
    if changed:
        existing_thread["lineage_comment_ids"] = merged
    return changed


def _task_store_for_oracle(
    task_list: list[dict[str, Any]],
) -> tuple[task_store_oracle.TaskStore, dict[int, dict[str, Any]]]:
    tasks_by_oracle_id: dict[int, dict[str, Any]] = {}
    rows: dict[int, task_store_oracle.TaskRow] = {}
    order: list[int] = []
    for oracle_id, task in enumerate(task_list, 1):
        order.append(oracle_id)
        tasks_by_oracle_id[oracle_id] = task
        rows[oracle_id] = task_store_oracle.TaskRow(
            title=task.get("title", ""),
            description=task.get("description", ""),
            kind=_task_kind_for_oracle(task),
            status=_task_status_for_oracle(task),
            source_comment=_task_source_comment_for_oracle(task),
            lineage_comments=_thread_lineage_comment_ids(task.get("thread")),
        )
    return task_store_oracle.TaskStore(order, rows), tasks_by_oracle_id


def _rescope_task_kind_for_oracle(task: dict[str, Any]) -> rescope_oracle.TaskKind:
    title_upper = task.get("title", "").upper()
    if title_upper.startswith("ASK:"):
        return rescope_oracle.TaskAsk()
    if title_upper.startswith("DEFER:"):
        return rescope_oracle.TaskDefer()
    if title_upper.startswith("CI FAILURE:") or task.get("type") == TaskType.CI:
        return rescope_oracle.TaskCI()
    if task.get("type") == TaskType.THREAD:
        return rescope_oracle.TaskThread()
    return rescope_oracle.TaskSpec()


def _rescope_task_status_for_oracle(
    task: dict[str, Any],
) -> rescope_oracle.TaskStatus:
    match task.get("status", TaskStatus.PENDING):
        case TaskStatus.COMPLETED | "completed":
            return rescope_oracle.StatusCompleted()
        case TaskStatus.BLOCKED | "blocked":
            return rescope_oracle.StatusBlocked()
        case _:
            return rescope_oracle.StatusPending()


def _rescope_task_source_comment_for_oracle(task: dict[str, Any]) -> int | None:
    comment_id = (task.get("thread") or {}).get("comment_id")
    if comment_id is None:
        return None
    return int(comment_id)


def _rescope_state_for_oracle(
    task_list: list[dict[str, Any]],
) -> tuple[
    dict[str, int],
    dict[int, dict[str, Any]],
    list[int],
    dict[int, rescope_oracle.TaskRow],
]:
    ids_by_task_id: dict[str, int] = {}
    tasks_by_oracle_id: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    rows: dict[int, rescope_oracle.TaskRow] = {}
    for oracle_id, task in enumerate(task_list, 1):
        task_id = task["id"]
        ids_by_task_id[task_id] = oracle_id
        tasks_by_oracle_id[oracle_id] = task
        order.append(oracle_id)
        rows[oracle_id] = rescope_oracle.TaskRow(
            title=task.get("title", ""),
            description=task.get("description", ""),
            kind=_rescope_task_kind_for_oracle(task),
            status=_rescope_task_status_for_oracle(task),
            source_comment=_rescope_task_source_comment_for_oracle(task),
            # #1717 plumbing: lineage_comments mirrors the task's
            # thread.lineage_comment_ids origin metadata.  Today the
            # existing reducer ops preserve it unchanged; the next leaf
            # adds the merge op that writes it.
            lineage_comments=_thread_lineage_comment_ids(task.get("thread")),
        )
    return ids_by_task_id, tasks_by_oracle_id, order, rows


def _rescope_snapshot_order_for_oracle(
    current: list[dict[str, Any]],
    snapshot_ids: frozenset[str],
    ids_by_task_id: dict[str, int],
) -> list[int]:
    return [
        ids_by_task_id[task["id"]]
        for task in current
        if task["id"] in snapshot_ids and task.get("status") != TaskStatus.COMPLETED
    ]


def _split_target_specs(item: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return the validated list of split-target dicts on ``item``, or None.

    Empty list is treated as "no split" (matches ``merge_sources=[]`` —
    accepted as a no-op).  Returns ``None`` when ``split_targets`` is
    absent, not a list, or empty so the call site can short-circuit
    without re-parsing the shape.  The validator (run before this) has
    already rejected malformed shapes; this helper only filters the
    no-split case.
    """
    targets = item.get("split_targets")
    if not isinstance(targets, list) or not targets:
        return None
    return targets


def _allocate_split_child_ids(
    ordered_items: list[dict[str, Any]],
) -> dict[str, list[tuple[str, str]]]:
    """Pre-allocate a fresh ``(id, created_at)`` per declared split child.

    ``_apply_reorder`` runs the rescope plan twice — once via
    ``_apply_reorder_with_oracle`` to compute the result, once via
    ``_assert_rescope_matches_oracle`` to verify oracle/runtime
    agreement.  Both calls re-derive ops from the same ``ordered_items``,
    so the divergence assertion can only hold if the **id and the
    timestamp** are identical across the two passes.  Wall-clock reads
    in the materializer would make the verifier raise on any batch
    that crosses a second boundary; allocating both fields once at the
    top removes the race surface entirely.

    Ids come from :func:`uuid.uuid7`: a 48-bit ms timestamp prefix +
    74 bits of entropy per millisecond, so collisions with on-disk
    ids or with new ids minted by ``_make_new_tasks_from_opus`` in
    the same batch are statistically impossible — no shared
    used-set bookkeeping needed.
    """
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out: dict[str, list[tuple[str, str]]] = {}
    for item in ordered_items:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            continue
        targets = _split_target_specs(item)
        if targets is None:
            continue
        out[item_id] = [(str(uuid.uuid7()), created_at) for _ in targets]
    return out


def _split_source_ids(ordered_items: list[dict[str, Any]]) -> set[str]:
    """Items whose ``split_targets`` decomposes them — same suppression as merge.

    Mirror of :func:`_merge_source_ids`: a SplitTask flips the source's
    status to completed (per-task coverage invariant), but the original
    ask is still being honored — just decomposed across the children.
    A split isn't a scope change from the commenter's perspective, so
    the auto "covered by recent commits" notification doesn't apply.
    Per-source reply decisions belong to the reply-back filter epic
    (#1256) — its material-vs-aggregation classifier (#1723) explicitly
    treats split as one of the material cases, and per-intent
    notification (#1724) routes the chosen replies.
    """
    # The validator (run before reorder_tasks calls this) rejects
    # split_targets on null/missing/non-string ids, so every
    # split-bearing item here has a valid string id by contract.
    return {
        item["id"] for item in ordered_items if _split_target_specs(item) is not None
    }


def _union_intents(*lists: list[int]) -> list[int]:
    """Insertion-ordered union of intent comment id lists (#1722).

    Used everywhere a task's ``contributing_intents`` field needs to
    accumulate intents across rescope rounds (e.g. merge target =
    op's intents ∪ each source's pre-existing intents) without
    losing chronology.
    """
    seen: set[int] = set()
    out: list[int] = []
    for src in lists:
        for value in src:
            if value not in seen:
                seen.add(value)
                out.append(value)
    return out


def _split_child_synthetic_task(
    source_task: dict[str, Any],
    child_id: str,
    child_title: str,
    child_description: str,
    created_at: str,
    op_contributing_intents: list[int],
) -> dict[str, Any]:
    """Build the synthetic original task dict for a split child.

    ``_materialize_rescope_oracle_result`` reads ``tasks_by_oracle_id``
    to assemble the persisted task: it copies this dict, then overwrites
    title/description/status/lineage from the oracle row.  Children
    inherit ``thread`` and ``type`` from the source so reply paths still
    reach the original commenter (the SplitTask reducer also folds the
    source's ``lineage_comments`` into each child via
    ``add_split_kids``, and the materializer syncs that back into
    ``thread.lineage_comment_ids`` whenever the lineage differs).

    ``created_at`` is supplied by the caller (computed once per
    ``_apply_reorder`` and shared across the apply/verify passes); a
    wall-clock read here would make the divergence verifier raise on
    any batch that straddles a second boundary (codex P1).

    ``op_contributing_intents`` (#1722) is the SplitTask op's own
    contributing-intent list — every child inherits the parent op's
    intents in addition to whatever intents already lived on the
    source task, so a downstream classifier can decide whether the
    split materially affected each commenter.

    The thread is shallow-copied per child: every ``lineage_comment_ids``
    write in this codebase rebuilds the list and re-assigns the slot
    rather than mutating in place, so the inherited list reference is
    safe to share — but the dict itself does get re-keyed on writes, so
    each child needs its own dict.
    """
    source_intents = source_task.get("contributing_intents") or []
    child: dict[str, Any] = {
        "id": child_id,
        "title": child_title,
        "type": source_task.get("type") or "spec",
        "description": child_description,
        "status": str(TaskStatus.PENDING),
        "created_at": created_at,
        "contributing_intents": _union_intents(source_intents, op_contributing_intents),
    }
    source_thread = source_task.get("thread")
    if isinstance(source_thread, dict):
        child["thread"] = dict(source_thread)
    return child


def _rescope_releases_for_oracle(
    current: list[dict[str, Any]],
    ordered_items: list[dict[str, Any]],
    snapshot_ids: frozenset[str],
    ids_by_task_id: dict[str, int],
    *,
    split_child_ids: dict[str, list[tuple[str, str]]],
    tasks_by_oracle_id: dict[int, dict[str, Any]],
) -> list[rescope_oracle.RescopeRelease]:
    """Translate Opus's rescope items into typed Rocq ops.

    ``split_child_ids`` is a pre-allocated mapping of ``source_id ->
    [(child_string_id, created_at), ...]`` (one fresh tuple per
    declared child), computed once by ``_allocate_split_child_ids`` so
    the apply and assert paths share the same ids AND timestamps and
    don't trip the divergence check.  For every emitted ``SplitTask``
    op this function also registers a synthetic original task dict
    per child in ``tasks_by_oracle_id`` so
    ``_materialize_rescope_oracle_result`` can read the inherited
    ``thread`` / ``type`` fields when assembling the persisted child.
    """
    ordered_by_id: dict[str, dict[str, Any]] = {}
    for item in ordered_items:
        if item.get("id") and item["id"] not in ordered_by_id:
            ordered_by_id[item["id"]] = item
    existing_by_id = _existing_titles_by_id(current)
    next_oracle_id = max(ids_by_task_id.values(), default=0) + 1
    releases: list[rescope_oracle.RescopeRelease] = []
    for task in current:
        task_id = task["id"]
        if task_id not in snapshot_ids or task.get("status") == TaskStatus.COMPLETED:
            continue
        oracle_id = ids_by_task_id[task_id]
        item = ordered_by_id.get(task_id)
        if item is None:
            # Opus omitted this task from its rescope output.  Treat as
            # "keep as-is", not "completed" (#1357 case A).  Completion is a
            # decision a worker turn must explicitly signal — it is not
            # something the rescope reducer is allowed to infer from
            # omission.  Legitimate consolidation/dedup still happens via the
            # explicit-completion path; omission here just means Opus didn't
            # speak for this task on this iteration.
            decision: rescope_oracle.RescopeOp = rescope_oracle.KeepTask(oracle_id)
        else:
            existing_title = task.get("title", "")
            existing_description = task.get("description", "")
            existing_anchor = _task_source_comment_for_oracle(task)
            new_title = _effective_title(item, existing_by_id)
            new_description = (
                item["description"] if "description" in item else existing_description
            )
            merge_sources = _merge_source_oracle_ids(item, ids_by_task_id)
            split_targets = _split_target_specs(item)
            # #1722: stamp the synthetic original-task dict with the
            # union of any pre-existing intents and the op's
            # contributing_intents so the materializer copies them
            # through to the persisted task.  For merges, sources'
            # existing intents fold into the target below.
            op_intents = item.get("contributing_intents") or []
            if op_intents:
                existing_intents = task.get("contributing_intents") or []
                tasks_by_oracle_id[oracle_id]["contributing_intents"] = _union_intents(
                    existing_intents, op_intents
                )
            # Op precedence (most structural first):
            #   1. Explicit completion (#1716): item.status == "completed".
            #      Removes the task — strictly more structural than any
            #      metadata change, so it preempts the others.  Omission
            #      still means keep-as-is (#1357), not delete.
            #   2. Merge (#1717): folds source lineages into target +
            #      rewrites target text.  The model only mutates the
            #      target row; source rows are closed by their own
            #      CompleteTask items in the same batch.
            #   3. Split (#1718): closes the source and spawns N children
            #      that inherit the source's lineage_comments and
            #      source_comment.  Mutually exclusive with merge / explicit
            #      completion (validator rejects co-occurrence).
            #   4. Anchor rewrite (#1714): re-targets the reply destination.
            #   5. Text rewrite (#1713): title/description.
            #   6. KeepTask: nothing changed.
            #
            # The model carries one op per task per batch, so a combined
            # request rides multiple rescope iterations — the highest-
            # precedence change lands first.
            if item.get("status") == str(TaskStatus.COMPLETED):
                decision: rescope_oracle.RescopeOp = rescope_oracle.CompleteTask(
                    oracle_id
                )
            elif merge_sources:
                decision = rescope_oracle.MergeTasks(
                    oracle_id, merge_sources, new_title, new_description
                )
                # #1722: fold every source's pre-existing intents into
                # the merge target's contributing_intents.  Without
                # this, intents that drove a source task to exist
                # would be invisible to the classifier on the merged
                # row, and the corresponding commenters wouldn't get a
                # reply when the merged target eventually completes.
                merged_intents = list(
                    tasks_by_oracle_id[oracle_id].get("contributing_intents") or []
                )
                for src_oracle_id in merge_sources:
                    src_task = tasks_by_oracle_id.get(src_oracle_id)
                    if src_task is not None:
                        merged_intents = _union_intents(
                            merged_intents, src_task.get("contributing_intents") or []
                        )
                if merged_intents:
                    tasks_by_oracle_id[oracle_id]["contributing_intents"] = (
                        merged_intents
                    )
            elif split_targets:
                # Allocator and validator both iterate the same
                # ``split_targets`` list, so the pre-allocated child
                # string ids and the targets agree on length by
                # construction; index access fails fast on any drift.
                child_allocations = split_child_ids[task_id]
                children_specs: list[rescope_oracle.SplitChild] = []
                for child_index, target in enumerate(split_targets):
                    child_string_id, child_created_at = child_allocations[child_index]
                    child_oracle_id = next_oracle_id
                    next_oracle_id += 1
                    child_title = _normalize_title(str(target.get("title") or ""))
                    child_description = str(target.get("description") or "")
                    tasks_by_oracle_id[child_oracle_id] = _split_child_synthetic_task(
                        task,
                        child_string_id,
                        child_title,
                        child_description,
                        child_created_at,
                        op_intents,
                    )
                    children_specs.append(
                        rescope_oracle.SplitChild(
                            child_task=child_oracle_id,
                            child_title=child_title,
                            child_description=child_description,
                        )
                    )
                decision = rescope_oracle.SplitTask(oracle_id, children_specs)
            elif (
                isinstance(task.get("thread"), dict)
                and _is_valid_anchor_id(item.get("anchor_comment_id"))
                and item["anchor_comment_id"] != existing_anchor
            ):
                # An ``anchor_comment_id`` on a non-thread task is garbage
                # and is ignored — fall through to the text path.  A non-
                # positive or boolean value is also garbage (Python's
                # ``bool`` is a subclass of ``int``; GitHub comment ids
                # are positive 64-bit ints).  The materialization step
                # preserves the previous anchor in lineage_comment_ids.
                decision = rescope_oracle.RewriteAnchor(
                    oracle_id, item["anchor_comment_id"]
                )
            elif new_title != existing_title or new_description != existing_description:
                decision = rescope_oracle.RewriteTask(
                    oracle_id, new_title, new_description
                )
            else:
                decision = rescope_oracle.KeepTask(oracle_id)
        releases.append(
            rescope_oracle.RescopeRelease(rescope_oracle.ReleaseACT(), decision)
        )
    return releases


def _materialize_rescope_oracle_result(
    oracle_order: list[int],
    oracle_rows: dict[int, rescope_oracle.TaskRow],
    tasks_by_oracle_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for oracle_id in oracle_order:
        task = dict(tasks_by_oracle_id[oracle_id])
        row = oracle_rows[oracle_id]
        task["title"] = row.title
        task["description"] = row.description
        if isinstance(row.status, rescope_oracle.StatusCompleted):
            task["status"] = str(TaskStatus.COMPLETED)
        elif isinstance(row.status, rescope_oracle.StatusBlocked):
            task["status"] = str(TaskStatus.BLOCKED)
        else:
            task["status"] = task.get("status", str(TaskStatus.PENDING))
        # #1714: when rescope changes the source-comment anchor, sync the
        # task's thread metadata.  The previous anchor moves into
        # lineage_comment_ids (preserved as origin metadata so reply-back
        # paths can still walk back to the original commenter) and the
        # new anchor becomes the primary comment_id reply/resolve paths
        # read.  Identity is the durable task id; anchor is mutable
        # metadata.  We only re-target to a non-None int — the adapter
        # never emits RewriteAnchor with None, and clearing a thread
        # task's anchor isn't a supported use case in #1714 scope.
        existing_thread = task.get("thread")
        new_anchor = row.source_comment
        # Compare via _task_source_comment_for_oracle so a legacy ``'42'``
        # string in tasks.json compares equal to the oracle-materialized
        # ``42`` (codex on #1731).  Without normalization a no-op rescope
        # would still trip the re-anchor path and drop per-comment
        # metadata even though the anchor didn't actually change.
        if (
            isinstance(existing_thread, dict)
            and isinstance(new_anchor, int)
            and new_anchor != _task_source_comment_for_oracle(task)
        ):
            task["thread"] = _reanchored_thread(existing_thread, new_anchor)
        # #1717: when a MergeTasks op folded source lineages into this
        # row, sync the larger lineage back to thread.lineage_comment_ids
        # so reply-back paths see every contributing commenter.  Only
        # mutate when the model row's lineage_comments differs from what
        # the existing thread already canonicalises to — otherwise a
        # no-op rescope (e.g. a task with comment_id=42 and no explicit
        # lineage_comment_ids) would synthesise a lineage_comment_ids
        # field on every pass.
        existing_thread = task.get("thread")
        if isinstance(existing_thread, dict):
            merged_lineage = list(row.lineage_comments)
            existing_lineage = _thread_lineage_comment_ids(existing_thread)
            if merged_lineage != existing_lineage:
                thread = dict(task["thread"])
                thread["lineage_comment_ids"] = merged_lineage
                task["thread"] = thread
        materialized.append(task)
    return materialized


def _reanchored_thread(
    existing_thread: dict[str, Any], new_anchor: int
) -> dict[str, Any]:
    """Return a copy of ``existing_thread`` re-anchored to ``new_anchor``.

    Per-comment fields populated from the old anchor (``url``, ``author``,
    review-thread ``path`` / ``line`` / ``diff_hunk``, the legacy
    ``lineage_key`` chain identifier) are dropped because they no longer
    describe the new anchor.  Worker code that needs them must re-fetch
    via the new ``comment_id`` rather than read stale metadata.

    ``comment_type`` is preserved.  It tags the lane (``"pulls"`` for
    review-thread comments, ``"issues"`` for top-level PR/issue comments)
    and ``_notify_thread_change`` reads it to choose the correct GitHub
    API for the rescope reply — a missing value defaults to ``"issues"``
    and silently drops review-thread notifications.  In #1714 scope the
    soft adapter contract (``anchor_comment_id``) carries no kind hint;
    re-anchors are assumed to stay within the same lane.  Cross-lane
    re-anchoring needs an explicit kind in the rescope item, which is
    #1247 territory.
    """
    refreshed = {
        key: value
        for key, value in existing_thread.items()
        if key not in _STALE_AFTER_REANCHOR
    }
    refreshed["comment_id"] = new_anchor
    # lineage_comment_ids is owned by the model row's lineage_comments
    # field (#1717): the RewriteAnchor reducer extends it with the new
    # anchor, and the materializer below syncs the row back to thread.
    # Don't double-write here.
    return refreshed


_STALE_AFTER_REANCHOR = frozenset(
    {"url", "author", "lineage_key", "path", "line", "diff_hunk"}
)


def _assert_rescope_matches_oracle(
    current: list[dict[str, Any]],
    ordered_items: list[dict[str, Any]],
    snapshot_ids: frozenset[str],
    result: list[dict[str, Any]],
    split_child_ids: dict[str, list[tuple[str, str]]] | None = None,
) -> None:
    ids_by_task_id, tasks_by_oracle_id, current_order, rows = _rescope_state_for_oracle(
        current
    )
    snapshot_order = _rescope_snapshot_order_for_oracle(
        current, snapshot_ids, ids_by_task_id
    )
    releases = _rescope_releases_for_oracle(
        current,
        ordered_items,
        snapshot_ids,
        ids_by_task_id,
        split_child_ids=split_child_ids
        if split_child_ids is not None
        else _allocate_split_child_ids(ordered_items),
        tasks_by_oracle_id=tasks_by_oracle_id,
    )
    oracle_order, oracle_rows = rescope_oracle.apply_batched_rescope(
        snapshot_order, current_order, rows, releases
    )
    expected = _materialize_rescope_oracle_result(
        oracle_order, oracle_rows, tasks_by_oracle_id
    )
    if result != expected:
        raise AssertionError("rescope result diverged from Rocq oracle")


def _apply_reorder_with_oracle(
    current: list[dict[str, Any]],
    ordered_items: list[dict[str, Any]],
    snapshot_ids: frozenset[str],
    split_child_ids: dict[str, list[tuple[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    ids_by_task_id, tasks_by_oracle_id, current_order, rows = _rescope_state_for_oracle(
        current
    )
    snapshot_order = _rescope_snapshot_order_for_oracle(
        current, snapshot_ids, ids_by_task_id
    )
    releases = _rescope_releases_for_oracle(
        current,
        ordered_items,
        snapshot_ids,
        ids_by_task_id,
        split_child_ids=split_child_ids
        if split_child_ids is not None
        else _allocate_split_child_ids(ordered_items),
        tasks_by_oracle_id=tasks_by_oracle_id,
    )
    oracle_order, oracle_rows = rescope_oracle.apply_batched_rescope(
        snapshot_order, current_order, rows, releases
    )
    _assert_merge_lineage_preserved(releases, rows, oracle_rows)
    _assert_split_lineage_preserved(releases, rows, oracle_rows)
    return _materialize_rescope_oracle_result(
        oracle_order, oracle_rows, tasks_by_oracle_id
    )


def _assert_merge_lineage_preserved(
    releases: list[rescope_oracle.RescopeRelease],
    rows_before: dict[int, rescope_oracle.TaskRow],
    rows_after: dict[int, rescope_oracle.TaskRow],
) -> None:
    """Per-merge runtime assertion of the Rocq lineage-preservation predicate.

    The model's ``merge_preserves_source_lineage`` predicate proves "no
    source comment id is lost" computationally; we evaluate it for every
    actual MergeTasks op the adapter emits, so the per-run agreement
    between the executable predicate and the materialized output is
    surfaced as a fail-closed assertion (issue #1717 acceptance criteria
    "Rocq proves no source lineage comment id is lost; runtime oracle
    agrees with Python").  Any divergence raises immediately rather than
    silently dropping a source comment.
    """
    for release in releases:
        op = release.release_decision
        if not isinstance(op, rescope_oracle.MergeTasks):
            continue
        target_after = rows_after.get(op.task)
        if target_after is None:
            raise AssertionError(
                f"merge target {op.task} missing from rescope output rows"
            )
        if not rescope_oracle.merge_preserves_source_lineage(
            op.sources, rows_before, target_after
        ):
            raise AssertionError(
                f"merge into task {op.task} dropped source lineage "
                f"(sources={op.sources!r}); merge_preserves_source_lineage "
                "returned False"
            )


def _assert_split_lineage_preserved(
    releases: list[rescope_oracle.RescopeRelease],
    rows_before: dict[int, rescope_oracle.TaskRow],
    rows_after: dict[int, rescope_oracle.TaskRow],
) -> None:
    """Per-split runtime assertion of the Rocq lineage-preservation predicate.

    Mirror of :func:`_assert_merge_lineage_preserved` for the SplitTask
    op.  ``split_preserves_source_lineage`` proves "every child carries
    the source's lineage_comments verbatim and the same source_comment
    anchor"; we evaluate it for every emitted SplitTask op as a
    fail-closed runtime check (issue #1718 acceptance criteria
    "Rocq proves no source lineage comment id is lost for split;
    runtime oracle agrees with Python").
    """
    for release in releases:
        op = release.release_decision
        if not isinstance(op, rescope_oracle.SplitTask):
            continue
        source_before = rows_before.get(op.task)
        if source_before is None:
            raise AssertionError(
                f"split source {op.task} missing from rescope input rows"
            )
        child_ids = [c.child_task for c in op.children]
        if not rescope_oracle.split_preserves_source_lineage(
            child_ids, source_before, rows_after
        ):
            raise AssertionError(
                f"split of task {op.task} dropped child lineage "
                f"(children={child_ids!r}); split_preserves_source_lineage "
                "returned False"
            )


def _locked(path: Path) -> "_TaskFileLock":
    """Context manager: flock the task file for a shared read.

    Mutators go through :meth:`Tasks.modify` (inherited from
    :class:`~fido.state.JsonFileStore`), which holds an exclusive flock
    and fires :meth:`Tasks.on_mutate` automatically.  This helper is
    only used by the read-only :meth:`Tasks.list` /
    :meth:`Tasks.has_pending_for_comment` paths.
    """
    return _TaskFileLock(path)


class _TaskFileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.fd: IO[str] | None = None

    def __enter__(self) -> "_TaskFileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        self.fd = open(self._path, "r+")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()

    def _fd(self) -> IO[str]:
        assert self.fd is not None, "Lock used outside context manager"
        return self.fd

    def read(self) -> list[dict[str, Any]]:
        fd = self._fd()
        fd.seek(0)
        text = fd.read().strip()
        if not text:
            return []
        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"corrupt tasks.json: {e}") from e
        for t in result:
            if "type" not in t:
                raise ValueError(f"task {t.get('id', '?')} missing required type field")
        return result


def _format_work_queue(task_list: list[dict[str, Any]]) -> str:
    """Format a task list into work-queue markdown.

    Priority order: CI failures → everything else (preserving list order).
    Completed tasks appear in a collapsible ``<details>`` section.
    Each line includes a ``<!-- type:X -->`` HTML comment for round-tripping.
    """
    store, tasks_by_oracle_id = _task_store_for_oracle(task_list)
    projected_rows = task_store_oracle.project_task_store(store)
    pending: list[tuple[str, str]] = []
    completed: list[tuple[str, str]] = []

    def _fmt(row: task_store_oracle.PRBodyRow) -> tuple[str, str]:
        task = tasks_by_oracle_id[row.task]
        title = row.title
        url = (task.get("thread") or {}).get("url", "")
        task_type = task.get("type", TaskType.SPEC)
        display = f"[{title}]({url})" if url else title
        return display, task_type

    for row in projected_rows:
        display = _fmt(row)
        if isinstance(row.status, task_store_oracle.PRCompleted):
            completed.append(display)
        else:
            pending.append(display)

    lines: list[str] = []
    for i, (display, task_type) in enumerate(pending):
        suffix = " **→ next**" if i == 0 else ""
        lines.append(f"- [ ] {display}{suffix} <!-- type:{task_type} -->")

    if completed:
        lines.append("")
        lines.append(f"<details><summary>Completed ({len(completed)})</summary>")
        lines.append("")
        for display, task_type in completed:
            lines.append(f"- [x] {display} <!-- type:{task_type} -->")
        lines.append("</details>")

    return "\n".join(lines)


def _apply_queue_to_body(body: str, queue: str) -> str:
    """Replace the WORK_QUEUE_START/END section in a PR body with *queue*.

    Returns *body* unchanged if the markers are absent.
    """
    start_marker = "<!-- WORK_QUEUE_START -->"
    end_marker = "<!-- WORK_QUEUE_END -->"
    start = body.find(start_marker)
    end = body.find(end_marker)
    if start == -1 or end == -1:
        return body
    start += len(start_marker)
    return body[:start] + "\n" + queue + "\n" + body[end:]


def _auto_complete_ask_tasks(
    work_dir: Path,
    gh: GitHub,
    repo: str,
    pr_number: int | str,
) -> None:
    """Mark pending ASK tasks complete when their review thread is resolved."""
    tasks = Tasks(work_dir)
    task_list = tasks.list()
    ask_tasks = [
        t
        for t in task_list
        if t.get("status") == TaskStatus.PENDING
        and t.get("title", "").upper().startswith("ASK:")
        and t.get("thread")
    ]
    if not ask_tasks:
        return

    owner, repo_name = repo.split("/", 1)
    nodes = gh.get_review_threads(owner, repo_name, pr_number)

    resolved_ids: set[int] = set()
    for node in nodes:
        if node["isResolved"]:
            comments = node["comments"]["nodes"]
            if comments and comments[0].get("databaseId"):
                resolved_ids.add(int(comments[0]["databaseId"]))

    for task in ask_tasks:
        comment_id = (task.get("thread") or {}).get("comment_id")
        if comment_id and int(comment_id) in resolved_ids:
            log.info(
                "sync_tasks: ASK task thread resolved — completing: %s", task["title"]
            )
            tasks.complete_by_id(task["id"])


@contextmanager
def pr_body_lock(work_dir: Path) -> Iterator[None]:
    """Blocking exclusive lock on the PR-body sync.lock file.

    Acquires LOCK_EX (blocking, not LOCK_NB) so callers wait rather than
    skip.  Use to serialize any full-body PR edit against sync_tasks, which
    also acquires this same lock (with LOCK_NB).  Prevents a description
    rewrite from overwriting a concurrent work-queue sync.
    """
    git_dir = _resolve_git_dir(work_dir)
    fido_dir = git_dir / "fido"
    fido_dir.mkdir(parents=True, exist_ok=True)
    lock_path = fido_dir / "sync.lock"
    fd = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fd.close()


def sync_tasks(
    work_dir: Path,
    gh: GitHub,
    *,
    blocking: bool = False,
    _resolve_git_dir_fn: Callable[[Path], Path] = _resolve_git_dir,
    _auto_complete_ask_tasks_fn: Callable[..., None] = _auto_complete_ask_tasks,
) -> None:
    """Sync tasks.json → PR body work queue.

    Protected by a flock so concurrent calls don't race.  By default
    (``blocking=False``) a concurrent sync causes this call to silently skip.
    Pass ``blocking=True`` at authoritative call sites (e.g. post-completion)
    to wait for the lock instead — this guarantees the PR body converges even
    if a background sync holds the lock with stale data.
    """
    try:
        git_dir = _resolve_git_dir_fn(work_dir)
    except subprocess.CalledProcessError:
        log.warning("sync_tasks: could not resolve git dir for %s", work_dir)
        return

    fido_dir = git_dir / "fido"
    fido_dir.mkdir(parents=True, exist_ok=True)
    sync_lock_path = fido_dir / "sync.lock"
    sync_lock_fd = open(sync_lock_path, "w")  # noqa: SIM115
    if blocking:
        fcntl.flock(sync_lock_fd, fcntl.LOCK_EX)
    else:
        try:
            fcntl.flock(sync_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("sync_tasks: another sync running — skipping")
            sync_lock_fd.close()
            return

    try:
        state = State(fido_dir).load()
        issue = state.get("issue")
        if issue is None:
            log.info("sync_tasks: no current issue — nothing to sync")
            return

        repo = gh.get_repo_info(cwd=work_dir)
        user = gh.get_user()

        pr_data = gh.find_pr(repo, issue, user)
        if pr_data is None or pr_data.get("state") != "OPEN":
            log.info("sync_tasks: no open PR for issue #%s — nothing to sync", issue)
            return

        pr_number = pr_data["number"]
        _auto_complete_ask_tasks_fn(work_dir, gh, repo, pr_number)

        task_list = Tasks(work_dir).list()
        if not task_list:
            log.info("sync_tasks: no tasks — nothing to sync")
            return

        queue = _format_work_queue(task_list)
        log.info("sync_tasks: syncing task list → PR #%s", pr_number)

        body = gh.get_pr_body(repo, pr_number)

        has_start = "WORK_QUEUE_START" in body
        has_end = "WORK_QUEUE_END" in body
        if not has_start and not has_end:
            log.info(
                "sync_tasks: PR #%s has no work queue markers — skipping",
                pr_number,
            )
            return
        if not has_start or not has_end:
            log.warning(
                "sync_tasks: PR #%s has incomplete work queue markers "
                "(start=%s end=%s) — skipping",
                pr_number,
                has_start,
                has_end,
            )
            return

        new_body = _apply_queue_to_body(body, queue)
        if new_body == body:
            log.info(
                "sync_tasks: PR #%s work queue already up to date — no change",
                pr_number,
            )
            return
        gh.edit_pr_body(repo, pr_number, new_body)
        log.info("sync_tasks: PR #%s work queue synced", pr_number)
    finally:
        sync_lock_fd.close()


# ── Explicit rescope operation schema (#1719) ─────────────────────────────────
#
# The rescope output is a typed sum of operations over the snapped task
# queue.  Each operation says exactly what it does — no "infer mutation
# from omission" guessing.  The parser below decodes a strict
# {"operations": [...]} envelope into this dataclass union and collects
# every malformation in one pass so a follow-up nudge can hand Opus the
# full complaint list (rather than fail-on-first).
#
# The translator below converts the typed operations back into the
# existing item-dict shape that the validator/reducer/materializer
# pipeline already consumes — so this leaf only redesigns the I/O
# contract with Opus, not the downstream apply path.


# #1722 — every operation may carry a list of contributing
# RescopeIntent comment ids: the originating intents that drove this
# rescope decision.  The classifier in #1723 reads these to decide
# notification behavior (material rescope vs. pure aggregation
# per intent).  Empty list = the model didn't attribute the op to
# any specific intent (typical for KeepTask omissions and for
# implicit-context decisions); the field is optional in the wire
# schema.
_RescopeIntentIds = list[int]


@dataclass(frozen=True)
class _RescopeOpKeep:
    """Keep an existing task unchanged."""

    id: str
    contributing_intents: _RescopeIntentIds


@dataclass(frozen=True)
class _RescopeOpRewrite:
    """Rewrite an existing task's title and/or description."""

    id: str
    title: str
    description: str
    contributing_intents: _RescopeIntentIds


@dataclass(frozen=True)
class _RescopeOpRewriteAnchor:
    """Re-target an existing task's source-comment anchor (#1714)."""

    id: str
    anchor_comment_id: int
    contributing_intents: _RescopeIntentIds


@dataclass(frozen=True)
class _RescopeOpRemove:
    """Close an existing task (rocq CompleteTask)."""

    id: str
    contributing_intents: _RescopeIntentIds


@dataclass(frozen=True)
class _RescopeOpMerge:
    """Fold sources' lineage into target (rocq MergeTasks).  Sources close."""

    target_id: str
    sources: list[str]
    title: str
    description: str
    contributing_intents: _RescopeIntentIds


@dataclass(frozen=True)
class _RescopeOpSplitChild:
    title: str
    description: str


@dataclass(frozen=True)
class _RescopeOpSplit:
    """Close source and spawn N children inheriting its lineage (rocq SplitTask)."""

    id: str
    children: list[_RescopeOpSplitChild]
    contributing_intents: _RescopeIntentIds


@dataclass(frozen=True)
class _RescopeOpNew:
    """Create a brand-new task with a fresh id."""

    title: str
    description: str
    type: str  # noqa: A003 — schema field name
    contributing_intents: _RescopeIntentIds


_RescopeOp = (
    _RescopeOpKeep
    | _RescopeOpRewrite
    | _RescopeOpRewriteAnchor
    | _RescopeOpRemove
    | _RescopeOpMerge
    | _RescopeOpSplit
    | _RescopeOpNew
)


def _build_op_inputs(
    operations: list[_RescopeOp],
    ids_by_task_id: dict[str, int],
) -> list[rescope_oracle.OpInput]:
    """Adapter glue: parsed ``_RescopeOp`` list → oracle ``OpInput``.

    Shape conversion only — no rule logic.  The oracle's
    ``reply_back_intents_for`` consumes the resulting list and applies
    op classification (incl. merge/split reorganize override) and the
    reply-back rule (later cross-author destructive change) internally.

    ``_RescopeOpNew`` ops are skipped — they create brand-new tasks
    with no pre-existing intent contributions, so they can never
    trigger reply-back.
    """
    op_inputs: list[rescope_oracle.OpInput] = []
    for op in operations:
        model_op: rescope_oracle.RescopeOp
        if isinstance(op, _RescopeOpKeep):
            model_op = rescope_oracle.KeepTask(ids_by_task_id[op.id])
        elif isinstance(op, _RescopeOpRewrite):
            model_op = rescope_oracle.RewriteTask(
                ids_by_task_id[op.id], op.title, op.description
            )
        elif isinstance(op, _RescopeOpRewriteAnchor):
            model_op = rescope_oracle.RewriteAnchor(
                ids_by_task_id[op.id], op.anchor_comment_id
            )
        elif isinstance(op, _RescopeOpRemove):
            model_op = rescope_oracle.CompleteTask(ids_by_task_id[op.id])
        elif isinstance(op, _RescopeOpMerge):
            model_op = rescope_oracle.MergeTasks(
                ids_by_task_id[op.target_id],
                [ids_by_task_id[s] for s in op.sources],
                op.title,
                op.description,
            )
        elif isinstance(op, _RescopeOpSplit):
            # Split children's positives don't need to be allocated for
            # the reply-back rule — the oracle treats SplitTask as
            # EffectReorganize on the source regardless of child ids.
            children = [
                rescope_oracle.SplitChild(
                    child_task=0,
                    child_title=child.title,
                    child_description=child.description,
                )
                for child in op.children
            ]
            model_op = rescope_oracle.SplitTask(ids_by_task_id[op.id], children)
        else:  # _RescopeOpNew — no existing task, no contribution lineage.
            continue
        op_inputs.append(
            rescope_oracle.OpInput(
                oi_op=model_op,
                oi_intents=list(op.contributing_intents),
            )
        )
    return op_inputs


def _parse_rescope_operations(
    raw: str,
) -> tuple[list[_RescopeOp], list[str]]:
    """Parse the explicit rescope-operation envelope, collecting every error.

    The strict schema is::

        {"operations": [
          {"op": "keep", "id": "..."},
          {"op": "rewrite", "id": "...", "title": "...", "description": "..."},
          {"op": "rewrite_anchor", "id": "...", "anchor_comment_id": 12345},
          {"op": "remove", "id": "..."},
          {"op": "merge", "target_id": "...", "sources": ["..."],
                          "title": "...", "description": "..."},
          {"op": "split", "id": "...",
                          "children": [{"title": "...", "description": "..."}]},
          {"op": "new", "title": "...", "description": "...", "type": "spec"}
        ]}

    Every malformation found is appended to the returned error list so
    one parse pass produces the full complaint set the retry nudge
    sends back to Opus (Rob: "useful retries that detail what was
    wrong, as many things as we can find at once").  When errors are
    non-empty the returned op list is also empty — callers must reject
    the whole batch rather than partially apply.
    """
    errors: list[str] = []
    envelope = _decode_first_json_object(raw)
    if envelope is None:
        return [], ["response: no JSON object found"]
    if "operations" not in envelope:
        return [], ['response: missing top-level "operations" array']
    raw_ops = envelope["operations"]
    if not isinstance(raw_ops, list):
        return [], [
            f"response.operations: must be a list, got {type(raw_ops).__name__}"
        ]
    operations: list[_RescopeOp] = []
    for index, raw_op in enumerate(raw_ops):
        path = f"operations[{index}]"
        if not isinstance(raw_op, dict):
            errors.append(f"{path}: must be a dict, got {type(raw_op).__name__}")
            continue
        op_name = raw_op.get("op")
        if not isinstance(op_name, str):
            errors.append(f"{path}.op: must be a string, got {op_name!r}")
            continue
        parser = _OP_PARSERS.get(op_name)
        if parser is None:
            errors.append(
                f"{path}.op: unknown operation {op_name!r} (expected one of "
                "keep, rewrite, rewrite_anchor, remove, merge, split, new)"
            )
            continue
        op_or_none, op_errors = parser(raw_op, path)
        errors.extend(op_errors)
        if op_or_none is not None:
            operations.append(op_or_none)
    if errors:
        return [], errors
    return operations, []


def _parse_rescope_verdicts(
    raw: str,
    intents: list[RescopeIntent],
) -> tuple[list[IntentVerdict], list[str]]:
    """Parse the intent-first rescope envelope into :class:`IntentVerdict` list.

    Schema (per #1798 INV-D)::

        {"verdicts": [
          {
            "intent_comment_id": <int — one of the batch's comment ids>,
            "outcome": "honored" | "reshaped" | "superseded" | "no_op",
            "ops": [<op record>, ...],            // optional, default []
            "affected_task_ids": ["T1", ...],     // optional, default []
            "by_intent_comment_id": <int | null>, // optional, null default
            "narrative": "..." | null             // optional, null default
          },
          ...
        ]}

    Validations (collected one-pass — same all-errors-at-once contract
    as :func:`_parse_rescope_operations`):

    * Top-level envelope shape (``{"verdicts": [...]}``).
    * Each verdict entry is a dict.
    * Per-verdict field types delegated to
      :class:`~fido.types.IntentVerdict`'s ``__post_init__`` — every
      ``TypeError`` / ``ValueError`` it raises gets captured as a
      parse error, not propagated.
    * **Coverage**: every intent in *intents* has exactly one verdict
      (no missing, no duplicate ``intent_comment_id``).
    * **In-batch references**: ``intent_comment_id`` and (when set)
      ``by_intent_comment_id`` must name a :class:`RescopeIntent`
      ``comment_id`` from *intents* — no inventing ids.
    * **Acyclic supersedence**: the directed graph
      ``intent_comment_id -> by_intent_comment_id`` over the batch's
      verdicts must be acyclic.  Self-loops are already rejected by
      :class:`IntentVerdict` itself.

    When errors are non-empty the returned verdict list is also empty
    — callers must reject the whole batch rather than partially apply
    (mirrors :func:`_parse_rescope_operations`).

    The op records inside each verdict's ``ops`` are passed through
    as-is — :func:`_parse_rescope_operations` (or its successor) is
    responsible for op-level shape validation in a subsequent slice.
    """
    errors: list[str] = []
    envelope = _decode_first_json_object(raw)
    if envelope is None:
        return [], ["response: no JSON object found"]
    if "verdicts" not in envelope:
        return [], ['response: missing top-level "verdicts" array']
    raw_verdicts = envelope["verdicts"]
    if not isinstance(raw_verdicts, list):
        return [], [
            f"response.verdicts: must be a list, got {type(raw_verdicts).__name__}"
        ]

    # codex P2 on PR #1809: detect duplicate ``RescopeIntent.comment_id``
    # values in the input batch up front.  If we ``set()``-collapsed
    # them downstream, a single verdict could "cover" both entries
    # silently, defeating the one-verdict-per-intent invariant
    # (callers that coalesce intent lists across comments may merge
    # without dedup).  Fail closed: name every duplicated id so the
    # upstream coalescer bug is debuggable on sight.
    intent_id_list = [intent.comment_id for intent in intents]
    intent_ids = set(intent_id_list)
    if len(intent_id_list) != len(intent_ids):
        seen: set[int] = set()
        duplicates: list[int] = []
        for cid in intent_id_list:
            if cid in seen and cid not in duplicates:
                duplicates.append(cid)
            seen.add(cid)
        return [], [
            "intents: duplicate comment_id(s) in input batch — coalescer must "
            f"dedup before calling the parser: {sorted(duplicates)}"
        ]
    verdicts: list[IntentVerdict] = []
    seen_intent_ids: set[int] = set()

    for index, raw_v in enumerate(raw_verdicts):
        path = f"verdicts[{index}]"
        if not isinstance(raw_v, dict):
            errors.append(f"{path}: must be a dict, got {type(raw_v).__name__}")
            continue
        if "intent_comment_id" not in raw_v:
            errors.append(f"{path}: missing required field 'intent_comment_id'")
            continue
        if "outcome" not in raw_v:
            errors.append(f"{path}: missing required field 'outcome'")
            continue
        intent_id = raw_v["intent_comment_id"]
        # Catch in-batch reference errors before construction so the
        # caller learns about *every* unknown comment_id rather than
        # just the first the IntentVerdict ctor checked.
        if isinstance(intent_id, int) and not isinstance(intent_id, bool):
            if intent_id not in intent_ids:
                errors.append(
                    f"{path}.intent_comment_id: {intent_id} not in batch "
                    f"(expected one of {sorted(intent_ids)})"
                )
                continue
            if intent_id in seen_intent_ids:
                errors.append(
                    f"{path}.intent_comment_id: {intent_id} already covered by "
                    "an earlier verdict (each intent gets exactly one verdict)"
                )
                continue
        by_id = raw_v.get("by_intent_comment_id")
        if (
            by_id is not None
            and isinstance(by_id, int)
            and not isinstance(by_id, bool)
            and by_id not in intent_ids
        ):
            errors.append(
                f"{path}.by_intent_comment_id: {by_id} not in batch "
                f"(expected one of {sorted(intent_ids)})"
            )
            continue
        # codex P2 on slice 2: pass raw fields straight to the
        # ctor — DO NOT collapse with ``or ()`` or pre-wrap with
        # ``tuple(...)``.  Both would silently coerce malformed
        # falsy inputs (``{}``, ``""``, ``None``, ``0``) into an
        # empty tuple and bypass ``IntentVerdict.__post_init__``'s
        # type rejection.  Absent fields get an empty-tuple default
        # via ``dict.get``; present-with-malformed-value fields
        # reach the ctor and raise a clean TypeError that we
        # capture into ``errors``.
        try:
            verdict = IntentVerdict(
                intent_comment_id=intent_id,
                outcome=raw_v["outcome"],
                ops=raw_v.get("ops", ()),
                affected_task_ids=raw_v.get("affected_task_ids", ()),
                by_intent_comment_id=by_id,
                narrative=raw_v.get("narrative"),
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        verdicts.append(verdict)
        seen_intent_ids.add(verdict.intent_comment_id)

    # Coverage: every intent in the batch must have a verdict.  Don't
    # short-circuit on earlier errors — list every missing id so Opus
    # gets the full picture on the nudge.
    missing = sorted(intent_ids - seen_intent_ids)
    for intent_id in missing:
        errors.append(
            f"verdicts: missing verdict for intent_comment_id {intent_id} "
            "(every intent in the batch needs exactly one verdict)"
        )

    # Acyclic check on the supersedence graph (intent → superseder).
    if not errors:
        cycle = _find_supersedence_cycle(verdicts)
        if cycle is not None:
            errors.append(
                "verdicts: supersedence graph has a cycle: "
                + " -> ".join(str(cid) for cid in cycle)
            )

    if errors:
        return [], errors
    return verdicts, []


def _find_supersedence_cycle(
    verdicts: list[IntentVerdict],
) -> list[int] | None:
    """Return one cycle's id sequence in the supersedence graph, or None.

    Walks each ``by_intent_comment_id`` edge and detects revisits.
    Returns the cycle as ``[a, b, c, a]`` so the error message reads
    naturally; ``None`` when the graph is acyclic.
    """
    by_pointer: dict[int, int] = {
        v.intent_comment_id: v.by_intent_comment_id
        for v in verdicts
        if v.by_intent_comment_id is not None
    }
    for start in by_pointer:
        seen: list[int] = [start]
        cursor: int | None = by_pointer.get(start)
        while cursor is not None:
            if cursor in seen:
                seen.append(cursor)
                # Trim to the cycle's start so the message is the
                # smallest visible loop, not its tail.
                first = seen.index(cursor)
                return seen[first:]
            seen.append(cursor)
            cursor = by_pointer.get(cursor)
    return None


def _flatten_verdicts_to_ops(
    verdicts: list[IntentVerdict],
) -> list[dict[str, Any]]:
    """Flatten per-verdict ops into the flat op list ``_apply_reorder`` consumes.

    Each emitted op inherits its originating verdict's
    ``intent_comment_id`` via the ``contributing_intents`` field (the
    #1722 attribution convention; any pre-existing well-formed
    ``contributing_intents`` on the op is unioned with the verdict's
    own id so explicit per-op attribution from Opus is preserved
    alongside the verdict-level one).

    Malformed ``contributing_intents`` shapes are **passed through
    untouched** so :func:`_parse_rescope_operations` raises the
    schema error on the next step rather than this helper silently
    coercing (codex P1/P2 on PR #1813: ``or []`` + set-union would
    turn ``42`` / ``""`` / ``{...}`` / ``(...)`` / mixed lists into
    a "valid" sorted int list, bypassing the fail-closed parse
    boundary).  Only two shapes are touched here:

    * field absent → set to ``[verdict.intent_comment_id]``
    * existing list of int (rejecting ``bool``, an ``int`` subclass)
      → union with verdict id, sorted

    Anything else stays as-is for the op parser to reject.

    Used by :func:`reorder_tasks` when ``intents`` is non-empty so the
    verdict-shaped parser output can flow through the existing op
    apply/validate machinery without duplicating op-level schema
    checks (#1812 INV-D wiring).
    """
    flat: list[dict[str, Any]] = []
    for verdict in verdicts:
        for op_mapping in verdict.ops:
            op = dict(op_mapping)
            if "contributing_intents" not in op:
                op["contributing_intents"] = [verdict.intent_comment_id]
            else:
                existing = op["contributing_intents"]
                # ``IntentVerdict.__post_init__`` ran ``deep_freeze``
                # which turned any list inside ``ops`` into a tuple,
                # so the well-formed shape we see here is
                # ``tuple[int, ...]`` (not ``list[int]``).  Accept
                # both for resilience; reject ``bool`` explicitly
                # (it's an ``int`` subclass and would otherwise sneak
                # through as attribution).
                if isinstance(existing, (list, tuple)) and all(
                    isinstance(x, int) and not isinstance(x, bool)
                    for x in existing  # pyright: ignore[reportUnknownVariableType]
                ):
                    op["contributing_intents"] = sorted(
                        {*existing, verdict.intent_comment_id}
                    )
                # Else: leave malformed value untouched so
                # _parse_rescope_operations rejects it at the next step.
            flat.append(op)
    return flat


def _decode_first_json_object(raw: str) -> dict[str, Any] | None:
    """Find and decode the first top-level JSON object in *raw*.

    Same scan loop the legacy parser used; left here so the caller can
    distinguish "no JSON at all" (parser-error nudge fires) from
    "decoded JSON but the schema is wrong" (per-op errors fire).
    """
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        brace = raw.find("{", pos)
        if brace == -1:
            return None
        try:
            obj, _end = decoder.raw_decode(raw, brace)
        except json.JSONDecodeError:
            pos = brace + 1
            continue
        # raw_decode starting at '{' always yields a dict (JSON object);
        # the only non-error outcome is a parsed object.
        return obj


def _require_string_field(
    raw_op: dict[str, Any], field: str, path: str, errors: list[str]
) -> str | None:
    value = raw_op.get(field)
    if not isinstance(value, str) or not value:
        errors.append(f"{path}.{field}: must be a non-empty string, got {value!r}")
        return None
    return value


def _require_string_field_allow_empty(
    raw_op: dict[str, Any], field: str, path: str, errors: list[str]
) -> str | None:
    value = raw_op.get(field)
    if not isinstance(value, str):
        errors.append(f"{path}.{field}: must be a string, got {value!r}")
        return None
    return value


def _parse_contributing_intents(
    raw_op: dict[str, Any], path: str, errors: list[str]
) -> _RescopeIntentIds:
    """Validate the optional ``contributing_intents`` field on every op (#1722).

    Missing or empty list = the model didn't attribute this op to any
    specific RescopeIntent (typical for KeepTask omissions and for
    decisions Opus made from implicit context).  Present means a list
    of positive int comment ids — duplicate entries are de-duplicated
    in arrival order so the materializer can persist a clean
    insertion-ordered set.
    """
    raw = raw_op.get("contributing_intents")
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append(
            f"{path}.contributing_intents: must be a list of positive ints "
            f"(intent comment ids), got {type(raw).__name__}"
        )
        return []
    seen: set[int] = set()
    out: _RescopeIntentIds = []
    for index, value in enumerate(raw):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(
                f"{path}.contributing_intents[{index}]: must be a positive "
                f"int (intent comment id), got {value!r}"
            )
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _parse_op_keep(
    raw_op: dict[str, Any], path: str
) -> tuple[_RescopeOp | None, list[str]]:
    errors: list[str] = []
    task_id = _require_string_field(raw_op, "id", path, errors)
    intents = _parse_contributing_intents(raw_op, path, errors)
    if task_id is None:
        return None, errors
    return _RescopeOpKeep(id=task_id, contributing_intents=intents), errors


def _parse_op_rewrite(
    raw_op: dict[str, Any], path: str
) -> tuple[_RescopeOp | None, list[str]]:
    errors: list[str] = []
    task_id = _require_string_field(raw_op, "id", path, errors)
    title = _require_string_field(raw_op, "title", path, errors)
    description = _require_string_field_allow_empty(raw_op, "description", path, errors)
    intents = _parse_contributing_intents(raw_op, path, errors)
    if task_id is None or title is None or description is None:
        return None, errors
    return (
        _RescopeOpRewrite(
            id=task_id,
            title=title,
            description=description,
            contributing_intents=intents,
        ),
        errors,
    )


def _parse_op_rewrite_anchor(
    raw_op: dict[str, Any], path: str
) -> tuple[_RescopeOp | None, list[str]]:
    errors: list[str] = []
    task_id = _require_string_field(raw_op, "id", path, errors)
    anchor = raw_op.get("anchor_comment_id")
    if not _is_valid_anchor_id(anchor):
        errors.append(
            f"{path}.anchor_comment_id: must be a positive int "
            f"(GitHub comment id), got {anchor!r}"
        )
        anchor = None
    intents = _parse_contributing_intents(raw_op, path, errors)
    if task_id is None or anchor is None:
        return None, errors
    return (
        _RescopeOpRewriteAnchor(
            id=task_id,
            anchor_comment_id=anchor,
            contributing_intents=intents,
        ),
        errors,
    )


def _parse_op_remove(
    raw_op: dict[str, Any], path: str
) -> tuple[_RescopeOp | None, list[str]]:
    errors: list[str] = []
    task_id = _require_string_field(raw_op, "id", path, errors)
    intents = _parse_contributing_intents(raw_op, path, errors)
    if task_id is None:
        return None, errors
    return _RescopeOpRemove(id=task_id, contributing_intents=intents), errors


def _parse_op_merge(
    raw_op: dict[str, Any], path: str
) -> tuple[_RescopeOp | None, list[str]]:
    errors: list[str] = []
    target = _require_string_field(raw_op, "target_id", path, errors)
    title = _require_string_field(raw_op, "title", path, errors)
    description = _require_string_field_allow_empty(raw_op, "description", path, errors)
    raw_sources = raw_op.get("sources")
    sources: list[str] = []
    if not isinstance(raw_sources, list) or not raw_sources:
        errors.append(
            f"{path}.sources: must be a non-empty list of task ids, got {raw_sources!r}"
        )
    else:
        for src_index, source in enumerate(raw_sources):
            if not isinstance(source, str) or not source:
                errors.append(
                    f"{path}.sources[{src_index}]: must be a non-empty string, "
                    f"got {source!r}"
                )
                continue
            sources.append(source)
        if not sources:
            errors.append(
                f"{path}.sources: every entry was malformed; merge needs at "
                "least one valid source id"
            )
    intents = _parse_contributing_intents(raw_op, path, errors)
    if target is None or title is None or description is None or not sources:
        return None, errors
    return (
        _RescopeOpMerge(
            target_id=target,
            sources=sources,
            title=title,
            description=description,
            contributing_intents=intents,
        ),
        errors,
    )


def _parse_op_split(
    raw_op: dict[str, Any], path: str
) -> tuple[_RescopeOp | None, list[str]]:
    errors: list[str] = []
    task_id = _require_string_field(raw_op, "id", path, errors)
    raw_children = raw_op.get("children")
    children: list[_RescopeOpSplitChild] = []
    if not isinstance(raw_children, list) or not raw_children:
        errors.append(
            f"{path}.children: must be a non-empty list of "
            f'{{"title": ..., "description": ...}} dicts, got {raw_children!r}'
        )
    else:
        for child_index, raw_child in enumerate(raw_children):
            child_path = f"{path}.children[{child_index}]"
            if not isinstance(raw_child, dict):
                errors.append(
                    f"{child_path}: must be a dict, got {type(raw_child).__name__}"
                )
                continue
            child_title = _require_string_field(raw_child, "title", child_path, errors)
            child_description = _require_string_field_allow_empty(
                raw_child, "description", child_path, errors
            )
            if child_title is not None and child_description is not None:
                children.append(
                    _RescopeOpSplitChild(
                        title=child_title, description=child_description
                    )
                )
        if not children:
            errors.append(
                f"{path}.children: every child was malformed; split needs at "
                "least one valid child"
            )
    intents = _parse_contributing_intents(raw_op, path, errors)
    if task_id is None or not children:
        return None, errors
    return (
        _RescopeOpSplit(id=task_id, children=children, contributing_intents=intents),
        errors,
    )


def _parse_op_new(
    raw_op: dict[str, Any], path: str
) -> tuple[_RescopeOp | None, list[str]]:
    errors: list[str] = []
    title = _require_string_field(raw_op, "title", path, errors)
    description = _require_string_field_allow_empty(raw_op, "description", path, errors)
    task_type = _require_string_field(raw_op, "type", path, errors)
    intents = _parse_contributing_intents(raw_op, path, errors)
    if title is None or description is None or task_type is None:
        return None, errors
    return (
        _RescopeOpNew(
            title=title,
            description=description,
            type=task_type,
            contributing_intents=intents,
        ),
        errors,
    )


_OP_PARSERS: dict[
    str, Callable[[dict[str, Any], str], tuple[_RescopeOp | None, list[str]]]
] = {
    "keep": _parse_op_keep,
    "rewrite": _parse_op_rewrite,
    "rewrite_anchor": _parse_op_rewrite_anchor,
    "remove": _parse_op_remove,
    "merge": _parse_op_merge,
    "split": _parse_op_split,
    "new": _parse_op_new,
}


def _find_cross_op_errors(
    operations: list[_RescopeOp], snapshot_ids: frozenset[str]
) -> list[str]:
    """Pre-reducer cross-op invariants from #1719's acceptance criteria.

    - Every existing-task op references an id in the current snapshot
      (unknown ids would silently no-op or error in the reducer).
    - No two ops claim the same snapshot id (per-task coverage; ambiguity
      between e.g. rewrite + remove on the same id).
    - No source id appears in more than one merge or split op (lineage
      duplication; the rocq invariants reject this too at the reducer).
    """
    errors: list[str] = []
    claim_count: dict[str, int] = {}

    def _claim(claimed_id: str, op_index: int) -> None:
        if claimed_id not in snapshot_ids:
            errors.append(
                f"operations[{op_index}]: id {claimed_id!r} is not in the "
                "pending snapshot (unknown task id — it may have been "
                "completed concurrently or never existed)"
            )
        claim_count[claimed_id] = claim_count.get(claimed_id, 0) + 1

    for index, op in enumerate(operations):
        match op:
            case _RescopeOpKeep(id=tid) | _RescopeOpRewrite(id=tid):
                _claim(tid, index)
            case _RescopeOpRewriteAnchor(id=tid):
                _claim(tid, index)
            case _RescopeOpRemove(id=tid):
                _claim(tid, index)
            case _RescopeOpMerge(target_id=tid, sources=sources):
                _claim(tid, index)
                for src in sources:
                    _claim(src, index)
            case _RescopeOpSplit(id=tid):
                _claim(tid, index)
            case _RescopeOpNew():
                pass

    for tid, count in claim_count.items():
        if count > 1:
            errors.append(
                f"id {tid!r}: claimed by {count} operations; each snapshot "
                "task may appear in at most one operation"
            )
    return errors


def _operations_to_items(operations: list[_RescopeOp]) -> list[dict[str, Any]]:
    """Convert typed operations into the item-dict shape the apply path uses.

    Lets the existing ``_validate_rescope_batch`` / ``_apply_reorder``
    pipeline keep working unchanged: each operation lowers to one or
    more items in the legacy schema (a merge expands into one target
    item plus N source-completion items, since the reducer's per-task
    coverage invariant requires every source to carry its own
    ``CompleteTask``).

    Each item carries the op's ``contributing_intents`` (a list of
    originating ``RescopeIntent.comment_id`` values; #1722) under the
    same key so the materializer can persist them on the resulting
    task.  Source-completion items synthesised by a merge expansion
    inherit the merge op's contributing_intents too — those intents
    drove the source's closure in addition to the target's mutation.
    """
    items: list[dict[str, Any]] = []
    for op in operations:
        match op:
            case _RescopeOpKeep(id=tid, contributing_intents=intents):
                items.append({"id": tid, "contributing_intents": list(intents)})
            case _RescopeOpRewrite(
                id=tid,
                title=title,
                description=desc,
                contributing_intents=intents,
            ):
                items.append(
                    {
                        "id": tid,
                        "title": title,
                        "description": desc,
                        "contributing_intents": list(intents),
                    }
                )
            case _RescopeOpRewriteAnchor(
                id=tid,
                anchor_comment_id=anchor,
                contributing_intents=intents,
            ):
                items.append(
                    {
                        "id": tid,
                        "anchor_comment_id": anchor,
                        "contributing_intents": list(intents),
                    }
                )
            case _RescopeOpRemove(id=tid, contributing_intents=intents):
                items.append(
                    {
                        "id": tid,
                        "status": str(TaskStatus.COMPLETED),
                        "contributing_intents": list(intents),
                    }
                )
            case _RescopeOpMerge(
                target_id=target,
                sources=sources,
                title=title,
                description=desc,
                contributing_intents=intents,
            ):
                items.append(
                    {
                        "id": target,
                        "title": title,
                        "description": desc,
                        "merge_sources": list(sources),
                        "contributing_intents": list(intents),
                    }
                )
                for src in sources:
                    items.append(
                        {
                            "id": src,
                            "status": str(TaskStatus.COMPLETED),
                            "contributing_intents": list(intents),
                        }
                    )
            case _RescopeOpSplit(
                id=tid, children=children, contributing_intents=intents
            ):
                items.append(
                    {
                        "id": tid,
                        "split_targets": [
                            {"title": c.title, "description": c.description}
                            for c in children
                        ],
                        "contributing_intents": list(intents),
                    }
                )
            case _RescopeOpNew(
                title=title,
                description=desc,
                type=task_type,
                contributing_intents=intents,
            ):
                items.append(
                    {
                        "id": None,
                        "title": title,
                        "description": desc,
                        "type": task_type,
                        "contributing_intents": list(intents),
                    }
                )
    return items


def _derive_thread_from_intents(
    op_intent_ids: list[int],
    intents: list[RescopeIntent] | None,
) -> dict[str, Any] | None:
    """Build a ``thread`` anchor for a rescope-spawned new task (#1843).

    Picks the first intent in *op_intent_ids* that resolves to a
    :class:`RescopeIntent` carrying ``repo`` + ``pr_number`` + ``comment_id``
    and returns ``{"repo", "pr", "comment_id"}``.  Returns ``None`` when no
    intent has the full context populated — synthetic test fixtures and
    legacy intents pre-#1843 may lack ``repo``/``pr_number`` and the
    caller leaves the task unanchored in that case.
    """
    if not op_intent_ids or not intents:
        return None
    by_cid = {i.comment_id: i for i in intents}
    for cid in op_intent_ids:
        intent = by_cid.get(cid)
        if intent is None or not intent.repo or intent.pr_number <= 0:
            continue
        return {
            "repo": intent.repo,
            "pr": intent.pr_number,
            "comment_id": intent.comment_id,
        }
    return None


def _make_new_tasks_from_opus(
    ordered_items: list[dict[str, Any]],
    snapshot_ids: frozenset[str],
    current: list[dict[str, Any]] | None = None,
    intents: list[RescopeIntent] | None = None,
) -> list[dict[str, Any]]:
    """Create fresh task dicts for items Opus returned with a null or absent id.

    Items with a non-null id (whether in the snapshot or not) are not treated
    as new — the caller handles snapshot ids via the oracle and ignores
    unrecognised string ids (unchanged behaviour).  Only items where the
    ``"id"`` key is absent or explicitly ``null`` are promoted to new tasks.

    Each new task receives a UUIDv7 id, ``status: "pending"``, and
    ``type: "spec"`` unless Opus specified a different type.  Items with
    blank titles are silently skipped.

    Dedup against post-snapshot thread tasks (#1337): when *current* and
    *intents* are provided, any rescope intent whose ``comment_id`` is already
    covered by a thread task added since the snapshot was taken is treated as
    "already serviced" — the entry-boundary ``create_task`` path produced the
    thread task while Opus was thinking, so any null-id item Opus emits for
    the same intent is a duplicate.  We suppress one null-id item per covered
    intent (in arrival order) to keep at most one task per intent.
    """
    covered_intents = 0
    if current is not None and intents:
        post_snapshot_lineage: set[int] = set()
        for task in current:
            if task.get("id") in snapshot_ids:
                continue
            for cid in _thread_lineage_comment_ids(task.get("thread")):
                post_snapshot_lineage.add(cid)
        for intent in intents:
            if intent.comment_id in post_snapshot_lineage:
                covered_intents += 1

    new_tasks: list[dict[str, Any]] = []
    skipped = 0
    for item in ordered_items:
        if "id" in item and item["id"] is not None:
            continue  # has an id — handled by oracle or ignored as unknown
        # Null-id (new task) items have no existing title to fall back to;
        # _effective_title with an empty existing-by-id map normalizes the
        # proposed value or returns "" for non-strings / blanks.  Same
        # normalization Tasks.add applies, so PR-body round-tripping holds.
        title = _effective_title(item, {})
        if not title:
            continue
        if skipped < covered_intents:
            log.info(
                "rescope: dropping duplicate new task %r — already serviced "
                "by a post-snapshot thread task (#1337)",
                title[:80],
            )
            skipped += 1
            continue
        task: dict[str, Any] = {
            "id": str(uuid.uuid7()),
            "title": title,
            "type": item.get("type") or "spec",
            "description": item.get("description") or "",
            "status": str(TaskStatus.PENDING),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # #1722: a brand-new task carries the originating intents the
        # `new` op declared so a downstream classifier can route the
        # eventual completion notification to the right commenter(s).
        op_intents = item.get("contributing_intents") or []
        if op_intents:
            task["contributing_intents"] = list(op_intents)
        # #1843: derive the thread anchor from the first contributing
        # intent that has repo + pr_number + comment_id populated.
        # Pre-INV-B the old ``events.create_task`` set ``thread`` directly;
        # in the intent path the rescope spawns the task and the only
        # carrier of the originating PR is the ``RescopeIntent`` itself.
        # Without this, reply-back has no idea where to post follow-ups
        # on the new task.
        thread = _derive_thread_from_intents(op_intents, intents)
        if thread is not None:
            task["thread"] = thread
        new_tasks.append(task)
    return new_tasks


def _validate_rescope_batch(
    current: list[dict[str, Any]],
    ordered_items: list[Any],
) -> list[str]:
    """Validate a rescope batch atomically.  Empty list = valid (#1715).

    Returns a list of error messages.  The whole batch is rejected if any
    check fails — partial commits would leave ``tasks.json`` in a state
    Opus didn't propose, defeating the snapshot/replay model.

    Checks:

    * Each item must be a dict.
    * An item with a non-null ``id`` must reference a task that currently
      exists in the queue (snapshot or post-snapshot).  An unknown id
      means Opus is operating on a hallucinated row; reject and let the
      next iteration retry against fresh state.
    * No two items may share the same non-null ``id`` (duplicate output —
      both would target the same task and the second would be silently
      dropped today).
    * A non-null ``id`` must be a non-empty string.

    Items with ``id is None`` (or absent) are new-task proposals; their
    own validation lives in :func:`_make_new_tasks_from_opus` (blank-title
    skip, etc.) — they are not rejected here.

    #1717 adds the merge-specific rules: ``item["merge_sources"]`` (when
    present) must be a list of strings naming OTHER existing task ids,
    each of which appears in the same batch as a separately-completed
    item.  The rocq reducer can fold any sources' lineage into the
    target row, but the per-task coverage invariant
    (``rescope_ops_cover_snapshot``) still requires every source to have
    its own ``CompleteTask`` op — so the source items in the batch must
    carry ``status="completed"``.

    Forward-looking: split (#1718) will add its own validation rules.
    """
    errors: list[str] = []
    known_ids = {t["id"] for t in current if "id" in t}
    # Lineage on disk lives in thread.lineage_comment_ids — a
    # non-thread target has nowhere to store comment origins.  If a
    # merge folds thread-bearing sources into a non-thread target, the
    # materializer can't sync row.lineage_comments and every source
    # comment id is silently lost, violating the "no source origin
    # lost" invariant this leaf is meant to enforce (codex P1 on #1738).
    # Same-kind merges (spec→spec, thread→thread) are fine; the rule
    # only fires when a thread source would feed a non-thread target.
    thread_bearing_ids = {
        t["id"] for t in current if isinstance(t.get("thread"), dict) and "id" in t
    }
    # Tasks already completed on disk: a merge into one of these would
    # be silently dropped by _rescope_releases_for_oracle (which skips
    # completed snapshot tasks) while _merge_source_ids would still
    # suppress the source's completion notification (codex on #1738).
    currently_completed_ids = {
        t["id"]
        for t in current
        if "id" in t and t.get("status") == str(TaskStatus.COMPLETED)
    }
    # Tasks currently blocked on disk: a merge into one of these
    # silently parks the merged work — the worker picker skips blocked
    # tasks (worker.py:_pick_next_task), so the source completes and
    # suppresses its notification but the merged target never runs
    # (codex on #1738).
    currently_blocked_ids = {
        t["id"]
        for t in current
        if "id" in t and t.get("status") == str(TaskStatus.BLOCKED)
    }
    # Kind classification is title-prefix driven (ASK:/DEFER:/CI FAILURE:),
    # so a split that copies child titles literally would silently
    # reclassify an ASK/DEFER source's children to executable spec
    # tasks (codex P1).  CI failure rows are atomic events that don't
    # decompose meaningfully.  Reject splits on any of these source
    # kinds; Opus has CompleteTask + new-task creation if it really
    # wants to convert one of these into actionable work.
    non_splittable_source_ids = {
        t["id"]
        for t in current
        if "id" in t
        and isinstance(
            _rescope_task_kind_for_oracle(t),
            rescope_oracle.TaskAsk | rescope_oracle.TaskDefer | rescope_oracle.TaskCI,
        )
    }
    seen_ids: set[str] = set()
    explicitly_completed_ids: set[str] = set()
    merge_targets: list[tuple[int, str, list[str]]] = []

    for index, item in enumerate(ordered_items):
        if not isinstance(item, dict):
            errors.append(f"item[{index}]: not a dict, got {type(item).__name__}")
            continue
        item_id = item.get("id")
        if item_id is None:
            # codex on #1738: a null-id item carrying merge_sources
            # bypasses every merge check below — _make_new_tasks_from_opus
            # creates the new task without folding lineage, and
            # _merge_source_ids still suppresses the source's completion
            # notification.  The source vanishes without lineage being
            # preserved anywhere.  Until new-target merge is implemented
            # end-to-end (a separate leaf), reject the shape.
            #
            # Use presence + value check rather than truthiness: a
            # malformed-but-falsy value (``""``, ``0``, ``False``) on
            # the merge_sources key still violates the fail-closed
            # contract — merge_sources is present but not a list, so the
            # whole rescope batch must reject, not partially apply (codex
            # follow-up on #1738).  The empty-list sentinel ``[]`` is
            # the documented "no merge" no-op and stays accepted.
            if "merge_sources" in item and item["merge_sources"] != []:
                errors.append(
                    f"item[{index}].merge_sources on a null/missing id: "
                    "merging into a not-yet-created task isn't implemented "
                    "(would lose source lineage); declare the target as an "
                    "existing task id or drop merge_sources"
                )
            continue
        if not isinstance(item_id, str) or not item_id:
            errors.append(
                f"item[{index}].id: must be non-empty string, got {item_id!r}"
            )
            continue
        if item_id not in known_ids:
            errors.append(
                f"item[{index}].id={item_id!r}: unknown source id "
                "(not in current task queue)"
            )
        if item_id in seen_ids:
            errors.append(
                f"item[{index}].id={item_id!r}: duplicate (already appears earlier)"
            )
        seen_ids.add(item_id)
        if item.get("status") == str(TaskStatus.COMPLETED):
            explicitly_completed_ids.add(item_id)
        merge_sources = item.get("merge_sources")
        if merge_sources is not None:
            if not isinstance(merge_sources, list):
                errors.append(
                    f"item[{index}].merge_sources: must be a list, "
                    f"got {type(merge_sources).__name__}"
                )
            elif merge_sources and (
                item.get("status") == str(TaskStatus.COMPLETED)
                or item_id in currently_completed_ids
            ):
                # codex on #1738: a non-empty merge_sources is silently
                # dropped two ways and both leave the source vanished
                # without lineage preserved:
                #
                # 1. Item carries status="completed" too — explicit-
                #    completion precedence (#1716) wins, MergeTasks
                #    never emitted, but _merge_source_ids still
                #    suppresses the source's completion notification.
                # 2. Target is already completed on disk —
                #    _rescope_releases_for_oracle skips completed
                #    snapshot tasks entirely, so MergeTasks again
                #    never emitted with the same suppressed-
                #    notification result.
                #
                # Reject both shapes at the validator: completing the
                # target means there is no pending work to absorb
                # merged sources.  Empty ``merge_sources`` is the
                # documented "no merge" sentinel and is harmless.
                errors.append(
                    f"item[{index}].merge_sources on {item_id!r}: target "
                    "is completed (either via this batch or already on "
                    "disk); merging into a completed task is contradictory "
                    "(no MergeTasks would run; the source's lineage would "
                    "be silently lost)"
                )
            elif merge_sources and item_id in currently_blocked_ids:
                # codex on #1738: a blocked target accepts the merge,
                # but the worker picker (_pick_next_task) skips blocked
                # tasks — the source flips to completed and its
                # completion notification gets suppressed, while the
                # merged work parks indefinitely on a row Fido won't
                # pick up.  Same shape of silent drop as the completed-
                # target case.  If Opus wants to merge into a blocked
                # task, the prompt vocabulary needs an explicit
                # unblock signal (#1247 territory); for now reject.
                errors.append(
                    f"item[{index}].merge_sources on {item_id!r}: target "
                    "is blocked; the worker picker skips blocked tasks, "
                    "so the merged work would never run while the "
                    "source's lineage was already absorbed and its "
                    "completion notification suppressed"
                )
            else:
                source_strs: list[str] = []
                for source_index, source in enumerate(merge_sources):
                    if not isinstance(source, str) or not source:
                        errors.append(
                            f"item[{index}].merge_sources[{source_index}]: "
                            f"must be non-empty string, got {source!r}"
                        )
                        continue
                    if source == item_id:
                        errors.append(
                            f"item[{index}].merge_sources[{source_index}]={source!r}: "
                            "cannot merge a task into itself"
                        )
                        continue
                    if source not in known_ids:
                        errors.append(
                            f"item[{index}].merge_sources[{source_index}]={source!r}: "
                            "unknown source id"
                        )
                        continue
                    source_strs.append(source)
                merge_targets.append((index, item_id, source_strs))

    # Every source named in a merge_sources list must also appear in the
    # batch with status="completed" so the model's per-task coverage
    # invariant still holds (each source needs its own CompleteTask op).
    # Additionally, if a thread-bearing source feeds a non-thread
    # target, the merge would silently drop the source's comment lineage
    # at materialization time (codex P1 on #1738).  And each source can
    # be consumed by at most one merge target — feeding the same source
    # into multiple targets duplicates its lineage instead of merging it
    # (split/rebuild semantics, deferred to #1718; codex Medium on
    # #1738).
    source_to_targets: dict[str, list[str]] = {}
    for _index, target_id, sources in merge_targets:
        for source in sources:
            source_to_targets.setdefault(source, []).append(target_id)
    merge_source_set: set[str] = set(source_to_targets)
    for index, target_id, sources in merge_targets:
        target_has_thread = target_id in thread_bearing_ids
        for source in sources:
            if source not in explicitly_completed_ids:
                errors.append(
                    f"item[{index}].merge_sources={source!r}: "
                    "source must also appear in the batch with "
                    'status="completed" so the rescope reducer closes it'
                )
            if not target_has_thread and source in thread_bearing_ids:
                errors.append(
                    f"item[{index}].merge_sources={source!r}: thread source "
                    f"into non-thread target {target_id!r} would drop its "
                    "comment lineage on disk"
                )
            other_targets = [t for t in source_to_targets[source] if t != target_id]
            if other_targets:
                errors.append(
                    f"item[{index}].merge_sources={source!r}: source is also "
                    f"merged into {other_targets!r}; each source may merge "
                    "into at most one target (split/rebuild semantics belong "
                    "to a later leaf, not this one)"
                )

    # #1718 split validation.  A SplitTask op closes the source row and
    # spawns N fresh children, each inheriting the source's
    # ``lineage_comments`` and ``source_comment`` so reply paths still
    # reach the original commenter.  Reject every shape that would leave
    # the source vanished without children running:
    #
    # * null/missing id — there is no source row to decompose.
    # * non-list / empty ``split_targets`` — no children means the
    #   source closes for nothing (the empty-list sentinel ``[]`` IS a
    #   no-op like ``merge_sources=[]``, but a present-but-broken value
    #   has to fail closed).
    # * any child not a dict with a non-empty string title.
    # * combined with ``status="completed"``, a non-empty
    #   ``merge_sources``, or appearing as another item's merge source —
    #   structural ops are mutually exclusive (split closes the source
    #   itself; merging would duplicate lineage).
    # * source already completed/blocked on disk — same shape as the
    #   merge guards above (reducer skips completed snapshot tasks; the
    #   picker skips blocked tasks, so children would never run).
    for index, item in enumerate(ordered_items):
        if not isinstance(item, dict):
            continue
        if "split_targets" not in item:
            continue
        targets = item["split_targets"]
        if not isinstance(targets, list):
            errors.append(
                f"item[{index}].split_targets: must be a list, "
                f"got {type(targets).__name__}"
            )
            continue
        if not targets:
            # ``split_targets: []`` is the documented "no split" sentinel
            # (matches ``merge_sources=[]`` semantics) — accepted, no op.
            continue
        item_id = item.get("id")
        if item_id is None:
            errors.append(
                f"item[{index}].split_targets on a null/missing id: "
                "split needs an existing source task to decompose "
                "(splitting into a not-yet-created task isn't supported)"
            )
            continue
        if not isinstance(item_id, str) or not item_id or item_id not in known_ids:
            # Already reported above by the id-shape / id-existence
            # checks; skip here to avoid duplicate messages.
            continue
        if item.get("status") == str(TaskStatus.COMPLETED):
            errors.append(
                f"item[{index}].split_targets on {item_id!r}: target "
                'also marked status="completed"; structural ops are '
                "mutually exclusive (SplitTask itself closes the source)"
            )
        if item_id in currently_completed_ids:
            errors.append(
                f"item[{index}].split_targets on {item_id!r}: target "
                "is already completed on disk; SplitTask wouldn't fire "
                "(the reducer skips completed snapshot tasks) and the "
                "children would never be spawned"
            )
        if item_id in currently_blocked_ids:
            errors.append(
                f"item[{index}].split_targets on {item_id!r}: target "
                "is blocked; the SplitTask op would close it but blocked-"
                "target semantics for split aren't defined yet (#1247 "
                "territory)"
            )
        if item_id in non_splittable_source_ids:
            # Kind classification is title-prefix driven, so a split
            # whose children carry plain titles silently reclassifies
            # an ASK/DEFER source's children to executable spec tasks
            # (codex P1).  CI-failure rows are atomic and don't
            # decompose either.
            errors.append(
                f"item[{index}].split_targets on {item_id!r}: source "
                "is an ASK/DEFER/CI task; splitting these would "
                "silently reclassify children (kind is title-prefix "
                'driven).  Use status="completed" + new tasks if you '
                "really want to convert it into actionable work."
            )
        if "merge_sources" in item and item["merge_sources"] != []:
            errors.append(
                f"item[{index}].split_targets on {item_id!r}: combined "
                "with merge_sources; structural ops are mutually "
                "exclusive (would duplicate lineage)"
            )
        if item_id in merge_source_set:
            errors.append(
                f"item[{index}].split_targets on {item_id!r}: source "
                "also appears in another item's merge_sources; would "
                "duplicate lineage (folded into merge target AND "
                "inherited by children)"
            )
        for child_index, child in enumerate(targets):
            if not isinstance(child, dict):
                errors.append(
                    f"item[{index}].split_targets[{child_index}]: must "
                    f"be a dict, got {type(child).__name__}"
                )
                continue
            title = child.get("title")
            if not isinstance(title, str) or not _normalize_title(title):
                errors.append(
                    f"item[{index}].split_targets[{child_index}].title: "
                    f"must be a non-empty string, got {title!r}"
                )

    return errors


def _find_duplicate_titles(
    ordered_items: list[dict[str, Any]],
    existing_by_id: dict[str, str],
) -> list[str]:
    """Return non-empty effective titles that appear more than once.

    The dedup runs on the same effective title each item will land as on
    disk — what _effective_title returns after normalization and after
    the existing-title fallback for blank/non-string proposals.  Without
    that, ``"A\\nB"`` and ``"A B"`` slip past the nudge as distinct then
    both persist as ``"A B"``; ``{title:""}`` and ``{title:"Alpha"}`` slip
    past then both end up ``"Alpha"`` (codex on #1729).

    Each duplicated title is listed exactly once in the result, in the
    order of its first repeated occurrence.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in ordered_items:
        title = _effective_title(item, existing_by_id)
        if not title:
            continue
        if title in seen and title not in duplicates:
            duplicates.append(title)
        seen.add(title)
    return duplicates


def _apply_reorder(
    current: list[dict[str, Any]],
    ordered_items: list[dict[str, Any]],
    original_ids: frozenset[str] = frozenset(),
    intents: list[RescopeIntent] | None = None,
) -> list[dict[str, Any]]:
    """Apply Opus-synthesised items to the current task list.

    Rules (in priority order):
    - CI tasks always come first.
    - Non-CI pending tasks follow the snapped queue order for existing tasks;
      new tasks (null/absent id in Opus output) are appended after existing
      pending tasks, CI-first within the new batch.
    - Pending/in_progress tasks that Opus omits are marked completed; the caller
      detects affected in-progress tasks and signals an abort so the worker picks
      the new next task.
    - Tasks added after the original snapshot (IDs not in the snapshot) are
      appended at the end so they are never silently dropped.
    - Completed tasks are always preserved at the end in their original order.
    - Title and description are updated from Opus's output for an existing
      task id (#1713 made title mutable); thread anchor is still preserved
      (#1714 will lift that).
    - Task identity is the durable task id, not title text or thread anchor.
    - Opus-returned IDs outside the snapshot or duplicated are ignored.
    - Opus-returned items with a null or absent id are treated as new tasks
      and inserted after existing pending tasks (before completed tasks).
    """
    snapshot_ids = original_ids or frozenset(
        t["id"] for t in current if t.get("status") != TaskStatus.COMPLETED
    )
    # Pre-allocate split children's ``(id, created_at)`` tuples once
    # so the apply and verify passes mint identical metadata even
    # across a wall-clock second boundary.  Ids come from
    # ``uuid.uuid7()`` so collisions across split children, new
    # tasks, and existing on-disk rows are statistically impossible
    # without any shared bookkeeping.
    split_child_ids = _allocate_split_child_ids(ordered_items)
    oracle_result = _apply_reorder_with_oracle(
        current, ordered_items, snapshot_ids, split_child_ids
    )
    _assert_rescope_matches_oracle(
        current, ordered_items, snapshot_ids, oracle_result, split_child_ids
    )

    new_tasks = _make_new_tasks_from_opus(
        ordered_items, snapshot_ids, current=current, intents=intents
    )
    if not new_tasks:
        return oracle_result

    # Merge new tasks into oracle result: CI tasks first (in both groups),
    # then non-CI pending (oracle then new), then completed at end.
    completed_status = str(TaskStatus.COMPLETED)
    oracle_pending = [t for t in oracle_result if t.get("status") != completed_status]
    oracle_completed = [t for t in oracle_result if t.get("status") == completed_status]

    ci_new = [t for t in new_tasks if t.get("type") == "ci"]
    non_ci_new = [t for t in new_tasks if t.get("type") != "ci"]

    ci_oracle = [t for t in oracle_pending if t.get("type") == "ci"]
    non_ci_oracle = [t for t in oracle_pending if t.get("type") != "ci"]

    return ci_oracle + ci_new + non_ci_oracle + non_ci_new + oracle_completed


def _merge_source_ids(ordered_items: list[dict[str, Any]]) -> set[str]:
    """Collect the set of task ids that appear as merge-sources in this batch.

    A merged source's status flips to COMPLETED (its own CompleteTask op),
    but the original ask is still being honored — just folded into a
    larger pending task.  Merging isn't a scope change from the
    commenter's perspective, so the auto "covered by recent commits"
    notification doesn't apply.  Whether any source comment was
    *materially* affected by the consolidation is an Opus judgment call,
    handled by the reply-back filter epic (#1256) — specifically the
    material-vs-aggregation classifier (#1723) and per-intent
    notification (#1724).  This deterministic layer's job is just to
    suppress the auto-reply.
    """
    merged: set[str] = set()
    for item in ordered_items:
        sources = item.get("merge_sources")
        if not isinstance(sources, list):
            continue
        for source in sources:
            if isinstance(source, str) and source:
                merged.add(source)
    return merged


def _compute_thread_changes(
    original: list[dict[str, Any]],
    result: list[dict[str, Any]],
    original_ids: frozenset[str],
    consumed_source_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return change records for thread tasks that were completed or materially modified.

    Only tasks in *original_ids* (those Opus knew about) with a ``thread``
    attachment are reported.  Already-completed tasks are excluded.
    Tasks listed in *consumed_source_ids* are also excluded — their
    completion is bookkeeping for the rescope reducer's per-task
    coverage invariant, not "Fido finished your work."  This covers
    both ``MergeTasks`` source rows (folded into a still-pending
    target) and ``SplitTask`` source rows (work moved into the
    children); in both cases the inheriting row(s) will fire their own
    change records when they eventually complete (#1717, #1718).

    Each record is one of:
    - ``{"task": ..., "kind": "completed"}`` — Opus omitted or marked it done
    - ``{"task": ..., "kind": "modified", "new_title": ..., "new_description": ...}``
      — the **title** changed.  Title is the contract Fido committed to in
      its initial reply; a title change indicates the scope materially
      shifted.  Pure description rewrites that preserve the title are
      treated as internal rephrasing and produce no change record — the
      reviewer's mental model is "I gave feedback, Fido replied, I'll see
      them again when the work lands," and rescope-internal description
      edits between those two events are noise (#1388).

    Note: the Rocq model (``TaskCompleted`` vs ``TaskCancelled``) distinguishes
    explicit completion from omission.  Here both map to ``"completed"`` because
    ``_apply_reorder`` normalises omitted tasks into completed rows — the
    ``r is None`` case cannot fire through the current ``reorder_tasks`` path.
    A future change could propagate the distinction (e.g. via a marker field on
    the task dict set by ``_apply_reorder``) so the reply body distinguishes
    "done" from "cancelled".
    """
    consumed_sources = consumed_source_ids or set()
    result_by_id = {t["id"]: t for t in result}
    changes: list[dict[str, Any]] = []
    for t in original:
        if t["id"] not in original_ids:
            continue
        if t["id"] in consumed_sources:
            continue
        if not t.get("thread"):
            continue
        if t.get("status") == TaskStatus.COMPLETED:
            continue
        tid = t["id"]
        r = result_by_id.get(tid)
        # #1722: surface the post-rescope task's contributing_intents
        # on every change record so the future material-vs-aggregation
        # classifier (#1723) and per-intent notifier (#1724) can route
        # replies to the right commenters.  Empty list = no intents
        # attributed.
        contributing = list((r or {}).get("contributing_intents") or [])
        if r is None or r.get("status") == TaskStatus.COMPLETED:
            changes.append(
                {
                    "task": t,
                    "kind": "completed",
                    "contributing_intents": contributing,
                }
            )
        elif r.get("title") != t.get("title"):
            changes.append(
                {
                    "task": t,
                    "kind": "modified",
                    "new_title": r.get("title", ""),
                    "new_description": r.get("description", ""),
                    "contributing_intents": contributing,
                }
            )
    return changes


def reorder_tasks(
    tasks: "Tasks",
    commit_summary: str,
    *,
    intents: list[RescopeIntent] | None = None,
    agent: ProviderAgent | None = None,
    prompts: Prompts | None = None,
    issue: ActiveIssue | None = None,
    pr: ActivePR | None = None,
    prior_attempts: list[ClosedPR] | None = None,
    _on_changes: Callable[[list[dict[str, Any]]], None] | None = None,
    _on_inprogress_affected: Callable[[str], None] | None = None,
    _on_rescope_apply: Callable[
        [
            list[dict[str, Any]],
            list[rescope_oracle.OpInput],
            dict[str, int],
            list[IntentVerdict],
        ],
        None,
    ]
    | None = None,
    _on_done: Callable[[], None] | None = None,
) -> None:
    """Reorder pending tasks by Opus dependency analysis.

    Reads the task list, asks Opus to reorder/rewrite/drop tasks based on
    dependency analysis and recent commits, then atomically writes the result.

    The task list is read twice: once before the Opus call (to build the
    prompt) and once inside the write-lock (to pick up any tasks added while
    Opus was thinking).  Tasks added in that window are preserved at the end
    of the list rather than silently dropped.

    CI tasks always stay first; completed tasks are always preserved.
    An empty or unparseable Opus response leaves the task list unchanged.

    If *_on_changes* is provided and any thread tasks were completed or modified,
    it is called with a list of change records (see :func:`_compute_thread_changes`).

    If *_on_inprogress_affected* is provided and the currently in-progress task
    is marked completed or modified by Opus, it is called with the affected
    task's id so the caller can abort the running worker (targeted at that
    task) and restart on the new next task.  When the in-progress task is
    modified its status is reset to ``pending`` so the worker loop picks it
    up again with the updated description.

    If *_on_done* is provided, it is called after a successful reorder write so
    callers can trigger follow-up work (e.g. rewriting the PR description).
    """
    task_list = tasks.list()
    if not task_list:
        log.info("reorder_tasks: no tasks — skipping")
        return

    if agent is None:
        agent = ClaudeClient()
    if prompts is None:
        prompts = Prompts("")

    original_ids = frozenset(t["id"] for t in task_list)
    # #1812 (INV-D wiring): when intents are present, drive Opus with
    # the verdict-shaped envelope (#1810) and parse via
    # ``_parse_rescope_verdicts`` (#1809).  The no-intent path
    # (``rescope_before_pick``) keeps the legacy ops-shape prompt —
    # there's nothing to attribute and the verdict envelope would be
    # empty noise.
    use_verdicts = bool(intents)
    if use_verdicts:
        prompt = prompts.rescope_prompt_verdicts(
            task_list,
            commit_summary,
            intents=intents or [],
            issue=issue,
            pr=pr,
            prior_attempts=prior_attempts,
        )
    else:
        prompt = prompts.rescope_prompt(
            task_list,
            commit_summary,
            issue=issue,
            pr=pr,
            prior_attempts=prior_attempts,
            intents=intents,
        )
    raw = agent.run_turn(
        prompt,
        model=agent.voice_model,
        allowed_tools=READ_ONLY_ALLOWED_TOOLS,
    )
    if not raw:
        log.warning("reorder_tasks: Opus returned empty response — skipping")
        return

    # Parse + cross-op-validate; on errors, nudge Opus with the FULL
    # complaint list (not just the first defect) so it can fix
    # everything at once (#1719).  Budget shared with the duplicate-
    # title nudge below.
    snapshot_ids = frozenset(
        t["id"]
        for t in task_list
        if t.get("status") != TaskStatus.COMPLETED and "id" in t
    )
    operations: list[_RescopeOp] = []
    parse_errors: list[str] = ["initial parse"]
    # Hoisted across nudge iterations so the final successful parse is
    # available to the post-apply ``_on_intent_dispositions`` callback —
    # INV-F (#1804) needs the verdict list to drive the extracted
    # reply-back precedence decision (``oracle.reply_back_intents``).
    verdicts: list[IntentVerdict] = []
    for nudge_attempt in range(_RESCOPE_MAX_NUDGES + 1):
        if use_verdicts:
            verdicts, verdict_errors = _parse_rescope_verdicts(raw, intents or [])
            if verdict_errors:
                operations, parse_errors = [], verdict_errors
            else:
                # Flatten the per-verdict ops into the flat op list
                # the existing ``_apply_reorder`` consumes.  Each op
                # inherits its verdict's ``intent_comment_id`` via the
                # ``contributing_intents`` field (#1722 attribution
                # convention).  Re-parsing through
                # ``_parse_rescope_operations`` reuses op-level shape
                # validation rather than duplicating it.
                flat_ops = _flatten_verdicts_to_ops(verdicts)
                synthetic = json.dumps({"operations": flat_ops})
                operations, parse_errors = _parse_rescope_operations(synthetic)
        else:
            operations, parse_errors = _parse_rescope_operations(raw)
        if not parse_errors:
            cross_errors = _find_cross_op_errors(operations, snapshot_ids)
            if not cross_errors:
                break
            parse_errors = cross_errors
        attempts_remaining = _RESCOPE_MAX_NUDGES - nudge_attempt
        if attempts_remaining <= 0:
            log.warning(
                "reorder_tasks: rescope batch dropped after %d parse-error "
                "nudges; final errors: %s",
                _RESCOPE_MAX_NUDGES,
                "; ".join(parse_errors),
            )
            return
        log.warning(
            "reorder_tasks: Opus rescope response had %d problem(s) — "
            "nudging (attempt %d/%d, %d remaining)",
            len(parse_errors),
            nudge_attempt + 1,
            _RESCOPE_MAX_NUDGES,
            attempts_remaining - 1,
        )
        nudge = (
            prompts.rescope_verdicts_parse_nudge(
                parse_errors, attempts_remaining=attempts_remaining - 1
            )
            if use_verdicts
            else prompts.rescope_parse_nudge(
                parse_errors, attempts_remaining=attempts_remaining - 1
            )
        )
        nudge_raw = agent.run_turn(
            nudge,
            model=agent.voice_model,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
        )
        if not nudge_raw:
            log.warning(
                "reorder_tasks: empty response after parse-error nudge — dropping batch"
            )
            return
        raw = nudge_raw
    else:  # pragma: no cover — loop body always returns or breaks
        return

    ordered_items = _operations_to_items(operations)

    # Snapshot existing titles for the dedup check.  The check runs on
    # *effective* titles (post-normalization, post-existing-title fallback)
    # so what the nudge accepts as unique matches what the apply path will
    # land on disk.  This snapshot can be stale by apply time if a new task
    # arrives concurrently, but dedup here is best-effort anyway — apply
    # re-derives the effective titles under flock.
    pre_nudge_existing = _existing_titles_by_id(tasks.list())

    # Nudge Opus up to _RESCOPE_MAX_NUDGES times if it proposed duplicate
    # titles.  Each turn runs in the same conversation so the model sees its
    # prior responses and the remaining-attempt count in each nudge.  If
    # duplicates remain after all nudges, the proposed titles are still
    # applied (#1713 made title mutable).  Uniqueness during rewrites is
    # best-effort via the nudge — the durable invariant is on enqueueing
    # new tasks (find_pending_title_duplicate in the oracle), not on edits.
    #
    # #1812: verdict-shape mode skips this nudge entirely — its nudge
    # text asks Opus to "resubmit the full operations array," which
    # would shift Opus mid-conversation from the verdict envelope
    # back to ops.  The existing best-effort framing covers this: if
    # duplicates survive, the apply path tolerates them.  A future
    # leaf can add a verdict-shape duplicate nudge if practice shows
    # it matters.
    for nudge_attempt in range(0 if use_verdicts else _RESCOPE_MAX_NUDGES):
        duplicates = _find_duplicate_titles(ordered_items, pre_nudge_existing)
        if not duplicates:
            break
        attempts_remaining = _RESCOPE_MAX_NUDGES - nudge_attempt - 1
        log.warning(
            "reorder_tasks: Opus proposed duplicate titles %r — nudging "
            "(attempt %d/%d, %d remaining after this)",
            duplicates,
            nudge_attempt + 1,
            _RESCOPE_MAX_NUDGES,
            attempts_remaining,
        )
        nudge = prompts.rescope_duplicate_nudge(
            duplicates, attempts_remaining=attempts_remaining
        )
        nudge_raw = agent.run_turn(
            nudge,
            model=agent.voice_model,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
        )
        if not nudge_raw:
            log.warning(
                "reorder_tasks: empty response after duplicate nudge — "
                "proceeding with fallback"
            )
            break
        nudge_ops, nudge_errors = _parse_rescope_operations(nudge_raw)
        if nudge_errors:
            log.warning(
                "reorder_tasks: unparseable response after duplicate nudge — "
                "proceeding with fallback (errors: %s)",
                "; ".join(nudge_errors),
            )
            break
        ordered_items = _operations_to_items(nudge_ops)

    # Route the write through Tasks's public modify() — its on_mutate
    # hook (e.g. the SCADA snapshot publisher) fires automatically on
    # exit while the flock is still held, so concurrent writers
    # serialize on the same lock.
    inprogress_affected = False
    pre_rescope: list[dict[str, Any]] = []
    result: list[dict[str, Any]] = []
    inprogress: dict[str, Any] | None = None
    rejected = False
    with tasks.modify() as current:
        # #1715: validate the batch atomically before applying any of it.
        # If validation fails, every error is logged and the with block
        # exits without slice-assigning — JsonFileStore.modify writes
        # back the same content, so the durable list is unchanged.
        # Partial commits aren't safe: they'd leave tasks.json in a
        # state Opus didn't propose, breaking snapshot/replay reasoning.
        validation_errors = _validate_rescope_batch(current, ordered_items)
        if validation_errors:
            rejected = True
            for error in validation_errors:
                log.warning("reorder_tasks: rejecting batch — %s", error)
        else:
            # Snapshot the pre-rescope list so _compute_thread_changes can
            # diff against it after the in-place slice-assign below (codex
            # P2 #1696: comparing post-rescope to itself suppresses
            # _on_changes notifications for thread tasks Opus completed/
            # modified).
            pre_rescope = [dict(t) for t in current]
            inprogress = next(
                (t for t in current if t.get("status") == TaskStatus.IN_PROGRESS), None
            )
            result = _apply_reorder(
                current, ordered_items, original_ids, intents=intents
            )
            if inprogress is not None:
                # The omission ⇒ completed branch is gone (#1357 case A): the
                # rescope reducer now uses KeepTask for omitted snapshot
                # tasks, so the in-progress task always survives the rescope
                # at its current status.  Triggers for _on_inprogress_affected
                # are explicit modifications by Opus that change the task's
                # contract: title, description, source-comment anchor (#1714),
                # or — under #1716 — explicit completion.
                inprogress_in_result = next(
                    (t for t in result if t["id"] == inprogress["id"]), None
                )
                # Compare anchors via _task_source_comment_for_oracle so a
                # legacy ``'42'`` string in tasks.json compares equal to the
                # oracle-materialized ``42`` (codex on #1731): int(comment_id)
                # normalization on both sides means spurious type-mismatch
                # aborts can't fire.
                if inprogress_in_result is not None:
                    completed_by_rescope = inprogress_in_result.get("status") == str(
                        TaskStatus.COMPLETED
                    )
                    # Anchor and lineage are both compared via the canonical
                    # oracle helpers so int/string drift can't trigger a
                    # spurious abort (codex on #1731).  Lineage drift covers
                    # the merge case (codex on #1738): a MergeTasks op only
                    # adds entries to thread.lineage_comment_ids and leaves
                    # title/desc/anchor/status untouched, so without this
                    # check a worker turn that started with the OLD lineage
                    # would never see the merged-in source comments.
                    text_or_anchor_changed = (
                        inprogress_in_result.get("title") != inprogress.get("title")
                        or inprogress_in_result.get("description")
                        != inprogress.get("description")
                        or _task_source_comment_for_oracle(inprogress_in_result)
                        != _task_source_comment_for_oracle(inprogress)
                        or _thread_lineage_comment_ids(
                            inprogress_in_result.get("thread")
                        )
                        != _thread_lineage_comment_ids(inprogress.get("thread"))
                    )
                    if completed_by_rescope or text_or_anchor_changed:
                        inprogress_affected = True
                        if completed_by_rescope:
                            # The in-progress row's status flipped to completed
                            # via #1716 (explicit completion), #1717 (merged
                            # into a different target), or #1718 (split into
                            # children) — each closes the row but represents
                            # a different decision.  In every case the worker
                            # just needs to abort its now-stale turn so the
                            # picker can advance to the new next task; no
                            # reset to pending.
                            log.info(
                                "reorder_tasks: in-progress task closed by "
                                "rescope (completed/merged/split) — aborting "
                                "turn: %s",
                                inprogress_in_result.get("title", "")[:60],
                            )
                        else:
                            # Text/anchor/lineage change: reset to pending so
                            # the worker re-picks under the new scope.
                            inprogress_in_result["status"] = str(TaskStatus.PENDING)
                            log.info(
                                "reorder_tasks: in-progress task modified by "
                                "Opus — reset to pending: %s",
                                inprogress_in_result.get("title", "")[:60],
                            )
            # Slice-assign so JsonFileStore.modify writes the recomputed
            # list back (it persists the same dict/list object it yielded).
            current[:] = result

    if rejected:
        # _on_done's contract is "post-successful reorder" — production wires
        # it to sync_tasks (git push) and _rewrite_pr_description (GitHub API
        # write).  Neither makes sense after a rejection: nothing committed,
        # nothing to sync.  Skip every post-commit callback (codex on #1733).
        return

    if _on_changes is not None:
        # Always call with the (possibly empty) list — the production
        # callback iterates and a no-change run is a no-op.  Avoids a
        # branch that's hard to reach in tests now that title
        # preservation by the reducer means rescope rarely produces a
        # material change record (#1388).
        _on_changes(
            _compute_thread_changes(
                pre_rescope,
                result,
                original_ids,
                consumed_source_ids=_merge_source_ids(ordered_items)
                | _split_source_ids(ordered_items),
            )
        )

    if _on_rescope_apply is not None and intents is not None:
        # INV-F (#1804): hand the post-apply state to the notifier so it
        # can ask the oracle which intents need a reply.  We hand the
        # adapter (a) the raw result task list, (b) the ``OpInput`` list
        # (typed as the oracle expects) the model needs to apply its
        # reply-back rule, (c) the task-string-id → positive map the
        # adapter uses to translate oracle output back to result-task
        # ids, and (d) the parsed verdicts (reserved; not consumed by
        # the current rule, kept for forward compatibility).
        ids_by_task_id, _, _, _ = _rescope_state_for_oracle(pre_rescope)
        _on_rescope_apply(
            result,
            _build_op_inputs(operations, ids_by_task_id),
            ids_by_task_id,
            verdicts,
        )

    if inprogress_affected and _on_inprogress_affected is not None:
        assert inprogress is not None  # inprogress_affected is True
        _on_inprogress_affected(str(inprogress["id"]))

    log.info("reorder_tasks: applied reorder — %d tasks", len(result))

    if _on_done is not None:
        _on_done()


def sync_tasks_background(
    work_dir: Path,
    gh: GitHub,
    *,
    _start: Callable[[threading.Thread], None] = threading.Thread.start,
) -> None:
    """Launch :func:`sync_tasks` in a daemon background thread."""
    t = threading.Thread(
        target=sync_tasks,
        args=(work_dir, gh),
        name=f"sync-{work_dir.name}",
        daemon=True,
    )
    _start(t)


def _build_task_list_snapshot(task_list: list[dict[str, Any]]) -> "TaskListSnapshot":
    """Project a tasks.json list into a :class:`TaskListSnapshot` leaf.

    Pure data transform — used by :meth:`Tasks.on_mutate` to publish
    after every successful tasks.json write.  Counts, current-task
    selection, and task position follow the same semantics the legacy
    flat ``_collect_fido_state`` used: in-progress task wins for
    ``current_task``; otherwise the first pending entry; ``task_total``
    counts non-completed entries.
    """
    from fido.appstate import TaskListSnapshot

    pending = sum(1 for t in task_list if t.get("status") == "pending")
    completed = sum(1 for t in task_list if t.get("status") == "completed")

    current_task = ""
    for t in task_list:
        if t.get("status") == "in_progress":
            current_task = t.get("title", "")
            break
    if not current_task:
        for t in task_list:
            if t.get("status") == "pending":
                current_task = t.get("title", "")
                break

    non_completed = [t for t in task_list if t.get("status") != "completed"]
    task_number = 0
    task_total = 0
    if non_completed:
        task_total = len(non_completed)
        task_number = 1
        for idx, t in enumerate(non_completed, start=1):
            if t.get("status") == "in_progress":
                task_number = idx
                break

    return TaskListSnapshot(
        pending_task_count=pending,
        completed_task_count=completed,
        current_task=current_task,
        task_number=task_number,
        task_total=task_total,
    )


class Tasks(JsonFileStore):
    """Encapsulates task file operations for a single worker directory.

    Abstracts all file access so callers never touch the filesystem directly.
    Instantiate with the work_dir path and inject wherever tasks are needed.

    Inherits :meth:`~JsonFileStore.modify` for atomic read-modify-write of
    the entire task list.

    *state_updater* and *repo_name* (when supplied) wire :meth:`on_mutate`
    to publish a fresh :class:`~fido.appstate.TaskListSnapshot` to the
    repo's leaf field on :class:`~fido.appstate.RepoState` after every
    successful write.  Independent leaf — task mutations never touch the
    issue/PR leaf or the worker/provider leaves, so there's no
    cross-source publish lock and no flock-order inversion (#1696).
    Tests that don't care about the snapshot leave them at the
    defaults; the hook becomes a no-op in that case.
    """

    def __init__(
        self,
        work_dir: Path,
        *,
        state_updater: "AtomicUpdater[FidoState] | None" = None,
        repo_name: str = "",
        fido_dir: Path | None = None,
    ) -> None:
        self._work_dir = work_dir
        self._state_updater = state_updater
        self._repo_name = repo_name
        # *fido_dir* is the canonical resolved fido directory (under
        # the registry-resolved git_dir, #1696).  When omitted (test
        # scaffolding without a registry) fall back to the
        # conventional layout.
        self._fido_dir = (
            fido_dir if fido_dir is not None else work_dir / ".git" / "fido"
        )

    @property
    def _data_path(self) -> Path:
        return self._fido_dir / "tasks.json"

    def _default(self) -> list[dict[str, Any]]:
        return []

    def _validate(self, data: object) -> None:
        """Ensure every task has a ``type`` field."""
        assert isinstance(data, list), "tasks.json must hold a list"
        for t in data:
            assert isinstance(t, dict), "task entries must be JSON objects"
            if "type" not in t:
                raise ValueError(f"task {t.get('id', '?')} missing required type field")

    def on_mutate(self, data: object) -> None:
        """Publish the new :class:`~fido.appstate.TaskListSnapshot` leaf
        to :class:`~fido.appstate.FidoState` after each task-list write.

        Fires *after* :meth:`~JsonFileStore.modify` releases the
        tasks.json flock so the lens-write CAS retry doesn't run while
        we hold any flock.  Pure leaf publish — no cross-source reads,
        no shared lock with other publishers; sibling mutators
        (State for the issue/PR leaf, registry for activity/thread/
        provider leaves) update independent fields on RepoState.

        No-op when *state_updater* or *repo_name* were not supplied at
        construction (tests, one-off CLI use)."""
        if self._state_updater is None or not self._repo_name:
            return
        assert isinstance(data, list), "tasks.json must hold a list"
        snapshot = _build_task_list_snapshot(data)
        _name = self._repo_name
        self._state_updater.update(lambda root: root.repos[_name].task_list, snapshot)

    def list(self) -> list[dict[str, Any]]:
        """Return all tasks."""
        with _locked(self._data_path) as lock:
            return lock.read()

    def add(
        self,
        title: str,
        task_type: TaskType,
        description: str = "",
        status: TaskStatus = TaskStatus.PENDING,
        thread: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a task. Returns the new (or existing duplicate) task."""
        if not isinstance(task_type, TaskType):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"task_type must be TaskType, got {type(task_type).__name__}"
            )
        title = _normalize_title(title)
        task: dict[str, Any] = {
            "id": str(uuid.uuid7()),
            "title": title,
            "type": str(task_type),
            "description": description,
            "status": str(status),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if thread:
            task["thread"] = thread
        comment_id = (thread or {}).get("comment_id")
        with self.modify() as existing:
            for t in existing:
                existing_thread = t.get("thread") or {}
                existing_comment_id = existing_thread.get("comment_id")

                # Exact-comment-id replay protection: the only dedup at the
                # add boundary (#1665).  Lineage stops being a join key —
                # three distinct comments on the same PR thread produce
                # three distinct tasks.  Lineage stays on the task as
                # origin metadata via :func:`_merge_thread_lineage`,
                # answering "which comments contributed to this task,"
                # not "is this task the same as that one."  Combine /
                # split / rewrite decisions move to the rescope reducer
                # under #1340 (#1666 / #1667).
                if comment_id is not None and existing_comment_id == comment_id:
                    if _merge_thread_lineage(existing_thread, thread or {}):
                        t["thread"] = existing_thread
                    log.info(
                        "task already exists for comment_id %s (status: %s)",
                        comment_id,
                        t["status"],
                    )
                    return t

                # Title-match dedup for non-comment-derived tasks (CI
                # failures, spec setup): a pending task with the same
                # title is the same work; don't double-add.  Comment-
                # derived adds skip this — exact comment_id is the only
                # dedup for them (#1665).
                if (
                    comment_id is None
                    and t["status"] == TaskStatus.PENDING
                    and t["title"] == title
                ):
                    log.info("task already exists: %s", title[:80])
                    return t
            existing.append(task)
        log.info("task added: %s", title[:80])
        return task

    def complete_by_id(self, task_id: str) -> dict[str, Any] | None:
        """Mark a task completed. Returns its thread dict or None.

        Returns ``None`` silently if no matching task is found.
        """
        with self.modify() as tasks:
            for t in tasks:
                if t["id"] == task_id and t["status"] != TaskStatus.COMPLETED:
                    t["status"] = str(TaskStatus.COMPLETED)
                    log.info("task completed (id=%s): %s", task_id, t["title"][:80])
                    return t.get("thread")
        return None

    def complete_with_resolve(
        self,
        task_id: str,
        gh: GitHub,
        *,
        collaborators: frozenset[str] = frozenset(),
        allowed_bots: frozenset[str] = frozenset(),
    ) -> None:
        """Mark a task completed and resolve its review thread if we posted last.

        Combines :meth:`complete_by_id` with the per-task thread-resolve logic
        so both the worker and CLI share one path.  If the task has no thread
        metadata, or we are not the last commenter, the resolve step is skipped
        silently.

        Always triggers a background :func:`sync_tasks` after the completion
        so the PR body checkbox flips even when the worker loop doesn't run
        another sync between this completion and the PR-ready/merge step
        (#988).
        """
        thread = self.complete_by_id(task_id)
        sync_tasks_background(self._work_dir, gh)
        if not thread:
            return
        repo = thread.get("repo", "")
        pr = thread.get("pr")
        comment_id = thread.get("comment_id")
        if not (repo and pr and comment_id):
            return
        try:
            us = gh.get_user()
            owner, repo_name = repo.split("/", 1)
            threads = gh.get_review_threads(owner, repo_name, pr)
            pending_tasks = thread_tasks_for_auto_resolve_oracle(self.list())
            for t in threads:
                if _review_thread_contains_comment(t, int(comment_id)):
                    decision = thread_resolve_oracle.resolution_decision(
                        review_thread_for_auto_resolve_oracle(
                            t,
                            us,
                            owner=owner,
                            collaborators=collaborators,
                            allowed_bots=allowed_bots,
                        ),
                        pending_tasks,
                    )
                    if not isinstance(
                        decision, thread_resolve_oracle.ResolveReviewThread
                    ):
                        log.info("thread has pending same-thread work — not resolving")
                        return
                    gh.resolve_thread(t["id"])
                    log.info("thread resolved: %s", t["id"])
                    return
        except Exception as exc:  # noqa: BLE001
            log.warning("thread resolution skipped: %s", exc)

    def has_pending_for_comment(self, comment_id: int | str) -> bool:
        """Return True if any pending task references *comment_id*."""
        cid = int(comment_id)
        with _locked(self._data_path) as lock:
            for t in lock.read():
                if t.get("status") == TaskStatus.PENDING:
                    if int((t.get("thread") or {}).get("comment_id", -1)) == cid:
                        return True
        return False

    def reset_to_pending(self, task_id: str) -> bool:
        """Reset a task's status to ``pending``. Returns True if found.

        Used by the worker's abort cleanup so that an aborted in-progress
        task survives in the queue and can be re-picked under whatever
        scope the rescope cascade gave it (#1357 case B).
        """
        with self.modify() as tasks:
            for t in tasks:
                if t["id"] == task_id:
                    if t.get("status") != str(TaskStatus.PENDING):
                        t["status"] = str(TaskStatus.PENDING)
                    return True
            return False

    def remove(self, task_id: str) -> bool:
        """Remove a task. Returns True if found."""
        with self.modify() as tasks:
            new_tasks = [t for t in tasks if t["id"] != task_id]
            if len(new_tasks) < len(tasks):
                tasks[:] = new_tasks
                return True
        return False

    def update(self, task_id: str, status: TaskStatus) -> bool:
        """Update a task's status. Returns True if found."""
        with self.modify() as tasks:
            for t in tasks:
                if t["id"] == task_id:
                    t["status"] = str(status)
                    log.info("task %s → %s", task_id, status)
                    return True
        return False

    def unblock_tasks(self) -> int:
        """Transition all BLOCKED tasks back to PENDING.

        Called when a new PR comment arrives so the worker can re-evaluate
        whether it is still blocked.  Returns the number of tasks unblocked.
        """
        count = 0
        with self.modify() as task_list:
            for t in task_list:
                if t.get("status") == TaskStatus.BLOCKED:
                    t["status"] = str(TaskStatus.PENDING)
                    count += 1
        if count:
            log.info("unblocked %d task(s)", count)
        return count
