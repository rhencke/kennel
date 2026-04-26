# Bug-Mined Coordination Invariants

**Source:** 23 closed `Bug:` issues + 8 open ones from
[FidoCanCode/home](https://github.com/FidoCanCode/home),
2026-04-08 → 2026-04-26.
**Filed as:** [#1040](https://github.com/FidoCanCode/home/issues/1040)

Each cluster maps a class of bugs to one invariant, the subsystem that
should enforce it, the proposed Rocq model, and the existing D-series
issue (if any) that should absorb the work.

---

## A. Session ownership & FIFO handoff

**Invariant.** At most one of {worker, handler} holds the claude session at
any moment. When the holder releases, ownership goes to the
oldest-queued handler if any; only when the queue is empty does the worker
get it. Worker→worker, handler→handler, and handler→worker transitions
must not pass through an unprotected `Free` window where another caller
can race in.

**Bugs.** #955, #983, #1017 (and #984's pre-FSM history).

**Status.** Modeled in `session_lock.v`. Live in production (#1031, validated
2026-04-26). FSM transition logging on every event for audit.

**Open.** Trim transition logs to debug or abnormal-only (#1037).

---

## B. Claude protocol stream — single-writer + cancel scoping

**Invariant.** Exactly one thread writes to claude's stdin at a time —
including the cancel-control byte. The cancel flag is per-turn: cleared
on every `prompt()` entry, never observed across turn boundaries. A
preempt-cancelled drain leaves the session in a well-defined post-cancel
state from which the next caller can immediately enter a fresh turn
without a stale `_in_turn` flag, stale cancel, or stale empty result
masquerading as success.

**Bugs.** #973, #975, #979, #1032 (false alarm but same neighborhood).

**Status.** Hand-fixed by lock discipline + cancel-clear in `prompt()`.
Not modeled. Highest-value next target — 4 bugs, all subtle, all
mid-session-protocol.

**Proposed model.** `claude_session.v` — states `Idle | Sending |
AwaitingReply | Draining | Cancelled`; events `Send | ReplyChunk |
ReplyEnd | CancelFire | DrainObserve | TurnReturn`. Theorems:
`single_writer`, `cancel_does_not_persist_across_turns`,
`empty_result_is_not_completion`.

**New issue filed:** [#1041](https://github.com/FidoCanCode/home/issues/1041)
— Rocq→Python — claude protocol stream FSM (claude_session.v).

---

## C. Talker-kind / preempt-detection coherence

**Invariant.** The `(holder_thread, holder_kind)` pair associated with the
session is updated atomically on every acquire/release. There is no
window where holder_thread = worker but holder_kind = webhook (or
vice versa).

**Bugs.** #981 (the canonical instance — worker thread registered with
kind=webhook, breaks subsequent preempt detection).

**Status.** ✓ Audited 2026-04-26 — structurally impossible. Both
`ClaudeSession.__enter__` and `CopilotCLISession.__enter__` read
`kind = provider.current_thread_kind()` exactly once, then use that same
value to (a) choose the FSM acquisition path (`_fsm_acquire_worker` →
`OwnedByWorker`, else `_fsm_acquire_handler` → `OwnedByHandler`) and
(b) construct the `SessionTalker(kind=kind, ...)` passed to
`register_talker`. No separate `_pending_talker_kind` field exists; the
FSM state and the talker registry kind are always derived from the same
variable in the same call. The bug #981 class cannot recur.

**Residual note.** `preempt_worker()` reads from `get_talker().kind`
rather than from `_fsm_state` — a minor redundancy (both agree by
construction), but the FSM state alone could answer "is the holder a
worker?" via `isinstance(_fsm_state, OwnedByWorker)`. Not a bug; a
smell to address if the talker registry is ever removed.

---

## D. Task lifecycle — status FSM

**Invariant.** Task `status` transitions form a forward-only chain:
`PENDING → IN_PROGRESS → COMPLETED`. Backward transitions
(`COMPLETED → PENDING`, `IN_PROGRESS → PENDING`) are forbidden — they
have caused PR-body/tasks.json divergence and silent task loss.
`_pick_next_task` and the promote-to-ready predicate must agree on the
set of "blocking" statuses (today: `PENDING ∪ IN_PROGRESS`).

**Bugs.** #999 (picker/promote-gate disagreement → deadlock), #1013
(DEFER reverted task to PENDING after completion → PR body diverges).

**Proposed model.** Already filed as **D3 (#741) — task queue + rescope**,
plus we should formalize the status FSM specifically as part of the same
model. Theorems: `no_status_regression`, `picker_promote_agreement`,
`one_active_task_per_worker`.

---

## E. PR-body ↔ tasks.json equality

**Invariant.** After every `task_complete`, `task_add`, or
`rescope_tasks`, the PR body's work-queue section is re-synced to
`tasks.json` before the next durable state change. There is no
intermediate state visible to GitHub readers where the two disagree.

**Bugs.** #988 (final completion never re-synced), #1013 (DEFER bumped
PR body without going through sync_tasks).

**Status.** Already filed as **D10 (#748) — model PR-body↔tasks.json
invariant**. These bugs are the empirical motivation; cite them in the
issue body.

---

## F. Reply / claim dedup — exactly-once per anchor

**Invariant.** For every `(repo, anchor_comment_id)` pair, fido posts
**at most one reply** and **files at most one tracking issue** across
the lifetime of the PR. The dedup predicate uses the same identity at
post time as it does at check time (no `in_reply_to_id` strip masking
the predicate). Marker-based recovery validates author = `FidoCanCode`
before acking a promise.

**Bugs.** #266 (loop), #1004 (concurrent dedup drops legit replies),
#1014 (double-filed DEFER issue), #1015 (#1005's dedup is dead code),
#962 (open — recovery doesn't verify author), #953 (open —
substance-already-covered should also dedup).

**Status.** Partially modeled in **D1 (#739) — `_replied_comments` claim
set** and **D2 (#740) — reply-promise lifecycle**. The model needs to
explicitly cover: `(a)` post-side and check-side use the same identity,
`(b)` author validation in marker recovery, `(c)` substance-equivalence
dedup vs. queued thread tasks.

---

## G. Picker eligibility & fresh-retry

**Invariant.** The picker selects an issue iff
`(open ∧ assigned_to_FidoCanCode ∧ in_picker_descent_path)`. Closed-PR
fresh-retry re-attempts the previous issue iff it is **still assigned**
and **still in the descent path** at the moment of retry — not based on
a cached pre-close snapshot. When the current PR is blocked on a known-
bug failing CI and a fix-issue exists in the queue, the picker prefers
the fix-issue over napping.

**Bugs.** #523 (naps when fix-issue queued), #960 (re-picks unassigned
issue from cached state).

**Status.** Already filed as **D16 (#888) — issue tree cache + picker
eligibility**.

---

## H. Webhook ingress dedup & ordering

**Invariant.** For every `(event_id, comment_id)` pair, dispatch fires
exactly once. Events that GitHub naturally double-fires (review +
review_comment for the same inline comment) collapse to a single
handler invocation. Webhook handlers do **not** perform implementation
work inline — they queue a `thread`-type task and exit.

**Bugs.** #266, #1014, #950 (open — webhook does inline edits), #1007
(open — webhook starves worker for 5+ min doing implementation).

**Status.** Already filed as **D9 (#747) — webhook ingress dedupe +
ordering** for the (event_id, comment_id) part. The "handlers don't do
implementation work" rule is a separate invariant.

**New issue filed:** [#1042](https://github.com/FidoCanCode/home/issues/1042)
— D9b: webhook handlers must not perform implementation work inline.

---

## I. Issue-cache bootstrap + worker wake

**Invariant.** Workers' idle-wait is interruptible by issue-cache state
changes. When `bootstrap_issue_caches` completes, every worker that
returned "no eligible issue" against the empty cache is woken to re-scan.
The wake-vs-wait race is closed: a wake delivered between the picker's
"no result" return and the worker's `_wake.wait()` is observed.

**Bugs.** #995 (60s post-restart pickup gap, dominant restart cost).

**Status.** Already filed as **D16 (#888)** — same model as G. Should be
explicit theorem in that model.

---

## J. Rescope — title & anchor preservation

**Invariant.** `reorder_tasks` may reorder, mark-completed, or insert
tasks. It may **not** mutate any existing task's `title` or
`anchor_comment_id`. The rescope output is a permutation of the input
plus optional new entries plus optional terminal status flips —
nothing else.

**Bugs.** #1024 (rescope rewrote thread-task titles, collapsing
unrelated tasks).

**Status.** Already filed as **D11 (#749) — model rescope confluence**.
This bug is the empirical anchor.

---

## K. PR transition gate (draft→ready, request_review, merge)

**Invariant.** PR state-changing actions (`pr_ready`, `add_pr_reviewers`,
`pr_merge`) are gated on
`(all_tasks_completed ∧ ci_passing ∧ no_blocking_reviews)`. The same
predicate is used in every gate site — there is no per-site copy that
can drift.

**Bugs.** #988 (ready while pending), #1012 (re-request while pending).

**Status.** Already filed as **D18 (#890) — PR review + merge gate
lifecycle**. Theorem: `single_gate_predicate_used_at_every_site`.

---

## L. Auto-resolve review threads

**Invariant.** When a thread-type task completes successfully (code
change committed and pushed), the corresponding review thread is
resolved via `resolveReviewThread` — not just replied to.

**Bugs.** #961 (open — fido replies "done" but doesn't resolve thread).

**Status.** Likely a sub-invariant of **D13 (#751) — model thread
auto-resolve with race coverage**.

---

## M. Status display correctness — out of scope for Rocq

**Invariant.** Display layer; not coordination. Counters render
monotonically; numerator advances as tasks complete.

**Bugs.** #996 (open — task numerator stuck at 1/N), #1026 (counters
don't render after #1018 partial ship).

**Status.** Pure UI; doesn't belong in the Rocq epic. Standard bug fix.

---

## N. Single live runtime path

**Invariant.** After deploy, exactly one runtime code path is live for
each behavior. Old scripts/binaries are removed, not parallel-deployed.

**Bugs.** #76 (work.sh and worker.py both live; fix went into one).

**Status.** Operational; arguably a deployment invariant rather than a
runtime one. Mention in **D8 (#746) — self-restart topology** but
probably enforced by CI/PR review rather than runtime FSM.

---

## O. Build cache integrity — out of scope for Rocq

**Bugs.** #822 (workflow rebuilds without storing cache). CI/build
plumbing; not coordination.

---

## Summary

| Cluster | Invariant focus | Bugs | Already covered | Action |
|---|---|---|---|---|
| A | Session ownership FIFO | 3 | ✓ session_lock.v live | trim logs (#1037) |
| **B** | **Claude protocol stream + cancel scope** | **4** | **#1041 filed** | **claude_session.v** |
| C | Talker-kind coherence | 1 | folded into A? | audit session_lock.v |
| D | Task status FSM | 2 | partial in D3 | extend D3 (#741) |
| E | PR body ↔ tasks.json | 2 | D10 (#748) | cite bugs in issue |
| F | Reply / claim dedup | 6 | D1 (#739), D2 (#740) | extend with author + substance dedup |
| G | Picker + fresh-retry | 2 | D16 (#888) | cite bugs |
| H | Webhook ingress dedup + no-inline-impl | 4 | D9 (#747) for ids; **#1042 filed** | D9b done |
| I | Cache bootstrap + worker wake | 1 | D16 (#888) | explicit theorem |
| J | Rescope confluence | 1 | D11 (#749) | empirical anchor |
| K | PR transition gate | 2 | D18 (#890) | single-predicate theorem |
| L | Auto-resolve threads | 1 | D13 (#751) | sub-invariant |
| M | Status display | 2 | (UI, not Rocq) | standard fix |
| N | Single runtime path | 1 | (deploy, not runtime) | mention in D8 (#746) |
| O | Build cache | 1 | (CI, not coordination) | standard fix |

**New issues filed (as of this survey):**
- [#1041](https://github.com/FidoCanCode/home/issues/1041) — claude_session.v model (cluster B, 4 bugs of motivation)
- [#1042](https://github.com/FidoCanCode/home/issues/1042) — D9b: webhook handlers must not do implementation inline (cluster H)

**Existing issues to enrich with bug references:**
D3 (#741), D9 (#747), D10 (#748), D11 (#749), D13 (#751), D16 (#888), D18 (#890).
