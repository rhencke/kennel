# Bug-Mined Coordination Invariants

**Source:** 23 closed `Bug:` issues + 8 open ones from
[FidoCanCode/home](https://github.com/FidoCanCode/home),
2026-04-08 → 2026-04-27.
**Filed as:** [#1040](https://github.com/FidoCanCode/home/issues/1040)
**PR:** [#1043](https://github.com/FidoCanCode/home/pull/1043)

Each cluster maps a class of bugs to one invariant, the subsystem that
should enforce it, the proposed Rocq model, and the existing D-series
issue (if any) that should absorb the work.

**Status (2026-04-26):** Empirical bug citations added to all listed
D-series issues: D1 (#739), D2 (#740), D3 (#741), D9 (#747), D10 (#748),
D11 (#749), D13 (#751), D16 (#888), D18 (#890).
Cluster B modeled and live: `claude_session.v` + oracle in `ClaudeSession`
([#1052](https://github.com/FidoCanCode/home/pull/1052)).
Cluster P modeled and live: `worker_registry_crash.v` + oracle in `WorkerRegistry`
([#1056](https://github.com/FidoCanCode/home/pull/1056)).

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

**Status.** Modeled in `claude_session.v`. Live in production
([#1052](https://github.com/FidoCanCode/home/pull/1052), validated
2026-04-26). Four invariants formally proved: `single_writer`,
`cancel_does_not_persist_across_turns`, `empty_result_is_not_completion`,
`drain_terminates`. FSM oracle wired into `ClaudeSession._stream_transition`
— crashes loudly on any protocol violation. Regression tests in
`tests/test_claude_stream_fsm_oracle.py`.

**Model.** `claude_session.v` — states `Idle | Sending |
AwaitingReply | Draining | Cancelled`; events `Send | ReplyChunk |
ReplyEnd | CancelFire | DrainObserve | TurnReturn`.

**Closed:** [#1041](https://github.com/FidoCanCode/home/issues/1041)
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

**E1 flip point.** Today, `tasks.json` and the PR body are migration
surfaces: Python writes the durable task list, renders the work-queue
projection, and the extracted D10 oracle checks that returned states match
that projection. When the scheduler/reducer boundary becomes authoritative,
`task_complete` should stay an explicit reducer transition, while new reviewer
input should arrive as a `change_request(request : str)` command whose rescope
step lets the LLM rewrite the PR description and remaining tasks to fit the
new ask. A rescoped task may carry zero to many origin comments because later,
broader asks can absorb earlier ones. The rescope transition must preserve
that provenance so same-author absorbed asks can receive one final reply
instead of repeated pings, while tasks absorbed from other authors still get an
outbox notification that their task was folded into a larger one. Grouping a
task with another one should not count as a material deletion. The reducer
commits the durable task store first, then runs the PR-body update and any
notifications as outbox effects, making the handwritten sync choreography in
`sync_tasks` replaceable glue rather than the source of the invariant.

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
set**, **D2 (#740) — reply-promise lifecycle**, and **D14 (#752) —
durable reply/outbox protocol**. D14 is live as a runtime oracle around visible
reply posting and deferred tracking issue creation: Python projects durable
reply promises, semantic origins, delivery ids, and outbox effects into the
model before it records or reuses the external artifact. The remaining model
surface is `(a)` author validation in marker recovery and `(b)`
substance-equivalence dedup vs. queued thread tasks.

**E1 flip point.** D14 currently checks the handwritten Python side-effect
order: the durable promise exists first, the oracle accepts a reply-post or
deferred-issue effect, and Python then records the visible artifact or
deferred issue row for reuse. At E1, that extracted transition should become
the authoritative reducer/outbox boundary: Python commits the modeled reply
promise and outbox decision first, then posts the GitHub reply or opens the
tracking issue only as an emitted outbox effect, with delivered external ids
fed back through the same transition for idempotent reuse.

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

## J. Rescope — task identity is the durable id

**Invariant.** Rescope may reorder tasks, mark them completed, insert new
ones, revise mutable text metadata (title, description) for an existing
task id, and re-target the source-comment anchor. The durable task id
is the only invariant identity field. When the anchor changes, the
previous anchor is preserved in the task's `lineage_comment_ids` origin
metadata so reply/resolve paths can still walk back to the original
commenter.

**Bugs.** #1024 (rescope rewrote thread-task titles, collapsing
unrelated tasks). The "collapse" half is fixed by **#1665** (distinct
comments at the add boundary always produce distinct tasks — lineage is
no longer a dedup join key); the "rewrote titles" half is now an
intentional capability under **#1713** (title flows through the
reducer as mutable metadata for an existing id). The thread anchor
joins the mutable-metadata camp under **#1714** (RewriteAnchor in the
reducer; lineage preservation in the Python adapter). Explicit
removal lands under **#1716** (status="completed" → CompleteTask) and
explicit merge under **#1717** (MergeTasks folds source lineages into
the target row; lineage_comments is now a TaskRow field; the Rocq
predicate `merge_preserves_source_lineage` proves no source comment id
is lost).

**Status.** Modeled in **D11 (#749) — model rescope confluence**.
The original "title is immutable" framing has been reshaped by epic
**#1340**: identity is the id, not the text. See #1665, #1713, #1714,
#1666, #1667 for the per-leaf invariants.

**E1 flip point.** D11 currently runs as a runtime oracle around
`reorder_tasks`: Python translates Opus's omission-based result into explicit
ACT/DO rescope releases, compares the handwritten output with
`apply_batched_rescope`, and fails closed on divergence. At E1, that extracted
transition stops being a checker and becomes the durable rescope reducer: the
Python path should commit the modeled order/rows first, then run notifications,
wakeups, and PR-body sync as outbox effects after the commit.

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

**Status.** Covered by **D13 (#751) — model thread auto-resolve with
race coverage**. The Rocq model is the oracle for today's
`Tasks.complete_with_resolve`, `Worker.resolve_addressed_threads`, and
resolved-thread duplicate suppression paths: Python projects GitHub
thread state plus durable task rows into the model, then only emits
`resolveReviewThread` when the oracle returns `ResolveReviewThread`.
Resolved-thread webhook comments are admitted only when the modeled
queue decision says the latest actionable input is fresh.

**E1 flip point.** When the scheduler/reducer boundary becomes
authoritative, D13 should stop being a side oracle around Python
resolution checks. The reducer should commit the task/feedback command
state first, then return a resolve-thread outbox effect only when the
same transition proves no pending same-thread work remains and the latest
actionable comment is Fido's.

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

## P. Worker registry slot lifecycle + crash recovery

**Invariant.** Each per-repo slot in `_threads` passes through a
forward-only FSM: `Absent → Active → (Crashed | Stopped) → Active`.
Four guarantees hold simultaneously:
`rescue_requires_prior_crash` — `Rescue` (i.e., `detach_provider`) is
accepted only from `Crashed`; an absent, live, or orderly-stopped slot
has no rescuable provider.
`no_start_while_active` — both `Launch` and `Rescue` are rejected from
`Active`; a live thread must exit before the slot can be reused, closing
the fido-lockfile race where a replacement thread starts before the old
one has a chance to exit.
`crash_must_rescue` — `Launch` is rejected from `Crashed`; a crashed
predecessor demands `Rescue` so the still-live provider subprocess is
not orphaned and session intent on the dead thread is not lost.
`crash_recovery_is_total` — `Rescue` from `Crashed` always yields
`Active`; every detected crash has a well-defined recovery path with no
stuck intermediate state.

**Bugs.** Lockfile race (old thread still alive when new one starts),
provider-orphaning on bypassed rescue path — both motivated the
`no_start_while_active` and `crash_must_rescue` invariants.

**Status.** Modeled in `worker_registry_crash.v`. Live in production
([#1056](https://github.com/FidoCanCode/home/pull/1056), validated
2026-04-26). Five invariants formally proved:
`rescue_requires_prior_crash`, `no_start_while_active`,
`crash_must_rescue`, `thread_events_only_from_active`,
`crash_recovery_is_total`. FSM oracle wired into
`WorkerRegistry._registry_fsm_transition` — crashes loudly on any slot
lifecycle violation. Regression tests in
`tests/test_registry_fsm_oracle.py`.

**Model.** `worker_registry_crash.v` — states `Absent | Active |
Crashed | Stopped`; events `Launch | Rescue | ThreadDies | ThreadStops`.

**Closed:** [#745](https://github.com/FidoCanCode/home/issues/745)
— D7: Rocq→Python — model multi-repo worker registry + crash recovery.

---

## Q. Handler preemption — webhook interrupt turn admission

**Invariant.** When either legacy in-memory handler demand or durable
queued webhook demand for a repo is non-empty, the worker must not start
a new provider turn for that repo. Formally:
`legacy_demand(r) ≠ ∅ ∨ durable_demand(r) ≠ ∅ ⟹ next_turn(r) ∈ Handler`.
The worker yields at every turn boundary when comments, review feedback,
CI failures, or handler-owned rescope work are waiting, blocking until
both demand sources drain.

**Bugs.** [#1067](https://github.com/FidoCanCode/home/issues/1067)
(worker grinds through in-progress task without checking for pending
webhooks — fresh comments and CI events are starved).

**Model.** `handler_preemption.v` — product state
`legacy_demand × durable_demand × provider_interrupt`; events
`WebhookArrives`, `DurableDemandRecorded`, `InterruptRequested`,
`HandlerDone`, `DurableDemandDrained`, `WorkerTurnStart`. Theorems:
`WorkerTurnStart` is rejected while either demand field is non-empty,
durable demand must precede interrupt requests, provider interrupt state
does not authorize or block worker admission, and mixed durable/legacy
interleavings remain blocked until both demand sources drain.

**Status.** Runtime implementation live and covered by the extracted
oracle: per-repo `enter_untriaged` / `exit_untriaged` legacy demand on
`WorkerRegistry`, enqueue-time durable demand and interrupt recording in
`WebhookHandler`, pre-provider turn admission in `Worker.execute_task`,
CI-failure interrupts, and handler-owned rescope blocking. The original
runtime path landed in
[#1070](https://github.com/FidoCanCode/home/pull/1070); product-state
demand modeling and enqueue-time wiring are tracked by
[#1132](https://github.com/FidoCanCode/home/issues/1132) /
[#1134](https://github.com/FidoCanCode/home/pull/1134).

---

## R. Session-lock liveness — bounded hold time + force-release escape

**Invariant.** No FSM lock acquire can wait forever for a holder that
will never release.  The original `session_lock.v` model proved
**safety** (`no_dual_ownership`, `release_only_by_owner`) but every
transition out of an owned state required the holder to *voluntarily*
fire its release event — a property no real-world IO substrate can
guarantee.  Liveness adds the dual property: from every state, at
least one event drives the FSM to `Free`, and the watchdog's
escape-hatch event is accepted in every state.

Formally proved in `session_lock.v`:
- `force_release_to_free`: ∀s, transition(s, ForceRelease) = Some Free.
- `unconditional_liveness`: ∃ev, ∀s, transition(s, ev) = Some Free —
  strong liveness, with `ForceRelease` as the witness.  The watchdog
  fires that single event without first inspecting FSM state and is
  guaranteed `Free` regardless of where the holder was.
- `every_state_reaches_free`: ∀s, ∃ev, transition(s, ev) = Some Free —
  weaker form, named to document the *primary* path: voluntary
  release for owned states, ForceRelease only for `Free` (idempotent
  self-loop) and the unhappy case.  Strictly weaker than
  `unconditional_liveness`; both are kept for pedagogical clarity.
- `owned_state_exit_paths` (worker + handler): the only events that
  produce `Some _` from an owned state are the corresponding
  voluntary `*Release`, `Preempt` (worker only), and `ForceRelease`.
  Confirms adding `ForceRelease` did not open any other accidental
  exit edges.

**Bugs.** [#1377](https://github.com/FidoCanCode/home/issues/1377)
(handler thread wedged inside `consume_until_result` on a
streaming-forever subprocess; FSM lock leaked indefinitely; webhooks
queued at HandlerAcquire positions 1, 2, … forever; no exception
ever raised because `idle_timeout` kept getting reset by streamed
events).

**Model.** `session_lock.v` extended with `ForceRelease : Event` and
three transition arms (`Free | OwnedByWorker | OwnedByHandler` →
`Free`).  The new lemma set complements the original safety pair —
sibling property of `no_dual_ownership`, not a replacement.

**IO substrate.** `session_lock_io.v` — separate reference model
(no Python extraction; OS provides the actual transitions).
Captures the subprocess lifecycle (`Spawned | Killed | Reaped`) and
couples it with the lock FSM via two composite events:

- `WatchdogPreempt` — cooperative preemption.  Webhook fires
  `_fire_worker_cancel`, worker drains its turn, lock transfers via
  the existing `Preempt` event.  Subprocess stays alive — the new
  holder uses it without a respawn.
- `WatchdogEvict` — forcible eviction.  Watchdog fires both
  `ForceRelease` on the lock and `IssueKill` on the subprocess.
  Lock advances to `Free`, subprocess advances to `Killed`.  The
  wedged holder's `select` returns EOF when the OS closes stdout
  (`OsCloseStdout`).

Headline theorems:

- `kill_eof_chain_terminates`: from `Spawned`, `IssueKill` then
  `StdoutEof` always reaches `Reaped`.  IO-side liveness analogue
  of `force_release_to_free`.
- `evict_releases_lock`: for any prior lock state and a `Spawned`
  subprocess, `WatchdogEvict` lands the coupled state at
  `(Free, Killed)`.  The full coordination handshake in one step.
- `evict_then_eof_reaps`: `WatchdogEvict` followed by `OsCloseStdout`
  reaches `(Free, Reaped)` — the *complete* recovery: lock available,
  subprocess fully torn down, stdout closed, holder unblocked.
- `preempt_does_not_kill_subprocess`: cooperative preemption
  transfers ownership without touching the subprocess.
- `evict_kills_subprocess` + `preempt_and_evict_distinct_outcomes`:
  the two paths produce structurally different coupled states —
  the model proves the cooperative-vs-forcible distinction at the
  IO layer, not just in the docstring.

This is the "Rocq IO" piece — formalises the chain (1) watchdog
fires kill → (2) OS propagates EOF → (3) holder's `iter_events`
sees EOF and breaks → (4) holder's `__exit__` runs → (5)
`_evicted_tids` guard skips `_fsm_release` — at the substrate level.
Steps (3)–(5) are runtime implementation guarded by unit + smoke
tests; (1)–(2) are now first-class theorems.  The model also pins
down that webhook preemption *honors* the worker thread by
preserving subprocess state, separate from the eviction path.

**Status.** Runtime implementation live: `OwnedSession.force_release`
fires the modeled event through the existing oracle wrapper,
`_evicted_tids` carries cross-thread eviction state so the wedged
holder's eventual `__exit__` skips the now-incorrect normal release,
`ClaudeSession._on_force_release` kills the subprocess so the parked
`select()` returns EOF and the holder escapes
`consume_until_result`, and `SessionLockWatchdog` is the daemon that
fires `force_release` when a holder has held past
`hold_deadline_seconds` (default 900 s).  Started from `server.run`
alongside the existing `Watchdog` and `ReconcileWatchdog`.

The wiring deliberately models the queued-handler transfer as two
oracle calls (`ForceRelease → Free` then `HandlerAcquire → OwnedByHandler`)
rather than the single-step direct mutation that `_fsm_release` does
for the same case — a small precedent toward closing that earlier
shortcut.

---

## Summary

| Cluster | Invariant focus | Bugs | Already covered | Action |
|---|---|---|---|---|
| A | Session ownership FIFO | 3 | ✓ session_lock.v live | trim logs (#1037) |
| B | Claude protocol stream + cancel scope | 4 | ✓ claude_session.v live (#1052) | — |
| C | Talker-kind coherence | 1 | folded into A? | audit session_lock.v |
| D | Task status FSM | 2 | partial in D3 | extend D3 (#741) |
| E | PR body ↔ tasks.json | 2 | D10 (#748) | cite bugs in issue |
| F | Reply / claim dedup | 6 | D1 (#739), D2 (#740), D14 (#752) | author + substance dedup remain |
| G | Picker + fresh-retry | 2 | D16 (#888) | cite bugs |
| H | Webhook ingress dedup + no-inline-impl | 4 | D9 (#747) for ids; **#1042 filed** | D9b done |
| I | Cache bootstrap + worker wake | 1 | D16 (#888) | explicit theorem |
| J | Rescope confluence | 1 | D11 (#749) | empirical anchor |
| K | PR transition gate | 2 | D18 (#890) | single-predicate theorem |
| L | Auto-resolve threads | 1 | D13 (#751) | sub-invariant |
| M | Status display | 2 | (UI, not Rocq) | standard fix |
| N | Single runtime path | 1 | (deploy, not runtime) | mention in D8 (#746) |
| O | Build cache | 1 | (CI, not coordination) | standard fix |
| P | Worker registry slot lifecycle + crash recovery | lockfile race, provider orphan | ✓ worker_registry_crash.v live (#1056) | — |
| Q | Handler preemption — product demand admission | 1 (#1067) | ✓ handler_preemption.v live (#1134) | — |
| R | Session-lock liveness — bounded hold + force release | 1 (#1377) | ✓ session_lock.v ForceRelease live | — |

**New issues filed (as of this survey):**
- [#1041](https://github.com/FidoCanCode/home/issues/1041) — claude_session.v model (cluster B, 4 bugs of motivation) — **closed** ([#1052](https://github.com/FidoCanCode/home/pull/1052))
- [#1042](https://github.com/FidoCanCode/home/issues/1042) — D9b: webhook handlers must not do implementation inline (cluster H)
- [#745](https://github.com/FidoCanCode/home/issues/745) — D7: worker_registry_crash.v model (cluster P, lockfile race + provider orphan) — **closed** ([#1056](https://github.com/FidoCanCode/home/pull/1056))

**Existing issues to enrich with bug references:**
D3 (#741), D9 (#747), D10 (#748), D11 (#749), D13 (#751), D16 (#888), D18 (#890).

**Runtime-first (model pending):**
- None currently listed from this survey.
