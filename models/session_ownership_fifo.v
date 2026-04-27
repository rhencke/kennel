(** Session-ownership FIFO: full queue-based ownership with worker deferral.

    Extends the binary session lock of [session_lock] into a per-session
    FIFO queue.  Non-worker contenders ([Handler], [CronSweep]) queue and
    drain in arrival order.  The background worker uses a separate
    [Deferred] slot and runs only when the queue is empty and the slot
    is set.

    Worker resumption is itself preemptible: any [Enqueue] while the worker
    is [WorkerActive] implicitly re-defers the worker and appends the new
    contender to the queue tail.  There is no explicit [Preempt] event —
    preemption is structural, not a named transition.

    This is the first cross-[.v] import in the repo: [Release] carries a
    [ReviewReplyOutcome] from [replied_comment_claims], letting the FIFO
    state machine observe the outcome without owning reply logic.

    D4 lands as specification ahead of implementation.  Today's webhook
    path blocks the HTTP thread for the full turn duration
    ([server.py:776-782]); the instant-exit [Enqueue]/[Dequeue] split is a
    future alignment tracked in D5 #743.  Six theorems over [transition]
    are proved below. *)

From FidoModels Require Import preamble.
From FidoModels Require Import replied_comment_claims.

From Stdlib Require Import
  Lists.List.

Import ListNotations.

(** * Contender

    Non-worker session-holder kinds.  [Handler] covers webhook-handler and
    CI-fix turns (CI is folded into Handler; no rank ordering inside the
    queue).  [CronSweep] covers the periodic idle-drain sweep.  The
    background worker never enters the FIFO queue — it is managed through
    the [fifo_worker_deferred] slot. *)
Inductive Contender : Type :=
| Handler   : Contender
| CronSweep : Contender.

(** * ActiveSlot

    Who currently holds the session.
    [Idle]             — nobody holds; the queue may be non-empty (pending
                         a [Dequeue]) or the worker may be deferred.
    [HolderActive c]   — the FIFO contender [c] holds the session.
    [WorkerActive]     — the background worker holds the session. *)
Inductive ActiveSlot : Type :=
| Idle                         : ActiveSlot
| HolderActive (c : Contender) : ActiveSlot
| WorkerActive                 : ActiveSlot.

(** * FifoState

    Full per-session FIFO ownership state.

    [fifo_queue]           — ordered list of pending contenders; head is
                             next to be activated by [Dequeue].
    [fifo_active_slot]     — who currently holds the session.
    [fifo_worker_deferred] — [true] when the worker is parked and waiting
                             to resume after the queue drains. *)
Record FifoState : Type := {
  fifo_queue           : list Contender;
  fifo_active_slot     : ActiveSlot;
  fifo_worker_deferred : bool
}.

(** * Event

    Five events cover all ownership transitions.  No explicit [Preempt]
    event exists — worker preemption is implicit in [Enqueue] when the
    worker holds the session.

    [Enqueue c]        — a contender arrives; total, always succeeds.  If
                         the worker is [WorkerActive], it is implicitly
                         re-deferred and [c] appends to the queue tail.
    [Dequeue]          — activate the queue head; only valid when [Idle]
                         and the queue is non-empty.
    [WorkerDefer]      — the worker explicitly parks into the deferred
                         slot; only valid when [WorkerActive].
    [WorkerResume]     — the worker re-activates; only valid when [Idle],
                         the queue is empty, and the deferred slot is set.
    [Release outcome]  — the current holder relinquishes; valid from
                         [HolderActive] or [WorkerActive]; rejected from
                         [Idle]. *)
Inductive Event : Type :=
| Enqueue     (c : Contender)                : Event
| Dequeue                                    : Event
| WorkerDefer                                : Event
| WorkerResume                               : Event
| Release     (outcome : ReviewReplyOutcome) : Event.

(** * Transition function

    [transition s event] returns [Some s'] when [event] is valid in [s],
    or [None] when it is rejected.

    [Enqueue] is the only total event — it always returns [Some _] — and
    is the formal statement that webhooks exit instantly: [Enqueue] never
    blocks on the current holder state. *)
Definition transition (s : FifoState) (event : Event) : option FifoState :=
  match event with

  | Enqueue c =>
      (* Total.  Worker preemption is implicit: if the worker is active,
         re-defer it and append the new contender. *)
      match fifo_active_slot s with
      | WorkerActive =>
          Some {| fifo_queue           := fifo_queue s ++ [c];
                  fifo_active_slot     := Idle;
                  fifo_worker_deferred := true |}
      | _ =>
          Some {| fifo_queue           := fifo_queue s ++ [c];
                  fifo_active_slot     := fifo_active_slot s;
                  fifo_worker_deferred := fifo_worker_deferred s |}
      end

  | Dequeue =>
      (* Activate head.  Rejected if the slot is occupied or queue empty. *)
      match fifo_active_slot s, fifo_queue s with
      | Idle, c :: rest =>
          Some {| fifo_queue           := rest;
                  fifo_active_slot     := HolderActive c;
                  fifo_worker_deferred := fifo_worker_deferred s |}
      | _, _ => None
      end

  | WorkerDefer =>
      (* Explicit park.  Only valid when the worker is active. *)
      match fifo_active_slot s with
      | WorkerActive =>
          Some {| fifo_queue           := fifo_queue s;
                  fifo_active_slot     := Idle;
                  fifo_worker_deferred := true |}
      | _ => None
      end

  | WorkerResume =>
      (* Re-activate the deferred worker.  Only valid when queue is empty
         and the deferred slot is set. *)
      match fifo_active_slot s, fifo_queue s, fifo_worker_deferred s with
      | Idle, [], true =>
          Some {| fifo_queue           := [];
                  fifo_active_slot     := WorkerActive;
                  fifo_worker_deferred := false |}
      | _, _, _ => None
      end

  | Release _ =>
      (* Holder relinquishes.  Valid from [HolderActive] or [WorkerActive];
         rejected from [Idle]. *)
      match fifo_active_slot s with
      | Idle => None
      | _    =>
          Some {| fifo_queue           := fifo_queue s;
                  fifo_active_slot     := Idle;
                  fifo_worker_deferred := fifo_worker_deferred s |}
      end

  end.

Python File Extraction session_ownership_fifo "transition".

(** * Proved invariants

    All six lemmas follow by computation ([reflexivity] or [destruct] +
    [reflexivity]): [transition] reduces on the concrete constructor shapes,
    and Rocq's kernel verifies the equalities by normalisation.

    These names surface in the runtime oracle: when the Python implementation
    diverges from [transition], the crash message includes the theorem name so
    the engineer knows exactly which invariant was violated. *)

(** [no_dual_ownership]: when a slot is occupied — either by a FIFO contender
    ([HolderActive]) or by the background worker ([WorkerActive]) — neither
    [Dequeue] nor [WorkerResume] can hand the session to a second caller.
    Ownership is always singular. *)
Lemma no_dual_ownership :
  forall c q d,
    transition {| fifo_queue := q; fifo_active_slot := HolderActive c;
                  fifo_worker_deferred := d |} Dequeue     = None /\
    transition {| fifo_queue := q; fifo_active_slot := HolderActive c;
                  fifo_worker_deferred := d |} WorkerResume = None /\
    transition {| fifo_queue := q; fifo_active_slot := WorkerActive;
                  fifo_worker_deferred := d |} Dequeue     = None /\
    transition {| fifo_queue := q; fifo_active_slot := WorkerActive;
                  fifo_worker_deferred := d |} WorkerResume = None.
Proof.
  intros c q d; repeat split; reflexivity.
Qed.

(** [release_only_by_owner]: [Release] is rejected from [Idle].  Nobody can
    relinquish a session that is not held — the D3 invariant extended to the
    D4 FIFO model. *)
Lemma release_only_by_owner :
  forall q d outcome,
    transition {| fifo_queue := q; fifo_active_slot := Idle;
                  fifo_worker_deferred := d |} (Release outcome) = None.
Proof.
  intros q d outcome; reflexivity.
Qed.

(** [fifo_order_preserved]: [Enqueue c] always appends [c] to the tail of
    the queue.  When [transition] succeeds (which it always does for [Enqueue]),
    the resulting queue is the old queue with [c] appended.  This is the
    formal statement that FIFO order is maintained on every arrival. *)
Lemma fifo_order_preserved :
  forall s c,
    match transition s (Enqueue c) with
    | Some s' => fifo_queue s' = fifo_queue s ++ [c]
    | None    => True
    end.
Proof.
  intros [q a d] c; destruct a; reflexivity.
Qed.

(** [worker_resume_requires_empty]: [WorkerResume] is rejected when the queue
    is non-empty.  The worker may only re-activate after every queued contender
    has been drained — preventing the worker from stealing turns ahead of
    waiting handlers. *)
Lemma worker_resume_requires_empty :
  forall c rest d,
    transition {| fifo_queue := c :: rest; fifo_active_slot := Idle;
                  fifo_worker_deferred := d |} WorkerResume = None.
Proof.
  intros c rest d; reflexivity.
Qed.

(** [worker_singleton]: [WorkerResume] is rejected when the worker is already
    [WorkerActive].  The model enforces at most one active worker slot at a
    time — a second [WorkerResume] while already active is always refused. *)
Lemma worker_singleton :
  forall q d,
    transition {| fifo_queue := q; fifo_active_slot := WorkerActive;
                  fifo_worker_deferred := d |} WorkerResume = None.
Proof.
  intros q d; reflexivity.
Qed.

(** [enqueue_total]: [Enqueue] always returns [Some _] — it never fails.  This
    is the formal statement that webhooks exit instantly: no matter who holds
    the session or what is in the queue, a new contender can always be
    appended. *)
Lemma enqueue_total :
  forall s c, exists s', transition s (Enqueue c) = Some s'.
Proof.
  intros [q a d] c; destruct a; eexists; reflexivity.
Qed.
