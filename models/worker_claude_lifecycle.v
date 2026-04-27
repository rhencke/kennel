(** Worker-Claude lifecycle FSM: per-repo background worker subprocess state.

    Tracks whether the worker's Claude subprocess exists and what
    task-execution mode it is in.  Composed orthogonally with
    [session_ownership_fifo]: the FIFO model owns session contention;
    this model owns subprocess existence and execution status.

    [Absent]   — subprocess does not exist.  Worker has no live Claude
                 session: either before the first [Spawn], after an
                 explicit [Shutdown], or following a full Fido restart.
    [Idle]     — subprocess alive; worker waiting for the next issue or
                 sleeping between polls.
    [Active]   — subprocess running a task; worker holds the FIFO
                 [WorkerActive] session slot.
    [Deferred] — subprocess alive but worker parked by an implicit FIFO
                 [Enqueue] preemption; waiting for the handler queue to
                 drain before resuming.

    Seven events name every lifecycle step.  [Shutdown] succeeds from
    any live state — subprocess termination is always safe.  [Restart]
    is restricted to [Idle]: model-change respawns and issue-boundary
    session resets ([TurnSessionMode.FRESH]) must not interrupt a
    running task or a deferred worker.  [TakeWork], [Yield], [Resume],
    and [Finish] mirror the D4 [session_ownership_fifo] worker events:
    the two FSMs are composed orthogonally across the same transitions.

    Six proved invariants capture which events are valid in which states. *)

From FidoModels Require Import preamble.

(** * WorkerState

    The four lifecycle phases of the background worker's Claude subprocess. *)
Inductive WorkerState : Type :=
| Absent   : WorkerState
| Idle     : WorkerState
| Active   : WorkerState
| Deferred : WorkerState.

(** * WorkerEvent

    [Spawn]    — create the Claude subprocess; [Absent] → [Idle].
    [TakeWork] — pick up the next issue; [Idle] → [Active].
    [Yield]    — implicit preemption by FIFO [Enqueue] while [Active];
                 [Active] → [Deferred].
    [Resume]   — FIFO queue has drained; [Deferred] → [Active].
    [Finish]   — task complete, session released; [Active] → [Idle].
    [Restart]  — model-change respawn or issue-boundary session reset;
                 [Idle] → [Idle] with a fresh subprocess and conversation.
    [Shutdown] — kill subprocess; [Idle | Active | Deferred] → [Absent]. *)
Inductive WorkerEvent : Type :=
| Spawn    : WorkerEvent
| TakeWork : WorkerEvent
| Yield    : WorkerEvent
| Resume   : WorkerEvent
| Finish   : WorkerEvent
| Restart  : WorkerEvent
| Shutdown : WorkerEvent.

(** * Transition function

    [transition w ev] returns [Some w'] when [ev] is valid in state [w],
    or [None] when it is rejected.

    [Shutdown] is valid from any live state and always returns [Absent] —
    subprocess termination is unconditional.  [Spawn] is the only event
    valid from [Absent].  [Restart] is the only event that leaves the
    subprocess in a live-but-replaced state from [Idle]. *)
Definition transition (w : WorkerState) (ev : WorkerEvent) : option WorkerState :=
  match w, ev with
  | Absent,   Spawn    => Some Idle
  | Idle,     TakeWork => Some Active
  | Idle,     Restart  => Some Idle
  | Idle,     Shutdown => Some Absent
  | Active,   Yield    => Some Deferred
  | Active,   Finish   => Some Idle
  | Active,   Shutdown => Some Absent
  | Deferred, Resume   => Some Active
  | Deferred, Shutdown => Some Absent
  | _,        _        => None
  end.

Python File Extraction worker_claude_lifecycle "transition".

(** * Proved invariants

    All six lemmas follow by computation ([reflexivity]): [transition]
    reduces on concrete (state, event) pairs, and Rocq's kernel verifies
    the equalities by normalisation.  No induction is needed.

    These names surface in the runtime oracle: when the Python
    implementation diverges from [transition], the crash message includes
    the theorem name so the engineer knows exactly which invariant was
    violated. *)

(** [spawn_only_from_absent]: [Spawn] is rejected from every live state.
    A subprocess cannot be created while one already exists — double
    [Spawn] is always refused. *)
Lemma spawn_only_from_absent :
  transition Idle     Spawn = None /\
  transition Active   Spawn = None /\
  transition Deferred Spawn = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [shutdown_clears_live_states]: [Shutdown] always succeeds from every
    live state and always returns [Absent].  Subprocess termination is
    unconditional — the worker can always be killed regardless of what
    task it is running. *)
Lemma shutdown_clears_live_states :
  transition Idle     Shutdown = Some Absent /\
  transition Active   Shutdown = Some Absent /\
  transition Deferred Shutdown = Some Absent.
Proof.
  repeat split; reflexivity.
Qed.

(** [work_only_from_idle]: [TakeWork] is rejected from every state except
    [Idle].  A worker cannot take a new task while absent, already active,
    or parked waiting for a handler to drain the session queue. *)
Lemma work_only_from_idle :
  transition Absent   TakeWork = None /\
  transition Active   TakeWork = None /\
  transition Deferred TakeWork = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [yield_only_from_active]: [Yield] is rejected unless the worker is
    [Active].  Only a running task can be preempted by a FIFO [Enqueue];
    an idle or absent worker has nothing to yield. *)
Lemma yield_only_from_active :
  transition Absent   Yield = None /\
  transition Idle     Yield = None /\
  transition Deferred Yield = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [resume_only_from_deferred]: [Resume] is rejected unless the worker is
    [Deferred].  The worker can only re-activate after an explicit [Yield]
    parks it; resuming from [Idle] or [Active] is refused. *)
Lemma resume_only_from_deferred :
  transition Absent   Resume = None /\
  transition Idle     Resume = None /\
  transition Active   Resume = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [restart_only_when_idle]: [Restart] is rejected from [Absent],
    [Active], and [Deferred].  Model-change respawns and issue-boundary
    session resets may only fire between tasks — never mid-execution and
    never while the worker is parked waiting for a handler. *)
Lemma restart_only_when_idle :
  transition Absent   Restart = None /\
  transition Active   Restart = None /\
  transition Deferred Restart = None.
Proof.
  repeat split; reflexivity.
Qed.
