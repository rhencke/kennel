(** Handler-preemption FSM: per-repo webhook-interrupt turn admission.

    Models the coordination contract between webhook-triggered interrupt
    work (comment/review triage, CI failure handling, and any rescope
    launched by that triage) and the worker turn boundary.  When interrupt
    work is waiting, the worker must not start a new provider turn — it
    yields until the interrupt blocker drains.

    [Empty]    — no webhook interrupt work pending for this repo.
                  The worker may start its next provider turn.
    [NonEmpty] — one or more webhook interrupt units are pending.
                  The worker must yield; only handlers may run.

    Three events name the observable transitions:

    [WebhookArrives]  — [enter_untriaged()] called on the HTTP thread
                        or by handler-owned rescope work.
                        [Empty] → [NonEmpty], [NonEmpty] → [NonEmpty].
    [HandlerDone]     — [exit_untriaged()] called when the handler
                        finishes processing.
                        [NonEmpty] → [NonEmpty] (if count > 1) or
                        [NonEmpty] → [Empty] (if count reaches 0).
                        Rejected from [Empty] (underflow).
    [WorkerTurnStart] — worker is about to call [provider_run()].
                        [Empty] → [Empty] (turn proceeds).
                        Rejected from [NonEmpty] — this is the
                        core invariant.

    Since the FSM tracks only {Empty, NonEmpty} and not the exact
    count, [WebhookArrives] from [NonEmpty] stays [NonEmpty] and
    [HandlerDone] from [NonEmpty] must be modeled as a transition
    back to [NonEmpty] — the runtime guards the [NonEmpty] → [Empty]
    transition on the actual count.  For the invariant proof, the
    key property is that [WorkerTurnStart] is rejected from
    [NonEmpty], which holds regardless of the exact count.

    Proved invariants:

      [worker_blocked_when_nonempty]  — [WorkerTurnStart] is rejected
                                       from [NonEmpty].  The worker must
                                       not start a new turn while the
                                       inbox has untriaged webhooks.
      [handler_done_rejected_from_empty] — [HandlerDone] is rejected
                                           from [Empty].  Underflow is
                                           a bug.
      [worker_turn_proceeds_when_empty]  — [WorkerTurnStart] from [Empty]
                                           yields [Some Empty].  When no
                                           webhooks are pending, the
                                           worker proceeds normally.
      [webhook_arrival_always_accepted]  — [WebhookArrives] is accepted
                                           from both [Empty] and
                                           [NonEmpty].  A new webhook can
                                           always enter the inbox. *)

From FidoModels Require Import preamble.

(** * State

    Two phases of the per-repo webhook interrupt blocker. *)
Inductive State : Type :=
| Empty    : State
| NonEmpty : State.

(** * Event

    [WebhookArrives]  — [enter_untriaged()]; any state → [NonEmpty].
    [HandlerDone]     — [exit_untriaged()]; [NonEmpty] → [NonEmpty]
                        (count-aware reduction handled at runtime).
    [WorkerTurnStart] — worker about to call [provider_run()];
                        [Empty] → [Empty]; rejected from [NonEmpty]. *)
Inductive Event : Type :=
| WebhookArrives  : Event
| HandlerDone     : Event
| WorkerTurnStart : Event.

(** * Transition function

    [transition current event] returns [Some new_state] when [event] is
    valid in [current], or [None] when it is rejected.

    Key rejection: [WorkerTurnStart] from [NonEmpty] — the core
    invariant.  The worker must yield when webhook interrupt work is
    pending.

    [HandlerDone] from [NonEmpty] stays [NonEmpty] in the FSM; the
    runtime uses the actual counter to decide when to flip to [Empty].
    This is safe because the invariant ([WorkerTurnStart] rejected
    from [NonEmpty]) holds regardless of count. *)
Definition transition (current : State) (event : Event) : option State :=
  match current, event with
  | Empty,    WebhookArrives  => Some NonEmpty
  | Empty,    WorkerTurnStart => Some Empty
  | Empty,    HandlerDone     => None
  | NonEmpty, WebhookArrives  => Some NonEmpty
  | NonEmpty, HandlerDone     => Some NonEmpty
  | NonEmpty, WorkerTurnStart => None
  end.

Python File Extraction handler_preemption "transition".

(** * Proved invariants *)

(** [worker_blocked_when_nonempty]: [WorkerTurnStart] is rejected from
    [NonEmpty].  This is the core guarantee — a worker must not start
    a new provider turn while the webhook interrupt blocker is non-empty.
    The worker yields at every turn boundary when [has_untriaged()]
    returns true, blocking until [wait_for_inbox_drain()] signals
    that all pending handlers and handler-owned rescope work have
    finished. *)
Lemma worker_blocked_when_nonempty :
  transition NonEmpty WorkerTurnStart = None.
Proof.
  reflexivity.
Qed.

(** [handler_done_rejected_from_empty]: [HandlerDone] is rejected from
    [Empty].  An [exit_untriaged()] call without a matching
    [enter_untriaged()] is a count underflow — the runtime logs a
    warning and refuses the decrement.  This proves the underflow
    check is correct: the FSM also rejects it. *)
Lemma handler_done_rejected_from_empty :
  transition Empty HandlerDone = None.
Proof.
  reflexivity.
Qed.

(** [worker_turn_proceeds_when_empty]: [WorkerTurnStart] from [Empty]
    yields [Some Empty].  When no untriaged webhooks are pending, the
    worker proceeds to its next provider turn without yielding.  The
    state remains [Empty] — a worker turn does not change the inbox
    state. *)
Lemma worker_turn_proceeds_when_empty :
  transition Empty WorkerTurnStart = Some Empty.
Proof.
  reflexivity.
Qed.

(** [webhook_arrival_always_accepted]: [WebhookArrives] is accepted
    from both [Empty] and [NonEmpty], always yielding [NonEmpty].
    A new webhook can always enter the inbox regardless of whether
    other untriaged webhooks are already pending. *)
Lemma webhook_arrival_always_accepted :
  transition Empty    WebhookArrives = Some NonEmpty /\
  transition NonEmpty WebhookArrives = Some NonEmpty.
Proof.
  split; reflexivity.
Qed.
