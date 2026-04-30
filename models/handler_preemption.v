(** Handler-preemption FSM: per-repo webhook-interrupt turn admission.

    Models the coordination contract between webhook-triggered interrupt
    work (comment/review triage, CI failure handling, and any rescope
    launched by that triage) and the worker turn boundary.  When interrupt
    work is waiting, the worker must not start a new provider turn — it
    yields until the interrupt blocker drains.

    [Empty]          — no webhook interrupt work pending for this repo.
                       The worker may start its next provider turn.
    [NonEmpty]       — one or more legacy in-memory webhook interrupt units
                       are pending.  The worker must yield; only handlers may
                       run.  Runtime code still uses this while the durable
                       FIFO migration lands.
    [DurableDemand]  — a webhook/comment demand has been committed to durable
                       state before provider cancellation.  Worker turns are
                       blocked even if no handler thread has acquired the
                       provider session yet.
    [PreemptedDemand] — durable demand exists and the provider interrupt signal
                        has been requested.  This is still a blocked state:
                        interrupt success is a latency detail, not the source
                        of correctness.

    Six events name the observable transitions:

    [WebhookArrives]  — [enter_untriaged()] called on the HTTP thread
                        or by handler-owned rescope work in the current
                        in-memory path.
                        [Empty] → [NonEmpty], [NonEmpty] → [NonEmpty].
    [DurableDemandRecorded] — a normalized webhook/comment demand was written
                              to durable local state.  This is the future
                              scheduler gate and must happen before interrupt.
    [InterruptRequested] — provider cancellation was requested after durable
                           demand was recorded.  Rejected from states without
                           durable demand.
    [HandlerDone]     — [exit_untriaged()] called when the handler
                        finishes processing.
                        [NonEmpty] → [NonEmpty] (if count > 1) or
                        [NonEmpty] → [Empty] (if count reaches 0).
                        Rejected from [Empty] (underflow).
    [DurableDemandDrained] — all durable demand for this repo has been claimed
                             and completed or made retryable.  Durable blocked
                             states return to [Empty].
    [WorkerTurnStart] — worker is about to call [provider_run()].
                        [Empty] → [Empty] (turn proceeds).
                        Rejected from every demand state — this is the core
                        scheduler-priority invariant.

    The legacy in-memory path still tracks only {Empty, NonEmpty} and not the
    exact count.  For that path, [WebhookArrives] from [NonEmpty] stays
    [NonEmpty] and [HandlerDone] from [NonEmpty] must be modeled as a
    transition back to [NonEmpty] — the runtime guards the [NonEmpty] →
    [Empty] transition on the actual count.  The durable path names the future
    store-backed gate explicitly: [DurableDemandRecorded] must happen before
    [InterruptRequested], and [WorkerTurnStart] is rejected until
    [DurableDemandDrained].

    Proved invariants:

      [worker_blocked_when_nonempty]  — [WorkerTurnStart] is rejected from
                                       [NonEmpty].  The worker must not start a
                                       new turn while the inbox has untriaged
                                       webhooks.
      [worker_blocked_until_durable_demand_drains] — [WorkerTurnStart] is
                                       rejected from [DurableDemand] and
                                       [PreemptedDemand].  Durable webhook
                                       demand owns scheduler priority.
      [interrupt_requires_durable_demand] — [InterruptRequested] is rejected
                                       until demand has been durably recorded.
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

    Four phases of the per-repo webhook interrupt blocker. *)
Inductive State : Type :=
| Empty           : State
| NonEmpty        : State
| DurableDemand   : State
| PreemptedDemand : State.

(** * Event

    [WebhookArrives]  — [enter_untriaged()]; legacy in-memory demand.
    [DurableDemandRecorded] — durable queue/store write committed before
                              interrupt.
    [InterruptRequested] — provider cancellation requested after durable write.
    [HandlerDone]     — [exit_untriaged()]; [NonEmpty] → [NonEmpty]
                        (count-aware reduction handled at runtime).
    [DurableDemandDrained] — durable demand is fully drained.
    [WorkerTurnStart] — worker about to call [provider_run()];
                        [Empty] → [Empty]; rejected from every demand state. *)
Inductive Event : Type :=
| WebhookArrives        : Event
| DurableDemandRecorded : Event
| InterruptRequested    : Event
| HandlerDone           : Event
| DurableDemandDrained  : Event
| WorkerTurnStart       : Event.

(** * Transition function

    [transition current event] returns [Some new_state] when [event] is
    valid in [current], or [None] when it is rejected.

    Key rejection: [WorkerTurnStart] from [NonEmpty], [DurableDemand], and
    [PreemptedDemand] — the core invariant.  The worker must yield when
    webhook interrupt work is pending, whether the blocker is still the legacy
    in-memory count or the future durable queue.

    Key ordering: [InterruptRequested] is accepted only after
    [DurableDemandRecorded].  Provider cancellation is useful for latency, but
    it cannot be the precondition that makes webhook handling durable.

    [HandlerDone] from [NonEmpty] stays [NonEmpty] in the FSM; the
    runtime uses the actual counter to decide when to flip to [Empty].
    This is safe because the invariant ([WorkerTurnStart] rejected
    from [NonEmpty]) holds regardless of count. *)
Definition transition (current : State) (event : Event) : option State :=
  match current, event with
  | Empty,           WebhookArrives        => Some NonEmpty
  | Empty,           DurableDemandRecorded => Some DurableDemand
  | Empty,           WorkerTurnStart       => Some Empty

  | NonEmpty,        WebhookArrives        => Some NonEmpty
  | NonEmpty,        DurableDemandRecorded => Some DurableDemand
  | NonEmpty,        HandlerDone           => Some NonEmpty
  | NonEmpty,        DurableDemandDrained  => Some Empty

  | DurableDemand,   WebhookArrives        => Some DurableDemand
  | DurableDemand,   DurableDemandRecorded => Some DurableDemand
  | DurableDemand,   InterruptRequested    => Some PreemptedDemand
  | DurableDemand,   DurableDemandDrained  => Some Empty

  | PreemptedDemand, WebhookArrives        => Some PreemptedDemand
  | PreemptedDemand, DurableDemandRecorded => Some PreemptedDemand
  | PreemptedDemand, InterruptRequested    => Some PreemptedDemand
  | PreemptedDemand, DurableDemandDrained  => Some Empty

  | _,               _                     => None
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

(** [worker_blocked_until_durable_demand_drains]: [WorkerTurnStart] is rejected
    while durable webhook/comment demand exists.  This prevents the worker
    release/reacquire race: once demand is committed to durable state, no new
    worker provider turn may start until that demand drains, regardless of
    whether provider interruption has already been requested. *)
Lemma worker_blocked_until_durable_demand_drains :
  transition DurableDemand WorkerTurnStart = None /\
  transition PreemptedDemand WorkerTurnStart = None.
Proof.
  split; reflexivity.
Qed.

(** [interrupt_requires_durable_demand]: [InterruptRequested] is rejected from
    [Empty] and [NonEmpty].  The interrupt RPC cannot be the first
    correctness step; demand must be recorded durably before cancellation is
    requested. *)
Lemma interrupt_requires_durable_demand :
  transition Empty    InterruptRequested = None /\
  transition NonEmpty InterruptRequested = None.
Proof.
  split; reflexivity.
Qed.

(** [durable_demand_precedes_preempted_demand]: recording demand moves to
    [DurableDemand], and only then can [InterruptRequested] move to
    [PreemptedDemand]. *)
Lemma durable_demand_precedes_preempted_demand :
  transition Empty         DurableDemandRecorded = Some DurableDemand /\
  transition DurableDemand InterruptRequested    = Some PreemptedDemand.
Proof.
  split; reflexivity.
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
  transition Empty           WebhookArrives = Some NonEmpty /\
  transition NonEmpty        WebhookArrives = Some NonEmpty /\
  transition DurableDemand   WebhookArrives = Some DurableDemand /\
  transition PreemptedDemand WebhookArrives = Some PreemptedDemand.
Proof.
  repeat split; reflexivity.
Qed.
