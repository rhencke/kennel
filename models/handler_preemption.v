(** Handler-preemption FSM: per-repo webhook-interrupt turn admission.

    Models the coordination contract between webhook-triggered interrupt work
    (comment/review triage, CI failure handling, and handler-owned rescope
    work) and the worker turn boundary.  When either legacy in-memory handler
    demand or durable queued demand exists, the worker must not start a new
    provider turn.

    The state is a product, not a phase enum:

    [legacy_demand]       — whether the current in-memory untriaged handler
                            counter is empty or non-empty.
    [durable_demand]      — whether durable queued webhook/comment demand is
                            empty or non-empty.
    [provider_interrupt]  — whether a provider interrupt has been requested.

    The demand fields are independent.  Durable demand is recorded when the
    webhook dispatch path has committed queued work, before best-effort
    provider cancellation or background handler execution.  Legacy demand is
    recorded when the HTTP handler enters the in-memory untriaged inbox.
    Either field may become non-empty first, and either may drain first; worker
    admission remains blocked until both are empty.

    Provider interrupt state is observable ordering evidence: the runtime must
    record [InterruptRequested] only after durable demand exists, so an
    interrupt can never stand in for lost durable work.  It is not authority
    for demand existence: [WorkerTurnStart] is accepted exactly when both
    demand fields are empty, regardless of interrupt state.

    Six events name the observable transitions:

    [WebhookArrives]        — legacy in-memory demand enters the untriaged
                              inbox.  Sets [legacy_demand] to non-empty and
                              preserves durable demand.
    [DurableDemandRecorded] — durable queue/store demand has been committed.
                              Sets [durable_demand] to non-empty and preserves
                              legacy demand.
    [InterruptRequested]    — provider cancellation was requested for already
                              recorded durable demand.  Sets only
                              [provider_interrupt].
    [HandlerDone]           — one legacy handler completes.  Clears only
                              [legacy_demand] in this boolean abstraction.
    [DurableDemandDrained]  — durable demand drains.  Clears only
                              [durable_demand].
    [WorkerTurnStart]       — worker is about to call [provider_run()].
                              Accepted exactly when both demand fields are
                              empty. *)

From FidoModels Require Import preamble.

(** * State *)

Inductive LegacyDemand : Type :=
| LegacyEmpty    : LegacyDemand
| LegacyNonEmpty : LegacyDemand.

Inductive DurableDemandState : Type :=
| DurableEmpty    : DurableDemandState
| DurableNonEmpty : DurableDemandState.

Inductive ProviderInterrupt : Type :=
| InterruptNotRequested : ProviderInterrupt
| InterruptWasRequested : ProviderInterrupt.

Record State : Type := {
  legacy_demand : LegacyDemand;
  durable_demand : DurableDemandState;
  provider_interrupt : ProviderInterrupt
}.

Definition empty_state : State := {|
  legacy_demand := LegacyEmpty;
  durable_demand := DurableEmpty;
  provider_interrupt := InterruptNotRequested
|}.

Definition legacy_state : State := {|
  legacy_demand := LegacyNonEmpty;
  durable_demand := DurableEmpty;
  provider_interrupt := InterruptNotRequested
|}.

Definition durable_state : State := {|
  legacy_demand := LegacyEmpty;
  durable_demand := DurableNonEmpty;
  provider_interrupt := InterruptNotRequested
|}.

Definition preempted_durable_state : State := {|
  legacy_demand := LegacyEmpty;
  durable_demand := DurableNonEmpty;
  provider_interrupt := InterruptWasRequested
|}.

Definition mixed_state : State := {|
  legacy_demand := LegacyNonEmpty;
  durable_demand := DurableNonEmpty;
  provider_interrupt := InterruptWasRequested
|}.

(** * Event *)

Inductive Event : Type :=
| WebhookArrives        : Event
| DurableDemandRecorded : Event
| InterruptRequested    : Event
| HandlerDone           : Event
| DurableDemandDrained  : Event
| WorkerTurnStart       : Event.

(** * Helpers *)

Definition with_legacy (s : State) (legacy : LegacyDemand) : State := {|
  legacy_demand := legacy;
  durable_demand := durable_demand s;
  provider_interrupt := provider_interrupt s
|}.

Definition with_durable
    (s : State)
    (durable : DurableDemandState) : State := {|
  legacy_demand := legacy_demand s;
  durable_demand := durable;
  provider_interrupt := provider_interrupt s
|}.

Definition with_interrupt
    (s : State)
    (interrupt : ProviderInterrupt) : State := {|
  legacy_demand := legacy_demand s;
  durable_demand := durable_demand s;
  provider_interrupt := interrupt
|}.

(** * Transition function *)

Definition transition (current : State) (event : Event) : option State :=
  match event with
  | WebhookArrives =>
      Some (with_legacy current LegacyNonEmpty)

  | DurableDemandRecorded =>
      Some (with_durable current DurableNonEmpty)

  | InterruptRequested =>
      match durable_demand current with
      | DurableNonEmpty =>
          Some (with_interrupt current InterruptWasRequested)
      | DurableEmpty => None
      end

  | HandlerDone =>
      match legacy_demand current with
      | LegacyNonEmpty => Some (with_legacy current LegacyEmpty)
      | LegacyEmpty => None
      end

  | DurableDemandDrained =>
      match durable_demand current with
      | DurableNonEmpty => Some (with_durable current DurableEmpty)
      | DurableEmpty => None
      end

  | WorkerTurnStart =>
      match legacy_demand current, durable_demand current with
      | LegacyEmpty, DurableEmpty => Some current
      | _, _ => None
      end
  end.

Python File Extraction handler_preemption
  "empty_state legacy_state durable_state preempted_durable_state mixed_state transition".

(** * Proved invariants *)

Lemma worker_blocked_when_legacy_nonempty :
  transition legacy_state WorkerTurnStart = None.
Proof.
  reflexivity.
Qed.

Lemma worker_blocked_until_durable_demand_drains :
  transition durable_state WorkerTurnStart = None /\
  transition preempted_durable_state WorkerTurnStart = None /\
  transition mixed_state WorkerTurnStart = None.
Proof.
  repeat split; reflexivity.
Qed.

Lemma worker_admission_ignores_interrupt_state :
  transition {|
    legacy_demand := LegacyEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |} WorkerTurnStart =
  Some {|
    legacy_demand := LegacyEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |}.
Proof.
  reflexivity.
Qed.

Lemma interrupt_requires_durable_demand :
  transition empty_state InterruptRequested = None /\
  transition legacy_state InterruptRequested = None.
Proof.
  split; reflexivity.
Qed.

Lemma durable_demand_precedes_interrupt_request :
  transition empty_state DurableDemandRecorded = Some durable_state /\
  transition durable_state InterruptRequested = Some preempted_durable_state.
Proof.
  split; reflexivity.
Qed.

Lemma handler_done_rejected_without_legacy_demand :
  transition empty_state HandlerDone = None /\
  transition durable_state HandlerDone = None.
Proof.
  split; reflexivity.
Qed.

Lemma handler_done_clears_only_legacy_demand :
  transition mixed_state HandlerDone = Some {|
    legacy_demand := LegacyEmpty;
    durable_demand := DurableNonEmpty;
    provider_interrupt := InterruptWasRequested
  |}.
Proof.
  reflexivity.
Qed.

Lemma durable_drain_clears_only_durable_demand :
  transition mixed_state DurableDemandDrained = Some {|
    legacy_demand := LegacyNonEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |}.
Proof.
  reflexivity.
Qed.

Lemma worker_turn_proceeds_when_demands_empty :
  transition empty_state WorkerTurnStart = Some empty_state.
Proof.
  reflexivity.
Qed.

Lemma webhook_arrival_preserves_durable_demand :
  transition durable_state WebhookArrives = Some {|
    legacy_demand := LegacyNonEmpty;
    durable_demand := DurableNonEmpty;
    provider_interrupt := InterruptNotRequested
  |}.
Proof.
  reflexivity.
Qed.

Lemma durable_record_preserves_legacy_demand :
  transition legacy_state DurableDemandRecorded = Some {|
    legacy_demand := LegacyNonEmpty;
    durable_demand := DurableNonEmpty;
    provider_interrupt := InterruptNotRequested
  |}.
Proof.
  reflexivity.
Qed.

Lemma mixed_handler_done_first_blocks_until_durable_drains :
  transition mixed_state HandlerDone = Some {|
    legacy_demand := LegacyEmpty;
    durable_demand := DurableNonEmpty;
    provider_interrupt := InterruptWasRequested
  |} /\
  transition {|
    legacy_demand := LegacyEmpty;
    durable_demand := DurableNonEmpty;
    provider_interrupt := InterruptWasRequested
  |} WorkerTurnStart = None /\
  transition {|
    legacy_demand := LegacyEmpty;
    durable_demand := DurableNonEmpty;
    provider_interrupt := InterruptWasRequested
  |} DurableDemandDrained = Some {|
    legacy_demand := LegacyEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |}.
Proof.
  repeat split; reflexivity.
Qed.

Lemma mixed_durable_drain_first_blocks_until_handler_done :
  transition mixed_state DurableDemandDrained = Some {|
    legacy_demand := LegacyNonEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |} /\
  transition {|
    legacy_demand := LegacyNonEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |} WorkerTurnStart = None /\
  transition {|
    legacy_demand := LegacyNonEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |} HandlerDone = Some {|
    legacy_demand := LegacyEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |}.
Proof.
  repeat split; reflexivity.
Qed.

Lemma durable_first_mixed_path_blocks_until_both_demands_drain :
  transition empty_state DurableDemandRecorded = Some durable_state /\
  transition durable_state InterruptRequested = Some preempted_durable_state /\
  transition preempted_durable_state WebhookArrives = Some mixed_state /\
  transition mixed_state DurableDemandDrained = Some {|
    legacy_demand := LegacyNonEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |} /\
  transition {|
    legacy_demand := LegacyNonEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |} WorkerTurnStart = None /\
  transition {|
    legacy_demand := LegacyNonEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |} HandlerDone = Some {|
    legacy_demand := LegacyEmpty;
    durable_demand := DurableEmpty;
    provider_interrupt := InterruptWasRequested
  |}.
Proof.
  repeat split; reflexivity.
Qed.
