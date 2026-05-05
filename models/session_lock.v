(** Session-lock FSM: first proved coordination model.

    Defines the ownership state machine for [OwnedSession] —
    the base class shared by [ClaudeSession] and [CopilotCLISession].

    [State] captures who currently holds the session lock (nobody,
    the background worker, or a webhook handler).  [Event] names the
    transitions a caller can request.  [transition] is the purely
    functional step function extracted to Python and run as a
    runtime-asserted oracle on every real state change.

    Six invariants are proved by computation ([reflexivity]) or simple
    case analysis:
      [no_dual_ownership]            — acquiring an already-owned session fails.
      [release_only_by_owner]        — voluntary releases only succeed for the owner.
      [force_release_to_free]        — [ForceRelease] always lands in [Free].
      [unconditional_liveness]       — strong liveness: there exists a *single*
                                       event ([ForceRelease]) that drives every
                                       state to [Free].  The watchdog fires this
                                       event without first inspecting FSM state.
      [every_state_reaches_free]     — weak liveness: every state has at least
                                       one event reaching [Free] — citing
                                       voluntary releases for owned states to
                                       document them as the primary path.
      [owned_state_exit_paths]       — the only events that move out of an owned state
                                       are the corresponding [*Release], [Preempt]
                                       (worker only), and [ForceRelease].

    Liveness is the property the model previously lacked: every transition
    out of an owned state required the holder to *voluntarily* fire its
    release event, with no escape hatch for "holder is wedged in IO and
    will never fire that event".  [ForceRelease] is that escape hatch —
    fired by a watchdog when a holder has held the lock past a deadline,
    or when an out-of-band IO failure (subprocess crash, broken pipe,
    runaway streaming) has prevented the normal release path from running.
    The watchdog also closes the wedged subprocess so the parked holder
    thread escapes [consume_until_result] cleanly via EOF (closes #1377).

    Divergence between [OwnedSession] and [transition] crashes loudly
    with the theorem name so the violated invariant is immediately
    visible in the traceback. *)

From FidoModels Require Import preamble.

(** * State

    The session has exactly one owner at a time.
    [Free] — nobody holds the lock.
    [OwnedByWorker] — the background worker thread holds the session.
    [OwnedByHandler] — a webhook handler holds the session via
    [hold_for_handler], typically while running triage → reply → react. *)
Inductive State :=
  | Free           : State
  | OwnedByWorker  : State
  | OwnedByHandler : State.

(** * Event

    [WorkerAcquire] / [HandlerAcquire] — regular acquisition requests.
    [WorkerRelease] / [HandlerRelease] — current owner voluntarily relinquishes.
    [Preempt] — handler forcibly takes the session from a running worker
    (corresponds to [hold_for_handler preempt_worker=True]); transfers
    ownership directly to a known successor handler.
    [ForceRelease] — watchdog evicts the current holder when the holder
    has held the lock past a deadline, or when an IO failure prevented
    voluntary release.  Lands in [Free]; no successor is assumed. *)
Inductive Event :=
  | WorkerAcquire  : Event
  | HandlerAcquire : Event
  | WorkerRelease  : Event
  | HandlerRelease : Event
  | Preempt        : Event
  | ForceRelease   : Event.

(** * Transition function

    [transition current event] returns [Some new_state] when [event] is
    valid in [current], or [None] when it is rejected.

    Rejection is fail-closed: there is no silent clobbering.  A handler
    must send [Preempt] to take a session from a worker; a plain
    [HandlerAcquire] against an occupied session is refused.

    [ForceRelease] is the only event accepted in every state — it is the
    sanctioned escape hatch for a wedged or dead holder.  See the
    [force_release_to_free] and [every_state_reaches_free] lemmas. *)
Definition transition (current : State) (event : Event) : option State :=
  match current with
  | Free =>
      match event with
      | WorkerAcquire  => Some OwnedByWorker
      | HandlerAcquire => Some OwnedByHandler
      | ForceRelease   => Some Free
      | _              => None
      end
  | OwnedByWorker =>
      match event with
      | WorkerRelease  => Some Free
      | Preempt        => Some OwnedByHandler
      | ForceRelease   => Some Free
      | _              => None
      end
  | OwnedByHandler =>
      match event with
      | HandlerRelease => Some Free
      | ForceRelease   => Some Free
      | _              => None
      end
  end.

Python Extraction transition.

(** * Proved invariants

    All lemmas follow by computation ([reflexivity]) or simple case
    analysis: [transition] reduces to a determinate result on each
    combination, and Rocq's kernel verifies the equality by
    normalisation.  No induction is needed.

    These names surface in the runtime oracle: when [OwnedSession]
    diverges from [transition], the crash message includes the theorem
    name so the engineer knows exactly which invariant was violated. *)

(** [no_dual_ownership]: a second acquire is always rejected when the
    session is already owned.  Both roles are blocked — explicit
    [Preempt] is the only sanctioned takeover path.  [ForceRelease]
    does not weaken this: it lands in [Free], not in another owned
    state, so a forced eviction is still followed by a fresh acquire
    via the normal path. *)
Lemma no_dual_ownership :
  transition OwnedByWorker  WorkerAcquire  = None /\
  transition OwnedByWorker  HandlerAcquire = None /\
  transition OwnedByHandler WorkerAcquire  = None /\
  transition OwnedByHandler HandlerAcquire = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [release_only_by_owner]: a *voluntary* release succeeds only when
    the releasing role is the current owner.  Cross-releases (handler
    releasing a worker-held session, worker releasing a handler-held
    session) are rejected, as are releases from the [Free] state.
    [ForceRelease] is the only sanctioned cross-role release path —
    see [force_release_to_free] below. *)
Lemma release_only_by_owner :
  transition OwnedByHandler WorkerRelease  = None /\
  transition OwnedByWorker  HandlerRelease = None /\
  transition Free           WorkerRelease  = None /\
  transition Free           HandlerRelease = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [force_release_to_free]: [ForceRelease] is accepted in every state
    and always lands in [Free].  This is the post-condition the
    watchdog relies on: after firing [ForceRelease], the lock is
    guaranteed acquirable by the next [WorkerAcquire] or
    [HandlerAcquire], regardless of the prior holder. *)
Lemma force_release_to_free :
  forall s, transition s ForceRelease = Some Free.
Proof.
  intros s; destruct s; reflexivity.
Qed.

(** [unconditional_liveness]: strong liveness.  There exists a *single*
    event that drives every state to [Free] — that event is
    [ForceRelease].  The shape is [exists ev, forall s, ...], strictly
    stronger than the [forall s, exists ev, ...] of
    [every_state_reaches_free]: the latter leaves open the possibility
    that no single event works universally and the watchdog would
    have to inspect FSM state before choosing what to fire.

    This is the property the watchdog actually relies on: it calls
    [force_release] without first reading [_fsm_state], and the
    Rocq-modeled transition is guaranteed to land in [Free]
    regardless of where the holder was.  Proved trivially as a
    corollary of [force_release_to_free]. *)
Theorem unconditional_liveness :
  exists ev, forall s, transition s ev = Some Free.
Proof. exists ForceRelease; exact force_release_to_free. Qed.

(** [every_state_reaches_free]: weak liveness.  From every state there
    is at least one event that drives the FSM to [Free].  Strictly
    weaker than [unconditional_liveness], which provides a single
    universal event; this lemma instead names a *different* event per
    state — voluntary releases for owned states — to document them as
    the primary path that real holders take.  [ForceRelease] is the
    safety net for [Free] (idempotent self-loop) and the only
    available event in the unhappy case where the holder will never
    fire its voluntary release. *)
Lemma every_state_reaches_free :
  forall s, exists ev, transition s ev = Some Free.
Proof.
  intros s; destruct s.
  - exists ForceRelease;   reflexivity.
  - exists WorkerRelease;  reflexivity.
  - exists HandlerRelease; reflexivity.
Qed.

(** [owned_state_exit_paths]: structural exhaustiveness.  The *only*
    events that move out of an owned state (i.e. that produce
    [Some _]) are the corresponding voluntary [*Release], [Preempt]
    (worker only), and [ForceRelease].  Pins down the design: adding
    [ForceRelease] did not open any other accidental exit edges. *)
Lemma owned_worker_exit_paths :
  forall ev s',
    transition OwnedByWorker ev = Some s' ->
    (ev = WorkerRelease /\ s' = Free) \/
    (ev = Preempt       /\ s' = OwnedByHandler) \/
    (ev = ForceRelease  /\ s' = Free).
Proof.
  intros ev s' H; destruct ev; simpl in H; inversion H; auto.
Qed.

Lemma owned_handler_exit_paths :
  forall ev s',
    transition OwnedByHandler ev = Some s' ->
    (ev = HandlerRelease /\ s' = Free) \/
    (ev = ForceRelease   /\ s' = Free).
Proof.
  intros ev s' H; destruct ev; simpl in H; inversion H; auto.
Qed.
