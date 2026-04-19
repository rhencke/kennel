(** Session-lock FSM: first proved coordination model.

    Defines the ownership state machine for [OwnedSession] —
    the base class shared by [ClaudeSession] and [CopilotCLISession].

    [State] captures who currently holds the session lock (nobody,
    the background worker, or a webhook handler).  [Event] names the
    transitions a caller can request.  [transition] is the purely
    functional step function extracted to Python and run as a
    runtime-asserted oracle on every real state change.

    Two key invariants are proved by computation ([reflexivity]):
      [no_dual_ownership]     — acquiring an already-owned session fails.
      [release_only_by_owner] — releasing only succeeds for the owner.

    Divergence between [OwnedSession] and [transition] crashes loudly
    with the theorem name so the violated invariant is immediately
    visible in the traceback. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

(* Prevent sort-polymorphism so nullary-constructor extraction is clean.
   See the note in [rocq-python-extraction/test/phase4.v] for context. *)
Unset Universe Polymorphism.

(* Remap [option] so [Some x] erases to [x] and [None] stays [None].
   This makes [transition] return the new [State] directly on success
   and [None] on rejection — a natural Python return type. *)
Extract Inductive option => ""
  [ "" "None" ]
  "(lambda fSome, fNone, x: fNone() if x is None else fSome(x))".

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
    [WorkerRelease] / [HandlerRelease] — current owner relinquishes.
    [Preempt] — handler forcibly takes the session from a running worker
    (corresponds to [hold_for_handler preempt_worker=True]). *)
Inductive Event :=
  | WorkerAcquire  : Event
  | HandlerAcquire : Event
  | WorkerRelease  : Event
  | HandlerRelease : Event
  | Preempt        : Event.

(** * Transition function

    [transition current event] returns [Some new_state] when [event] is
    valid in [current], or [None] when it is rejected.

    Rejection is fail-closed: there is no silent clobbering.  A handler
    must send [Preempt] to take a session from a worker; a plain
    [HandlerAcquire] against an occupied session is refused. *)
Definition transition (current : State) (event : Event) : option State :=
  match current with
  | Free =>
      match event with
      | WorkerAcquire  => Some OwnedByWorker
      | HandlerAcquire => Some OwnedByHandler
      | _              => None
      end
  | OwnedByWorker =>
      match event with
      | WorkerRelease  => Some Free
      | Preempt        => Some OwnedByHandler
      | _              => None
      end
  | OwnedByHandler =>
      match event with
      | HandlerRelease => Some Free
      | _              => None
      end
  end.

Python Extraction transition.

(** * Proved invariants

    Both lemmas follow by computation ([reflexivity]): [transition]
    reduces to [None] on each listed combination, and Rocq's kernel
    verifies the equality by normalisation.  No induction is needed.

    These names surface in the runtime oracle: when [OwnedSession]
    diverges from [transition], the crash message includes the theorem
    name so the engineer knows exactly which invariant was violated. *)

(** [no_dual_ownership]: a second acquire is always rejected when the
    session is already owned.  Both roles are blocked — explicit
    [Preempt] is the only sanctioned takeover path. *)
Lemma no_dual_ownership :
  transition OwnedByWorker  WorkerAcquire  = None /\
  transition OwnedByWorker  HandlerAcquire = None /\
  transition OwnedByHandler WorkerAcquire  = None /\
  transition OwnedByHandler HandlerAcquire = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [release_only_by_owner]: a release succeeds only when the releasing
    role is the current owner.  Cross-releases (handler releasing a
    worker-held session, worker releasing a handler-held session) are
    rejected, as are releases from the [Free] state. *)
Lemma release_only_by_owner :
  transition OwnedByHandler WorkerRelease  = None /\
  transition OwnedByWorker  HandlerRelease = None /\
  transition Free           WorkerRelease  = None /\
  transition Free           HandlerRelease = None.
Proof.
  repeat split; reflexivity.
Qed.
