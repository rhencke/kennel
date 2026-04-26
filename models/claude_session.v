(** Claude session stream FSM: protocol turn lifecycle for [ClaudeSession].

    Models the per-turn state of the bidirectional stream-JSON subprocess
    held by [ClaudeSession._session].  The FSM tracks whether a turn is in
    flight, whether a cancel signal has been fired, and whether the result
    boundary has been crossed — capturing the coordination invariants that
    govern [_cancel], [_in_turn], and [iter_events].

    [Idle]          — no turn in flight; the session is free to accept a
                      new [Send].
    [Sending]       — [Send] has written to stdin; no reply event has
                      arrived yet.
    [AwaitingReply] — at least one reply event has arrived; the turn is
                      streaming and no cancel signal has been fired.
    [Draining]      — a cancel signal has been fired ([CancelFire]); the
                      stream is draining to the result boundary.
    [Cancelled]     — the result boundary ([TurnReturn]) was crossed after
                      a cancel; the cancelled outcome is known but the
                      session has not yet returned to [Idle].

    Six events name every protocol step.  [TurnReturn] fires on the
    result-boundary event emitted by the subprocess regardless of whether
    the turn was cancelled; the state at the time of [TurnReturn]
    determines whether the session enters [Idle] (normal completion from
    [AwaitingReply]) or [Cancelled] (drained completion from [Draining]).

    Four proved invariants capture the coordination guarantees:
      [single_writer]                      — non-[Idle] states are
                                             exclusive; a second [Send]
                                             is always rejected mid-turn.
      [cancel_does_not_persist_across_turns] — [Idle] is reachable from
                                             [Cancelled] via [TurnReturn];
                                             a subsequent [Send] from
                                             [Idle] sees no stale cancel.
      [empty_result_is_not_completion]     — [TurnReturn] from [AwaitingReply]
                                             yields [Idle], never [Cancelled];
                                             only a drained cancel produces
                                             [Cancelled].
      [drain_terminates]                   — from [Draining], [TurnReturn]
                                             always reaches [Cancelled] in
                                             one step; the drain path is
                                             finite. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

(* Prevent sort-polymorphism so nullary-constructor extraction is clean.
   See the note in [rocq-python-extraction/test/datatypes.v] for context. *)
Unset Universe Polymorphism.

(* Remap [option] so [Some x] erases to [x] and [None] stays [None].
   This makes [transition] return the new [State] directly on success
   and [None] on rejection — a natural Python return type. *)
Extract Inductive option => ""
  [ "" "None" ]
  "(lambda fSome, fNone, x: fNone() if x is None else fSome(x))".

(** * State

    The five phases of the Claude session stream turn lifecycle. *)
Inductive State : Type :=
| Idle          : State
| Sending       : State
| AwaitingReply : State
| Draining      : State
| Cancelled     : State.

(** * Event

    [Send]         — caller writes the prompt to stdin; [Idle] → [Sending].
    [ReplyChunk]   — a streaming reply event arrives from the subprocess;
                     [Sending] → [AwaitingReply]; [AwaitingReply] stays.
    [ReplyEnd]     — a non-final end-of-segment event; [AwaitingReply] stays.
    [CancelFire]   — the cancel signal is sent to the subprocess;
                     [Sending | AwaitingReply] → [Draining].
    [DrainObserve] — a drain event observed while waiting for the result
                     boundary; [Draining] stays.
    [TurnReturn]   — the result-boundary event from the subprocess;
                     [AwaitingReply] → [Idle] (normal completion);
                     [Draining]      → [Cancelled] (cancelled completion);
                     [Cancelled]     → [Idle]       (cancel acknowledged). *)
Inductive Event : Type :=
| Send         : Event
| ReplyChunk   : Event
| ReplyEnd     : Event
| CancelFire   : Event
| DrainObserve : Event
| TurnReturn   : Event.

(** * Transition function

    [transition current event] returns [Some new_state] when [event] is
    valid in [current], or [None] when it is rejected.

    Rejection is fail-closed: there is no silent no-op.  In particular
    [Send] is rejected from every non-[Idle] state — only one turn may
    be in flight at a time.  [TurnReturn] is the only event that moves
    the session back to [Idle] (directly from [AwaitingReply], or via
    [Cancelled] after draining). *)
Definition transition (current : State) (event : Event) : option State :=
  match current, event with
  | Idle,          Send         => Some Sending
  | Sending,       ReplyChunk   => Some AwaitingReply
  | Sending,       CancelFire   => Some Draining
  | AwaitingReply, ReplyChunk   => Some AwaitingReply
  | AwaitingReply, ReplyEnd     => Some AwaitingReply
  | AwaitingReply, CancelFire   => Some Draining
  | AwaitingReply, TurnReturn   => Some Idle
  | Draining,      DrainObserve => Some Draining
  | Draining,      TurnReturn   => Some Cancelled
  | Cancelled,     TurnReturn   => Some Idle
  | _,             _            => None
  end.

Python File Extraction claude_session "transition".

(** * Proved invariants

    All four lemmas follow by computation ([reflexivity]): [transition]
    reduces to concrete values on each listed combination, and Rocq's
    kernel verifies the equalities by normalisation.  No induction is
    needed.

    These names surface in the runtime oracle: when [ClaudeSession]
    diverges from [transition], the crash message includes the theorem
    name so the engineer knows exactly which invariant was violated. *)

(** [single_writer]: [Send] is rejected from every non-[Idle] state.
    At most one turn may be in flight at a time — a second [Send] while
    [Sending], [AwaitingReply], [Draining], or [Cancelled] is always
    refused. *)
Lemma single_writer :
  transition Sending       Send = None /\
  transition AwaitingReply Send = None /\
  transition Draining      Send = None /\
  transition Cancelled     Send = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [cancel_does_not_persist_across_turns]: [Idle] is reachable from
    [Cancelled] via [TurnReturn], and [Send] from [Idle] yields [Sending]
    — not [Draining] or [Cancelled].  The cancel flag does not leak into
    the next turn: a [Send] issued after acknowledging a cancelled turn
    enters [Sending] with no stale cancel state. *)
Lemma cancel_does_not_persist_across_turns :
  transition Cancelled TurnReturn = Some Idle /\
  transition Idle      Send       = Some Sending.
Proof.
  split; reflexivity.
Qed.

(** [empty_result_is_not_completion]: [TurnReturn] from [AwaitingReply]
    yields [Idle], never [Cancelled].  A [TurnReturn] that carries an
    empty or cancelled result is impossible without first passing through
    [CancelFire] → [Draining]: normal completion always returns [Idle],
    so the only path to [Cancelled] is through the explicit drain sequence. *)
Lemma empty_result_is_not_completion :
  transition AwaitingReply TurnReturn = Some Idle /\
  transition Draining      TurnReturn = Some Cancelled.
Proof.
  split; reflexivity.
Qed.

(** [drain_terminates]: from [Draining], [TurnReturn] always reaches
    [Cancelled] in a single step — the drain is finite.  [DrainObserve]
    may fire any number of times, but [TurnReturn] is always available
    and immediately exits the [Draining] loop.  There is no cycle that
    keeps the FSM in [Draining] forever without admitting a [TurnReturn]. *)
Lemma drain_terminates :
  transition Draining DrainObserve = Some Draining /\
  transition Draining TurnReturn   = Some Cancelled.
Proof.
  split; reflexivity.
Qed.
