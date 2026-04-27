(** Worker-registry crash-recovery FSM: per-repo slot ownership and
    provider rescue.

    Models the coordination contract inside [WorkerRegistry.start] and
    the crash-recovery path driven by [Watchdog.run].  The FSM tracks
    the per-repo slot in [_threads] — not the internal mechanics of the
    thread itself (that is [worker_claude_lifecycle]) and not what the
    watchdog observes (that is [watchdog_transitions]).

    [Absent]  — no thread occupies this repo slot.  The initial state on
                 Fido startup before [registry.start] is first called.
    [Active]  — a thread is alive and running ([is_alive] returns [True]).
                 The slot is occupied; starting a second thread for the
                 same repo is refused.
    [Crashed] — the thread died unexpectedly ([crash_error] is set;
                 [_stop] is [False]).  The provider subprocess may still
                 be alive and is rescuable: [detach_provider] on the dead
                 thread reclaims it for the replacement.  Session intent
                 persists in [_session_issue] on the same dead thread.
                 A [Launch] (fresh start) is refused — the slot demands a
                 [Rescue] so the live provider is not orphaned.
    [Stopped] — the thread exited via an orderly [stop()] call ([_stop]
                 is [True]).  The provider was shut down or is no longer
                 rescuable; [Rescue] is refused.  A fresh [Launch] is
                 accepted to re-enable the repo.

    Four events name the observable transitions:

    [Launch]      — [start()] called with no crashable predecessor.
                    [Absent | Stopped] → [Active].
    [Rescue]      — [start()] called after a crash; provider rescued from
                    the dead thread via [detach_provider].
                    [Crashed] → [Active].
    [ThreadDies]  — thread exits unexpectedly; [crash_error] is set on
                    the dead thread object; provider subprocess survives.
                    [Active] → [Crashed].
    [ThreadStops] — thread exits via [stop()]; [_stop] is set.
                    [Active] → [Stopped].

    Five proved invariants capture the core guarantees:

      [rescue_requires_prior_crash]   — [Rescue] is rejected from every
                                        state except [Crashed].  A provider
                                        can only be reclaimed from a slot
                                        that actually crashed; not from an
                                        absent slot, a live slot, or a slot
                                        whose thread exited orderly.
      [no_start_while_active]         — both [Launch] and [Rescue] are
                                        rejected from [Active].  A live
                                        thread must die (crash or stop)
                                        before its slot can be reused.
      [crash_must_rescue]             — [Launch] is rejected from [Crashed].
                                        When a predecessor crashed, [start()]
                                        must use the rescue path — a fresh
                                        launch over a crashed slot would
                                        orphan the still-live provider
                                        subprocess.
      [thread_events_only_from_active] — [ThreadDies] and [ThreadStops]
                                        are rejected from every state except
                                        [Active].  Only a live thread can
                                        produce lifecycle events.
      [crash_recovery_is_total]       — [Rescue] from [Crashed] always
                                        yields [Some Active].  Crash
                                        recovery is a total function: every
                                        detected crash has a well-defined
                                        recovery path.

    E1 flip point: when the E-band work lands, [WorkerRegistry.start] can
    be split at the boundary — a thin entry point feeds the right event
    ([Launch] vs [Rescue]) into the extracted [transition] and asserts
    [Some Active] on success, replacing the hand-written [old_thread is
    not None and not old_thread.is_alive() and not old_thread._stop]
    guard entirely.  The oracle assertions become the control flow. *)

From FidoModels Require Import preamble.

(** * State

    Four phases of the per-repo worker slot as seen by the registry. *)
Inductive State : Type :=
| Absent  : State
| Active  : State
| Crashed : State
| Stopped : State.

(** * Event

    [Launch]      — [registry.start()] with no crashable predecessor.
                    [Absent | Stopped] → [Active].
    [Rescue]      — [registry.start()] after a crash; [detach_provider]
                    reclaims the live provider subprocess.
                    [Crashed] → [Active].
    [ThreadDies]  — thread exits unexpectedly; [crash_error] set;
                    provider subprocess survives.
                    [Active] → [Crashed].
    [ThreadStops] — thread exits via [stop()]; [_stop] set.
                    [Active] → [Stopped]. *)
Inductive Event : Type :=
| Launch      : Event
| Rescue      : Event
| ThreadDies  : Event
| ThreadStops : Event.

(** * Transition function

    [transition current event] returns [Some new_state] when [event] is
    valid in [current], or [None] when it is rejected.

    Key rejection groups:
    - [Rescue] is rejected from every state except [Crashed] — provider
      rescue requires a crashed predecessor, never an absent, active, or
      orderly-stopped one.
    - [Launch] and [Rescue] are both rejected from [Active] — a live
      thread must first produce a [ThreadDies] or [ThreadStops] before
      the slot can be reused.
    - [Launch] is rejected from [Crashed] — a crash demands [Rescue] so
      the still-live provider subprocess is not orphaned.
    - [ThreadDies] and [ThreadStops] are rejected from non-[Active]
      states — only a live thread can produce lifecycle events. *)
Definition transition (current : State) (event : Event) : option State :=
  match current, event with
  | Absent,  Launch      => Some Active
  | Active,  ThreadDies  => Some Crashed
  | Active,  ThreadStops => Some Stopped
  | Crashed, Rescue      => Some Active
  | Stopped, Launch      => Some Active
  | _,       _           => None
  end.

Python File Extraction worker_registry_crash "transition".

(** * Proved invariants

    All lemmas follow by computation ([reflexivity]).  The theorem names
    surface in the runtime oracle: when [WorkerRegistry.start] or the
    crash-restart path diverges from [transition], the crash message names
    the violated invariant. *)

(** [rescue_requires_prior_crash]: [Rescue] is rejected from [Absent],
    [Active], and [Stopped].  Only a [Crashed] slot holds a rescuable
    provider subprocess — the thread died unexpectedly with [_stop=False]
    and [crash_error] set.  Attempting rescue from any other state is
    refused: the absent slot has no predecessor, the active slot has a
    live thread, and the stopped slot had its provider shut down
    orderly.

    This is the machine-checked form of the registry invariant: a
    [detach_provider] call is always paired with a thread whose slot is
    [Crashed], never one whose thread exited for any other reason. *)
Lemma rescue_requires_prior_crash :
  transition Absent  Rescue = None /\
  transition Active  Rescue = None /\
  transition Stopped Rescue = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [no_start_while_active]: both [Launch] and [Rescue] are rejected from
    [Active].  The registry cannot install a second thread while one is
    already alive — [is_alive] returning [True] means the slot is
    occupied.  The active thread must produce [ThreadDies] or
    [ThreadStops] before the slot becomes reusable.

    This machine-checks the single-active-thread-per-repo rule: the
    [_threads] dict maps each repo name to exactly one live [WorkerThread]
    at a time, and the FSM enforces that the transition to a new thread
    always passes through [Crashed] or [Stopped] first. *)
Lemma no_start_while_active :
  transition Active Launch = None /\
  transition Active Rescue = None.
Proof.
  split; reflexivity.
Qed.

(** [crash_must_rescue]: [Launch] is rejected from [Crashed].  When the
    slot's previous thread died unexpectedly, [start()] must take the
    rescue path — a fresh [Launch] over a [Crashed] slot would leave the
    still-live provider subprocess running with no owner, and the session
    context stored in [_session_issue] on the dead thread would be lost.
    The FSM enforces that the only way out of [Crashed] is via [Rescue]. *)
Lemma crash_must_rescue :
  transition Crashed Launch = None.
Proof.
  reflexivity.
Qed.

(** [thread_events_only_from_active]: [ThreadDies] and [ThreadStops] are
    rejected from every state except [Active].  A thread lifecycle event
    can only fire for an alive thread — an absent slot has no thread to
    die, a crashed slot's thread is already dead, and a stopped slot's
    thread has already exited.

    This prevents double-counting: a thread cannot crash or stop twice,
    and a slot that was never occupied cannot produce lifecycle events. *)
Lemma thread_events_only_from_active :
  transition Absent  ThreadDies  = None /\
  transition Absent  ThreadStops = None /\
  transition Crashed ThreadDies  = None /\
  transition Crashed ThreadStops = None /\
  transition Stopped ThreadDies  = None /\
  transition Stopped ThreadStops = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [crash_recovery_is_total]: [Rescue] from [Crashed] always yields
    [Some Active].  Crash recovery is a total function from the crashed
    state — there is no conditional path where a crashed slot stays
    crashed without producing a new active thread.  Every crash the
    watchdog detects leads to [WorkerRegistry.start] which always
    succeeds in creating and starting the replacement thread. *)
Lemma crash_recovery_is_total :
  transition Crashed Rescue = Some Active.
Proof.
  reflexivity.
Qed.
