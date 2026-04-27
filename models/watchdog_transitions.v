(** Watchdog thread-lifecycle FSM: running/crashed/restarting/hung worker states.

    Models the coordination contract between [Watchdog.run] and the per-repo
    [WorkerThread] it supervises.  The FSM captures what the watchdog
    observes and what it is allowed to do — not the internal mechanics of
    the worker itself.

    [Running]    — thread alive ([is_alive] returns [True]); worker is
                   executing normally.  [Watchdog.run] takes no action.
    [Crashed]    — thread dead ([is_alive] returns [False]) due to an
                   unhandled exception; [crash_error] is set on the dead
                   thread.  The provider subprocess is still alive and will
                   be rescued ([detach_provider]) when the replacement
                   thread starts.  Task intent persists in [state.json].
    [Restarting] — watchdog has called [registry.start], provider has been
                   rescued from the crashed thread, and the replacement
                   thread has been started.  Brief transient; the next
                   watchdog tick either sees [Running] or [Crashed] again.
    [Hung]       — thread alive but no activity reported for longer than
                   [_STALE_THRESHOLD] (600 s).  Display-only: the watchdog
                   never forcibly restarts a live thread.  The claude
                   subprocess has its own idle timeout; a forced restart
                   races on the fido lockfile and caused a restart loop in
                   production before this rule was established.
    [Stopped]    — thread exited via an orderly [stop()] call.  Terminal:
                   the watchdog does not restart a stopped thread.

    Eight events name the observable transitions.

    Five proved invariants capture the core guarantees:
      [alive_never_forcibly_restarted] — [WatchdogDetectDead] is rejected
                                         from [Running] and [Hung]; the
                                         watchdog cannot restart a live
                                         thread via the dead-detection path.
      [hung_not_restarted]             — [WatchdogDetectAlive] from [Hung]
                                         stays [Hung]; no spontaneous
                                         restart of a stale-but-alive
                                         thread.
      [dead_always_restarted]          — [WatchdogDetectDead] from [Crashed]
                                         always yields [Some Restarting];
                                         restart is a total function from
                                         the crashed state.
      [restart_reaches_running]        — [RestartComplete] from [Restarting]
                                         yields [Some Running]; a successful
                                         restart always returns the thread
                                         to operational state.
      [stopped_is_terminal]            — from [Stopped], every event is
                                         rejected; orderly shutdown cannot
                                         be undone by the watchdog.

    E1 flip point: when the E-band work lands, [Watchdog.run] can be
    rewritten as a thin driver that feeds one of {[WatchdogDetectAlive],
    [WatchdogDetectDead]} into the extracted [transition] and asserts
    [Some new_state] on success, replacing the current hand-written
    [if not is_alive → start] conditional entirely. *)

From FidoModels Require Import preamble.

(** * State

    Five phases of the worker thread lifecycle as observed by the watchdog. *)
Inductive State : Type :=
| Running    : State
| Crashed    : State
| Restarting : State
| Hung       : State
| Stopped    : State.

(** * Event

    [WorkerCrash]        — the worker thread exits due to an unhandled
                           exception; [crash_error] is set; provider
                           subprocess survives.
                           [Running | Hung] → [Crashed].
    [WatchdogDetectAlive] — watchdog tick: [is_alive] returns [True];
                            no action taken.
                            [Running] → [Running]; [Hung] → [Hung].
    [WatchdogDetectDead] — watchdog tick: [is_alive] returns [False];
                           watchdog calls [registry.start], rescuing
                           the provider.
                           [Crashed] → [Restarting].
    [StaleTimeout]       — thread has been alive but idle longer than
                           [_STALE_THRESHOLD]; watchdog flags it for
                           display but does NOT restart.
                           [Running] → [Hung].
    [ActivityResume]     — worker reports activity after a stale period.
                           [Hung] → [Running].
    [RestartComplete]    — replacement thread has started and is alive;
                           provider has been handed over.
                           [Restarting] → [Running].
    [RestartCrash]       — replacement thread also dies immediately;
                           escalation: back to [Crashed] for the next
                           watchdog tick to detect.
                           [Restarting] → [Crashed].
    [StopRequest]        — orderly [stop()] called on the thread.
                           [Running | Hung] → [Stopped]. *)
Inductive Event : Type :=
| WorkerCrash         : Event
| WatchdogDetectAlive : Event
| WatchdogDetectDead  : Event
| StaleTimeout        : Event
| ActivityResume      : Event
| RestartComplete     : Event
| RestartCrash        : Event
| StopRequest         : Event.

(** * Transition function

    [transition current event] returns [Some new_state] when [event] is
    valid in [current], or [None] when it is rejected.

    The three rejection groups worth noting:
    - [WatchdogDetectDead] is rejected from [Running] and [Hung] —
      the watchdog can only restart a thread it observed as dead.
    - [WatchdogDetectAlive] from [Hung] stays [Hung], never triggering
      a restart — the live-but-stale thread is left to time out on its
      own.
    - [Stopped] rejects every event — orderly shutdown is terminal. *)
Definition transition (current : State) (event : Event) : option State :=
  match current, event with
  | Running,    WorkerCrash         => Some Crashed
  | Running,    WatchdogDetectAlive => Some Running
  | Running,    StaleTimeout        => Some Hung
  | Running,    StopRequest         => Some Stopped
  | Hung,       WorkerCrash         => Some Crashed
  | Hung,       WatchdogDetectAlive => Some Hung
  | Hung,       ActivityResume      => Some Running
  | Hung,       StopRequest         => Some Stopped
  | Crashed,    WatchdogDetectDead  => Some Restarting
  | Restarting, RestartComplete     => Some Running
  | Restarting, RestartCrash        => Some Crashed
  | _,          _                   => None
  end.

Python File Extraction watchdog_transitions "transition".

(** * Proved invariants

    All lemmas follow by computation ([reflexivity]).  The theorem names
    surface in the runtime oracle: when [Watchdog.run] diverges from
    [transition], the crash message names the violated invariant. *)

(** [alive_never_forcibly_restarted]: the dead-detection event
    [WatchdogDetectDead] is rejected from both live states.  The watchdog
    cannot trigger a restart of an [Running] or [Hung] thread via the
    normal dead-detection path — the only route to [Restarting] is through
    [Crashed].  This is the machine-checked form of the field lesson: a
    forced restart of a live thread raced on the fido lockfile and caused
    a restart loop. *)
Lemma alive_never_forcibly_restarted :
  transition Running WatchdogDetectDead = None /\
  transition Hung    WatchdogDetectDead = None.
Proof.
  split; reflexivity.
Qed.

(** [hung_not_restarted]: [WatchdogDetectAlive] from [Hung] stays [Hung].
    A thread that is alive but stale is never spontaneously restarted by
    the watchdog — it remains [Hung] until either the worker itself reports
    [ActivityResume], crashes ([WorkerCrash]), or is stopped ([StopRequest]).
    The watchdog's role on a live-but-stale thread is observation only. *)
Lemma hung_not_restarted :
  transition Hung WatchdogDetectAlive = Some Hung.
Proof.
  reflexivity.
Qed.

(** [dead_always_restarted]: [WatchdogDetectDead] from [Crashed] always
    yields [Some Restarting].  Restart is a total function from the
    crashed state — the watchdog never silently ignores a dead thread.
    Every detected crash is followed by a restart attempt. *)
Lemma dead_always_restarted :
  transition Crashed WatchdogDetectDead = Some Restarting.
Proof.
  reflexivity.
Qed.

(** [restart_reaches_running]: [RestartComplete] from [Restarting] always
    yields [Some Running].  A successful restart is not partial — the new
    thread enters [Running] with the rescued provider in hand.  The only
    way the state remains non-[Running] after a restart attempt is via
    [RestartCrash] back to [Crashed], which the next watchdog tick will
    detect and handle. *)
Lemma restart_reaches_running :
  transition Restarting RestartComplete = Some Running.
Proof.
  reflexivity.
Qed.

(** [stopped_is_terminal]: every event is rejected from [Stopped].
    An orderly shutdown cannot be undone by the watchdog.  The watchdog's
    supervision contract ends the moment [stop()] is called; the process
    lifecycle (daemon thread dying with the process) is what actually
    prevents the watchdog from restarting a stopped thread in practice,
    and this theorem makes that contract explicit. *)
Lemma stopped_is_terminal :
  transition Stopped WorkerCrash         = None /\
  transition Stopped WatchdogDetectAlive = None /\
  transition Stopped WatchdogDetectDead  = None /\
  transition Stopped StaleTimeout        = None /\
  transition Stopped ActivityResume      = None /\
  transition Stopped RestartComplete     = None /\
  transition Stopped RestartCrash        = None /\
  transition Stopped StopRequest         = None.
Proof.
  repeat split; reflexivity.
Qed.
