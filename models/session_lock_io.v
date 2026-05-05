(** Session-lock IO substrate — subprocess lifecycle + coupling with the lock FSM.

    [models/session_lock.v] proved that [ForceRelease] is accepted in
    every FSM state and lands in [Free].  That is the *coordination*
    layer — it answers "given the watchdog fired the event, the lock
    advances".  It does *not* answer "given the watchdog fired the
    event, the wedged holder thread will eventually escape its parked
    IO call".

    The wedge in [#1377] depended on a chain that runs *outside* the
    coordination FSM:

      1. Watchdog calls [_proc.kill()].
      2. OS kills the subprocess; stdout closes.
      3. Holder's [select] returns ready with EOF on the closed pipe.
      4. [iter_events] readline returns ``""``; the EOF branch breaks
         the loop.
      5. [consume_until_result] returns the (empty) result.
      6. [with self:] runs [__exit__]; the [_evicted_tids] guard
         skips [_fsm_release].

    Steps 1-3 are the IO substrate: subprocess lifecycle.  This model
    formalises that substrate as its own small FSM, then couples it
    with [session_lock.v] so the headline runtime invariant can be
    stated as a single theorem rather than an unmodeled chain.

    The model also distinguishes two runtime paths the watchdog can
    take when a worker holds the lock:

    - [WatchdogPreempt] — cooperative preemption.  Webhook fires
      [_fire_worker_cancel], the worker drains its turn cleanly, the
      lock transfers via [Preempt].  The subprocess stays alive so
      the new holder can use it without a respawn.
    - [WatchdogEvict]   — forcible eviction.  Watchdog fires
      [ForceRelease] on the lock and [IssueKill] on the subprocess.
      Lock advances to [Free], subprocess advances to [Killed].
      The wedged holder's [select] eventually returns EOF when the
      OS closes stdout ([OsCloseStdout]).

    Pinning down [preempt_does_not_kill_subprocess] vs
    [evict_kills_subprocess] at the model layer is what "honors
    preemption on the worker thread" means formally — preemption is
    a separate coupled event that does not touch the IO substrate.

    This is the [missing-Rocq-IO] piece flagged in cluster R of
    [models/BUG_MINED_INVARIANTS.md].  Reference model only — no
    [Python Extraction] directive: the OS provides the actual
    transitions, so there is no Python target to oracle against.  The
    purpose is to *state* the invariants formally so future code can
    cite them rather than restate them in prose. *)

From FidoModels Require Import preamble.
From FidoModels Require Import session_lock.

(** * Subprocess lifecycle

    The minimal three-state lifecycle that captures the wedge ⇒
    recovery dynamics.  Real subprocesses have many more transient
    states (zombie reaping, signal queueing, file-descriptor
    teardown), but for the coordination property we care about, three
    states suffice:

    [Spawned]  — subprocess is alive, stdout is open and the holder
                 thread can park in [select] on it.
    [Killed]   — [_proc.kill()] has been issued, but the OS has not
                 yet propagated the close to stdout.  The holder is
                 still parked; it will see EOF on the next OS tick.
    [Reaped]   — stdout has been closed by the OS.  The next
                 [readline] returns ``""``, [iter_events] breaks the
                 loop on EOF, and the holder escapes
                 [consume_until_result]. *)
Inductive ProcState : Type :=
  | Spawned : ProcState
  | Killed  : ProcState
  | Reaped  : ProcState.

(** [IssueKill]   — the watchdog (or self) calls [_proc.kill()].
    [StdoutEof]   — the OS has closed stdout following kill; subsequent
                    [readline] returns ``""``.  In real time this is
                    asynchronous — kill returns immediately, the OS
                    closes the pipe later — so we model it as a separate
                    event rather than folding it into [IssueKill]. *)
Inductive ProcEvent : Type :=
  | IssueKill : ProcEvent
  | StdoutEof : ProcEvent.

Definition proc_step (s : ProcState) (e : ProcEvent) : option ProcState :=
  match s, e with
  | Spawned, IssueKill  => Some Killed
  | Killed,  StdoutEof  => Some Reaped
  | _,       _          => None
  end.

(** ** Subprocess invariants *)

(** [kill_progresses_spawned]: from a running subprocess, [IssueKill]
    always succeeds and lands in [Killed].  No "kill might fail
    silently" case in the substrate model. *)
Lemma kill_progresses_spawned :
  proc_step Spawned IssueKill = Some Killed.
Proof. reflexivity. Qed.

(** [eof_progresses_killed]: once kill has been issued, the OS
    eventually closes stdout — modeled as a deterministic next step. *)
Lemma eof_progresses_killed :
  proc_step Killed StdoutEof = Some Reaped.
Proof. reflexivity. Qed.

(** [kill_eof_chain_terminates]: the two-step chain from [Spawned]
    via [IssueKill] then [StdoutEof] always reaches [Reaped].  This is
    the IO-side liveness analogue of [force_release_to_free]: once the
    runtime issues kill and the OS does its work, the holder's stdout
    *will* close. *)
Theorem kill_eof_chain_terminates :
  exists s1,
    proc_step Spawned IssueKill = Some s1 /\
    proc_step s1     StdoutEof  = Some Reaped.
Proof.
  exists Killed; split; reflexivity.
Qed.

(** [reaped_is_terminal]: no event makes progress from [Reaped].  The
    subprocess has been torn down; the only useful action left is to
    spawn a new one (which is a fresh [Spawned] state, modeled
    elsewhere). *)
Lemma reaped_is_terminal :
  proc_step Reaped IssueKill = None /\
  proc_step Reaped StdoutEof = None.
Proof. split; reflexivity. Qed.

(** [killed_does_not_accept_kill]: re-issuing [IssueKill] from
    [Killed] is a no-op (rejected) — calling [_proc.kill()] twice
    while the first kill is still propagating does not produce a new
    state.  Mirrors the [ProcessLookupError] tolerance in
    [ClaudeSession._on_force_release]. *)
Lemma killed_does_not_accept_kill :
  proc_step Killed IssueKill = None.
Proof. reflexivity. Qed.

(** * Coupled state — session lock × subprocess lifecycle

    The runtime invariant we care about — "watchdog fires
    ForceRelease and kill ⇒ lock reaches Free and holder will
    eventually escape" — lives at the *combined* level.  Capture
    that explicitly. *)

Record CoupledState : Type := {
  lock : State;
  proc : ProcState;
}.

(** [CoupledEvent] enumerates the events that can advance the
    coupled (lock × subprocess) state.

    [WatchdogPreempt] is the *cooperative* preemption path — webhook
    fires [_fire_worker_cancel], the worker drains its turn cleanly,
    and the FSM transfers ownership via the modeled [Preempt] event.
    The subprocess stays alive throughout: only ownership of the
    lock changes hands.  Models the worker-thread-cooperates path.

    [WatchdogEvict] is the *forcible* recovery path — the watchdog
    has detected a wedged holder and fires both [ForceRelease] on the
    lock and [IssueKill] on the subprocess.  The subprocess is torn
    down so the wedged holder's parked [select] returns EOF.  Models
    the worker-thread-doesn't-cooperate path.

    [OsCloseStdout] is the asynchronous follow-up after [IssueKill]
    has been issued: the operating system closes stdout, satisfying
    the holder's [readline → ""] expectation.

    Distinguishing [WatchdogPreempt] from [WatchdogEvict] is what
    "honors preemption on the worker thread" means at the model
    level: preemption transfers ownership without killing the IO
    substrate, while eviction tears the substrate down. *)
Inductive CoupledEvent : Type :=
  | WatchdogPreempt : CoupledEvent  (* cooperative — Preempt only *)
  | WatchdogEvict   : CoupledEvent  (* forcible — ForceRelease + IssueKill *)
  | OsCloseStdout   : CoupledEvent. (* asynchronous OS event *)

Definition coupled_step (s : CoupledState) (e : CoupledEvent) : option CoupledState :=
  match e with
  | WatchdogPreempt =>
      match transition s.(lock) Preempt with
      | Some lock' => Some {| lock := lock'; proc := s.(proc) |}
      | None       => None
      end
  | WatchdogEvict =>
      match transition s.(lock) ForceRelease, proc_step s.(proc) IssueKill with
      | Some lock', Some proc' => Some {| lock := lock'; proc := proc' |}
      | _, _                   => None
      end
  | OsCloseStdout =>
      match proc_step s.(proc) StdoutEof with
      | Some proc' => Some {| lock := s.(lock); proc := proc' |}
      | None       => None
      end
  end.

(** ** Coupled invariants *)

(** [evict_releases_lock]: when the watchdog fires [WatchdogEvict] on
    a state where the subprocess is [Spawned] (the wedge condition),
    the resulting state has [lock = Free] and [proc = Killed].  This
    is the headline composite property: the *modeled* runtime
    invariant for the recovery handshake. *)
Theorem evict_releases_lock :
  forall lock0,
    exists lock',
      coupled_step {| lock := lock0; proc := Spawned |} WatchdogEvict
      = Some {| lock := lock'; proc := Killed |}
      /\ lock' = Free.
Proof.
  intro lock0.
  exists Free.
  unfold coupled_step; cbn.
  rewrite (force_release_to_free lock0); cbn.
  split; reflexivity.
Qed.

(** [evict_then_eof_reaps]: after the watchdog evicts, an
    [OsCloseStdout] event progresses the subprocess to [Reaped] —
    completing the holder-escape chain.  Combined with
    [evict_releases_lock], this proves that the watchdog's eviction
    plus one OS event recovers both the lock (to [Free]) and the
    subprocess (to [Reaped]).  The holder's stdout is now closed,
    [iter_events] sees EOF and breaks, and the wedge is over. *)
Theorem evict_then_eof_reaps :
  forall lock0,
    exists s1 s2,
      coupled_step {| lock := lock0; proc := Spawned |} WatchdogEvict = Some s1
      /\ s1.(lock) = Free
      /\ s1.(proc) = Killed
      /\ coupled_step s1 OsCloseStdout = Some s2
      /\ s2.(lock) = Free
      /\ s2.(proc) = Reaped.
Proof.
  intro lock0.
  exists {| lock := Free; proc := Killed |}.
  exists {| lock := Free; proc := Reaped |}.
  unfold coupled_step; cbn.
  rewrite (force_release_to_free lock0).
  cbn.
  repeat split.
Qed.

(** [evict_lock_is_unconditional_on_lock_state]: even if the watchdog
    evicts from a state where the lock is already [Free] (a race
    against a holder that just released voluntarily), the post-state
    is still consistent: lock stays [Free] (idempotent
    [ForceRelease]) and the subprocess advances to [Killed].  The
    runtime ``force_release`` returns ``False`` from this branch
    ([already-Free no-op]) and skips the subclass kill hook; this
    theorem documents the *modeled* counterpart — that even in the
    race case, the FSM is well-defined. *)
Theorem evict_from_free_is_well_defined :
  exists lock' proc',
    coupled_step {| lock := Free; proc := Spawned |} WatchdogEvict
    = Some {| lock := lock'; proc := proc' |}
    /\ lock' = Free
    /\ proc' = Killed.
Proof.
  exists Free; exists Killed.
  unfold coupled_step; cbn.
  split; [ reflexivity | split; reflexivity ].
Qed.

(** ** Preemption vs eviction — distinguished IO behaviours

    The cooperative-preemption and forcible-eviction paths must
    differ at the substrate level.  Preempt transfers ownership but
    leaves the subprocess running for the new holder to use; Evict
    tears the subprocess down so the wedged holder can escape its
    parked IO.  These two theorems pin down that distinction at the
    model layer.  Together they answer the runtime question:
    "does the watchdog honor preemption on the worker thread?" —
    yes, because preemption is a *separate* coupled event that does
    not touch the subprocess. *)

(** [preempt_does_not_kill_subprocess]: cooperative preemption
    transfers ownership from worker to handler with no IO-side
    effect.  The subprocess stays in the same state so the new
    handler holder can immediately use it without a respawn.  This
    is the property that makes webhook-side preemption efficient:
    the handler does not pay for a fresh subprocess boot. *)
Theorem preempt_does_not_kill_subprocess :
  forall p,
    coupled_step {| lock := OwnedByWorker; proc := p |} WatchdogPreempt
    = Some {| lock := OwnedByHandler; proc := p |}.
Proof.
  intro p.
  unfold coupled_step; cbn.
  reflexivity.
Qed.

(** [evict_kills_subprocess]: forcible eviction does kill the
    subprocess.  Direct contrast to [preempt_does_not_kill_subprocess]:
    evicting a worker-held session lands at [Killed], not at the
    original [proc] state. *)
Theorem evict_kills_subprocess :
  coupled_step {| lock := OwnedByWorker; proc := Spawned |} WatchdogEvict
  = Some {| lock := Free; proc := Killed |}.
Proof.
  unfold coupled_step; cbn.
  reflexivity.
Qed.

(** [preempt_only_valid_from_owned_worker]: the cooperative
    preemption path is rejected from [Free] and from [OwnedByHandler]
    in the coupled model exactly the way it is in [session_lock.v]
    alone.  Confirms the IO substrate did not relax the
    handler-takes-over-only-from-worker invariant. *)
Lemma preempt_only_valid_from_owned_worker :
  forall p,
    coupled_step {| lock := Free;           proc := p |} WatchdogPreempt = None /\
    coupled_step {| lock := OwnedByHandler; proc := p |} WatchdogPreempt = None.
Proof.
  intro p.
  unfold coupled_step; cbn.
  split; reflexivity.
Qed.

(** [preempt_and_evict_distinct_outcomes]: from the worker-held
    state with a live subprocess, the two paths produce visibly
    different coupled states — the model proves the runtime
    distinction directly.  Useful as a sanity property: if these
    two outcomes ever collapse to one, the IO substrate has lost
    the cooperative-vs-forcible distinction. *)
Theorem preempt_and_evict_distinct_outcomes :
  let s := {| lock := OwnedByWorker; proc := Spawned |} in
  coupled_step s WatchdogPreempt <> coupled_step s WatchdogEvict.
Proof.
  unfold coupled_step; cbn.
  discriminate.
Qed.
