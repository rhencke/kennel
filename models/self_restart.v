(** Self-restart topology FSM: update exit, runner sync, worker teardown,
    and process replacement.

    Models the coordination contract inside [FidoHandler._self_restart] and
    the supervisor loop in [./fido up] that catches exit code 75.  The FSM
    tracks a single restart episode — from the webhook trigger that fires the
    attempt through to either process exit or graceful abort on sync failure.

    [Running]        — Fido process operating normally; no restart in progress.
                       The initial state on startup.  [_self_restart] has not
                       yet been entered for the current episode.
    [Syncing]        — trigger detected for our own repo; [_pull_with_backoff]
                       is running.  Workers have not been touched; if sync
                       fails the process will return to normal operation.
    [StoppingWorkers] — runner sync succeeded; workers are being stopped.
                        [stop_and_join(repo)] and [stop_all()] are in progress.
                        Claude subprocesses are still running.
    [KillingChildren] — all workers stopped; tracked claude subprocesses are
                        being SIGTERMed via [_fn_kill_active_children].  Any
                        subprocess not killed here would be reparented to init,
                        keep its stdin open, and keep writing to the workspace
                        after the new container starts (closed #829).
    [Exiting]        — [os._exit(75)] fired; the process is replacing itself.
                       The supervisor catches exit code 75, syncs the runner
                       clone, rebuilds the Docker image, and starts the next
                       container.  This state is terminal: the process dies
                       immediately.
    [Aborted]        — [_pull_with_backoff] exhausted all retries; Fido logs
                       an error and returns without touching workers.  The
                       process keeps running its current code version.  This
                       state is terminal for the restart episode: the abort
                       is logged, the method returns, and a fresh episode
                       begins only when the next matching webhook arrives.

    Five events name the observable transitions.

    [TriggerRestart]  — PR merged or push to the default branch detected for
                        the repo whose git remote matches [_get_self_repo].
                        [Running] → [Syncing].
    [SyncOk]          — [_pull_with_backoff] returned [True]; runner clone is
                        on [origin/main].
                        [Syncing] → [StoppingWorkers].
    [SyncFail]        — [_pull_with_backoff] returned [False]; all retries
                        exhausted or budget exceeded.
                        [Syncing] → [Aborted].
    [WorkersStopped]  — [registry.stop_and_join] and [registry.stop_all]
                        completed; every per-repo worker thread has exited.
                        [StoppingWorkers] → [KillingChildren].
    [ChildrenKilled]  — [_fn_kill_active_children] completed; every tracked
                        claude subprocess has been SIGTERMed.
                        [KillingChildren] → [Exiting].

    Five proved invariants capture the core guarantees:

      [sync_before_teardown]              — [SyncFail] from [Syncing] yields
                                           [Aborted], not [StoppingWorkers].
                                           Workers are never stopped when the
                                           runner sync fails — the process
                                           continues on its current code
                                           version with all workers intact.
      [abort_is_terminal]                 — every event is rejected from
                                           [Aborted].  A failed sync ends the
                                           restart episode; no teardown step
                                           can be performed after an abort.
      [workers_before_children]           — [ChildrenKilled] is rejected from
                                           [StoppingWorkers].  Claude
                                           subprocesses are only killed after
                                           all workers have exited, preventing
                                           the subprocess-orphan scenario from
                                           #829.
      [exit_requires_full_teardown]       — [Exit75] is rejected from every
                                           state except [KillingChildren].
                                           The process may only exit after
                                           sync succeeded, workers stopped,
                                           and children were killed.
      [exiting_is_terminal]               — every event is rejected from
                                           [Exiting].  Once [os._exit(75)] is
                                           called there is no subsequent state
                                           in this process.

    E1 flip point: when the E-band work lands, [FidoHandler._self_restart] can
    be rewritten as a thin driver that feeds events into the extracted
    [transition] and asserts [Some next_state] at each step, replacing the
    current hand-written sequential guard structure.  The oracle assertions
    become the control flow: [SyncFail → None on WorkersStopped] enforces the
    "sync before teardown" contract at the type level rather than by code
    ordering. *)

From FidoModels Require Import preamble.

(** * State

    Six phases of a self-restart episode as seen by [_self_restart]. *)
Inductive State : Type :=
| Running         : State
| Syncing         : State
| StoppingWorkers : State
| KillingChildren : State
| Exiting         : State
| Aborted         : State.

(** * Event

    [TriggerRestart] — PR merged or push to default branch for Fido's own
                       repo; [_self_restart] has been entered.
                       [Running] → [Syncing].
    [SyncOk]         — [_pull_with_backoff] returned [True]; runner clone
                       is now at [origin/main].
                       [Syncing] → [StoppingWorkers].
    [SyncFail]       — [_pull_with_backoff] returned [False]; all retries
                       exhausted or budget exceeded.
                       [Syncing] → [Aborted].
    [WorkersStopped] — [registry.stop_and_join] and [registry.stop_all]
                       both returned; every worker thread has exited.
                       [StoppingWorkers] → [KillingChildren].
    [ChildrenKilled] — [_fn_kill_active_children] returned; every tracked
                       claude subprocess has been SIGTERMed.
                       [KillingChildren] → [Exiting]. *)
Inductive Event : Type :=
| TriggerRestart : Event
| SyncOk         : Event
| SyncFail       : Event
| WorkersStopped : Event
| ChildrenKilled : Event.

(** * Transition function

    [transition current event] returns [Some new_state] when [event] is
    valid in [current], or [None] when it is rejected.

    The rejection structure encodes the ordering constraints:
    - [SyncFail] yields [Aborted] from [Syncing], never [StoppingWorkers] —
      workers are untouched when the runner sync fails.
    - [ChildrenKilled] is rejected from [StoppingWorkers] — children can only
      be killed after all workers have stopped.
    - [Exit75] is implicit: the only state from which exiting is valid is
      [KillingChildren] (after [ChildrenKilled] fires); all earlier states
      reject it because the FSM has no separate [Exit75] event — [Exiting]
      is reached by [ChildrenKilled] from [KillingChildren] precisely to
      make this guarantee structural.
    - [Exiting] and [Aborted] reject every event — both are terminal. *)
Definition transition (current : State) (event : Event) : option State :=
  match current, event with
  | Running,         TriggerRestart => Some Syncing
  | Syncing,         SyncOk         => Some StoppingWorkers
  | Syncing,         SyncFail       => Some Aborted
  | StoppingWorkers, WorkersStopped => Some KillingChildren
  | KillingChildren, ChildrenKilled => Some Exiting
  | _,               _              => None
  end.

Python File Extraction self_restart "transition".

(** * Proved invariants

    All lemmas follow by computation ([reflexivity]).  The theorem names
    surface in the runtime oracle: when [_self_restart] diverges from
    [transition], the crash message names the violated invariant. *)

(** [sync_before_teardown]: [SyncFail] from [Syncing] yields [Aborted].
    Workers are never stopped when the runner sync fails — the process
    continues on its current code version with all workers intact.  This
    is the machine-checked form of the comment in [_self_restart]: "Sync
    runner BEFORE tearing down the worker.  If the sync fails we log and
    return without touching the running workers." *)
Lemma sync_before_teardown :
  transition Syncing SyncFail = Some Aborted.
Proof.
  reflexivity.
Qed.

(** [abort_is_terminal]: every event is rejected from [Aborted].  A failed
    sync ends the restart episode — no teardown step (stopping workers,
    killing children, or exiting) is accepted after an abort.  The process
    keeps running its current code version and the [_self_restart] method
    returns normally; a fresh episode begins only when the next matching
    webhook arrives. *)
Lemma abort_is_terminal :
  transition Aborted TriggerRestart = None /\
  transition Aborted SyncOk         = None /\
  transition Aborted SyncFail       = None /\
  transition Aborted WorkersStopped = None /\
  transition Aborted ChildrenKilled = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [workers_before_children]: [ChildrenKilled] is rejected from
    [StoppingWorkers].  Claude subprocesses are only killed after all
    worker threads have exited — never before.  This is the machine-checked
    form of the ordering in [_self_restart]: [stop_and_join] and [stop_all]
    are called before [_fn_kill_active_children].  Skipping the
    [WorkersStopped] step and jumping straight to [ChildrenKilled] is
    structurally impossible in the FSM.

    The underlying operational risk: a subprocess killed while its worker
    thread is still alive causes the thread to crash mid-turn and leave
    tasks in an indeterminate state.  The ordering guarantee prevents
    that window. *)
Lemma workers_before_children :
  transition StoppingWorkers ChildrenKilled = None.
Proof.
  reflexivity.
Qed.

(** [exit_requires_full_teardown]: [ChildrenKilled] is only accepted from
    [KillingChildren], and that state is only reachable via [SyncOk] →
    [StoppingWorkers] → [WorkersStopped].  This lemma checks the direct
    rejection cases: [ChildrenKilled] is refused from every state except
    [KillingChildren], making it impossible to reach [Exiting] without
    passing through the full sync → stop → kill sequence. *)
Lemma exit_requires_full_teardown :
  transition Running         ChildrenKilled = None /\
  transition Syncing         ChildrenKilled = None /\
  transition StoppingWorkers ChildrenKilled = None /\
  transition Aborted         ChildrenKilled = None /\
  transition Exiting         ChildrenKilled = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [exiting_is_terminal]: every event is rejected from [Exiting].  Once
    [os._exit(75)] is called the process terminates immediately; there is
    no subsequent coordination state.  The supervisor catches the exit code,
    rebuilds the image, and starts a fresh container — that new container
    begins its own lifecycle from [Running], independent of this FSM. *)
Lemma exiting_is_terminal :
  transition Exiting TriggerRestart = None /\
  transition Exiting SyncOk         = None /\
  transition Exiting SyncFail       = None /\
  transition Exiting WorkersStopped = None /\
  transition Exiting ChildrenKilled = None.
Proof.
  repeat split; reflexivity.
Qed.
