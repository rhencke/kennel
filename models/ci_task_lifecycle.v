(** CI-task lifecycle coordination model.

    This model states the D12 contract for required-check failures.  CI state
    is durable and keyed by the required check name (represented here by a
    stable positive key supplied by the Python adapter).  For each required
    check, the store keeps the latest failing snapshot, at most one live CI
    task, the retry count, and the lifecycle phase.

    Today's Python still detects check-run failures through webhook actions
    and uses the handwritten task queue.  The intended boundary is stricter:
    a CI failure enters as a typed command, the reducer updates the per-check
    durable state before acting, and outbox/task effects happen after that
    durable transition.  While D12 is in oracle mode, Python should translate
    the current path into this model and fail closed if it would enqueue a
    second live task for the same required check or pick lower-priority work
    ahead of pending CI.

    E1 flip point: once the scheduler/reducer boundary is authoritative, the
    extracted transition from this model becomes the implementation for CI
    failure admission, retry/give-up policy, and resolution.  Python should
    only translate GitHub check events and worker outcomes into modeled
    inputs, commit the returned store/task state, then emit GitHub comments,
    worker wakeups, and task execution effects after the commit. *)

From FidoModels Require Import preamble task_queue_rescope.

From Stdlib Require Import
  FSets.FMapPositive
  Lists.List
  Numbers.BinNums
  PArith.BinPos
  Strings.String.

Open Scope positive_scope.
Open Scope string_scope.
Import ListNotations.

(** [RequiredCheck] is the durable key for one protected required-check name.
    The Python adapter maps the human check name to this stable key; the
    snapshot still carries the display string for task text and diagnostics. *)
Definition RequiredCheck := positive.

(** [CIConclusion] is the failing subset of GitHub check conclusions that
    cross the webhook-command boundary. *)
Inductive CIConclusion : Type :=
| CIConclusionFailure
| CIConclusionTimedOut.

(** [CIFailureSnapshot] is the latest failing observation for one required
    check.  Repeated failures for the same check replace this record instead
    of creating another live command. *)
Record CIFailureSnapshot : Type := {
  ci_run : positive;
  ci_check_name : string;
  ci_conclusion : CIConclusion
}.

(** [CIPhase] describes the durable lifecycle for one required check. *)
Inductive CIPhase : Type :=
| CINoFailure
| CIFailing
| CIFixing
| CIGivenUp
| CIResolved.

(** [CIRow] is the durable state for one required check. *)
Record CIRow : Type := {
  ci_snapshot : option CIFailureSnapshot;
  ci_phase : CIPhase;
  ci_task : option positive;
  ci_attempts : nat
}.

Definition CIStore := PositiveMap.t CIRow.

(** Keep the retry ceiling as policy data in the model. *)
Definition ci_max_attempts : nat := S (S (S O)).

Definition ci_task_title (snapshot : CIFailureSnapshot) : string :=
  ci_check_name snapshot.

Definition ci_task_row (snapshot : CIFailureSnapshot) : TaskRow := {|
  title := ci_task_title snapshot;
  description := "";
  kind := TaskCI;
  status := StatusPending;
  source_comment := None
|}.

Definition ci_live_task (row : CIRow) : option positive :=
  match ci_phase row, ci_task row with
  | CIFailing, Some task => Some task
  | CIFixing, Some task => Some task
  | _, _ => None
  end.

Definition ci_update_latest
    (snapshot : CIFailureSnapshot)
    (row : CIRow) : CIRow := {|
  ci_snapshot := Some snapshot;
  ci_phase := CIFailing;
  ci_task := ci_task row;
  ci_attempts := ci_attempts row
|}.

Definition ci_new_live_row
    (snapshot : CIFailureSnapshot)
    (task : positive) : CIRow := {|
  ci_snapshot := Some snapshot;
  ci_phase := CIFailing;
  ci_task := Some task;
  ci_attempts := O
|}.

(** [record_ci_failure] is the admission transition for a failing required
    check.  A fresh failure creates one CI task.  A newer failure for a check
    that already has a live CI task updates that check's snapshot in place and
    returns the existing task id, leaving task order and task rows unchanged. *)
Definition record_ci_failure
    (check : RequiredCheck)
    (snapshot : CIFailureSnapshot)
    (new_task : positive)
    (ci_store : CIStore)
    (task_order : list positive)
    (task_rows : PositiveMap.t TaskRow)
    : CIStore * list positive * PositiveMap.t TaskRow * positive :=
  match PositiveMap.find check ci_store with
  | Some row =>
      match ci_live_task row with
      | Some existing_task =>
          let row' := ci_update_latest snapshot row in
          (PositiveMap.add check row' ci_store,
            task_order,
            task_rows,
            existing_task)
      | None =>
          let row' := ci_new_live_row snapshot new_task in
          let '(task_order', task_rows', created_task) :=
            enqueue_task new_task (ci_task_row snapshot) task_order task_rows in
          (PositiveMap.add check row' ci_store,
            task_order',
            task_rows',
            created_task)
      end
  | None =>
      let row := ci_new_live_row snapshot new_task in
      let '(task_order', task_rows', created_task) :=
        enqueue_task new_task (ci_task_row snapshot) task_order task_rows in
      (PositiveMap.add check row ci_store,
        task_order',
        task_rows',
        created_task)
  end.

Definition start_ci_fix
    (check : RequiredCheck)
    (ci_store : CIStore)
    (task_rows : PositiveMap.t TaskRow)
    (lease : option ExecutionLease)
    : CIStore * option ExecutionLease :=
  match PositiveMap.find check ci_store with
  | Some row =>
      match ci_live_task row with
      | Some task =>
          match begin_task task lease task_rows with
          | Some lease' =>
              let row' := {|
                ci_snapshot := ci_snapshot row;
                ci_phase := CIFixing;
                ci_task := Some task;
                ci_attempts := ci_attempts row
              |} in
              (PositiveMap.add check row' ci_store, Some lease')
          | None => (ci_store, lease)
          end
      | None => (ci_store, lease)
      end
  | None => (ci_store, lease)
  end.

Definition ci_attempt_can_retry (attempts : nat) : bool :=
  Nat.ltb attempts ci_max_attempts.

Definition record_ci_attempt_failed
    (check : RequiredCheck)
    (ci_store : CIStore) : CIStore :=
  match PositiveMap.find check ci_store with
  | Some row =>
      match ci_phase row, ci_task row with
      | CIFixing, Some task =>
          let attempts' := S (ci_attempts row) in
          if ci_attempt_can_retry attempts'
          then
            let row' := {|
              ci_snapshot := ci_snapshot row;
              ci_phase := CIFailing;
              ci_task := Some task;
              ci_attempts := attempts'
            |} in
            PositiveMap.add check row' ci_store
          else
            let row' := {|
              ci_snapshot := ci_snapshot row;
              ci_phase := CIGivenUp;
              ci_task := None;
              ci_attempts := attempts'
            |} in
            PositiveMap.add check row' ci_store
      | _, _ => ci_store
      end
  | None => ci_store
  end.

Definition complete_ci_task_if_present
    (task : option positive)
    (task_rows : PositiveMap.t TaskRow)
    (lease : option ExecutionLease)
    : option ExecutionLease * PositiveMap.t TaskRow :=
  match task with
  | Some task_id => complete_task task_id lease task_rows
  | None => (lease, task_rows)
  end.

Definition record_ci_resolved
    (check : RequiredCheck)
    (ci_store : CIStore)
    (task_rows : PositiveMap.t TaskRow)
    (lease : option ExecutionLease)
    : CIStore * PositiveMap.t TaskRow * option ExecutionLease :=
  match PositiveMap.find check ci_store with
  | Some row =>
      let '(lease', task_rows') :=
        complete_ci_task_if_present (ci_task row) task_rows lease in
      let row' := {|
        ci_snapshot := ci_snapshot row;
        ci_phase := CIResolved;
        ci_task := None;
        ci_attempts := O
      |} in
      (PositiveMap.add check row' ci_store, task_rows', lease')
  | None => (ci_store, task_rows, lease)
  end.

Python File Extraction ci_task_lifecycle
  "ci_max_attempts ci_task_title ci_task_row ci_live_task ci_update_latest ci_new_live_row record_ci_failure start_ci_fix ci_attempt_can_retry record_ci_attempt_failed complete_ci_task_if_present record_ci_resolved pick_next_task".

(** Concrete witnesses used by the theorems below. *)
Definition sample_spec_row : TaskRow := {|
  title := "Implement feature";
  description := "";
  kind := TaskSpec;
  status := StatusPending;
  source_comment := None
|}.

Definition sample_snapshot_old : CIFailureSnapshot := {|
  ci_run := 10;
  ci_check_name := "ci / test";
  ci_conclusion := CIConclusionFailure
|}.

Definition sample_snapshot_new : CIFailureSnapshot := {|
  ci_run := 11;
  ci_check_name := "ci / test";
  ci_conclusion := CIConclusionTimedOut
|}.

(** [latest_failure_reuses_live_task] proves the latest-per-required-check
    rule: a new failing snapshot for a check with an existing live task updates
    that row, returns the existing task id, and does not alter task order. *)
Lemma latest_failure_reuses_live_task :
  let row := ci_new_live_row sample_snapshot_old 7 in
  let ci_store := PositiveMap.add 1 row (PositiveMap.empty CIRow) in
  let task_rows := PositiveMap.add 7 (ci_task_row sample_snapshot_old)
      (PositiveMap.empty TaskRow) in
  record_ci_failure 1 sample_snapshot_new 8 ci_store [7] task_rows =
    (PositiveMap.add 1 (ci_update_latest sample_snapshot_new row) ci_store,
      [7],
      task_rows,
      7).
Proof.
  reflexivity.
Qed.

(** [fresh_ci_failure_is_first_pickup] proves that once a CI failure is
    admitted, the existing task picker chooses the CI task ahead of older spec
    work. *)
Lemma fresh_ci_failure_is_first_pickup :
  let task_rows := PositiveMap.add 1 sample_spec_row (PositiveMap.empty TaskRow) in
  let '(_, task_order', task_rows', _) :=
    record_ci_failure 1 sample_snapshot_old 2
      (PositiveMap.empty CIRow) [1] task_rows in
  pick_next_task task_order' task_rows' = Some 2.
Proof.
  reflexivity.
Qed.

(** [third_failed_attempt_gives_up] states the retry ceiling: after the third
    failed fixing attempt, the check leaves the live-task set and enters
    [CIGivenUp]. *)
Lemma third_failed_attempt_gives_up :
  let row := {|
    ci_snapshot := Some sample_snapshot_old;
    ci_phase := CIFixing;
    ci_task := Some 7;
    ci_attempts := S (S O)
  |} in
  let ci_store := PositiveMap.add 1 row (PositiveMap.empty CIRow) in
  let ci_store' := record_ci_attempt_failed 1 ci_store in
  PositiveMap.find 1 ci_store' =
    Some {|
      ci_snapshot := Some sample_snapshot_old;
      ci_phase := CIGivenUp;
      ci_task := None;
      ci_attempts := S (S (S O))
    |}.
Proof.
  reflexivity.
Qed.

(** [successful_resolution_completes_live_task] states that a successful
    required check completes its live CI task and clears the check's live-task
    pointer. *)
Lemma successful_resolution_completes_live_task :
  let row := ci_new_live_row sample_snapshot_old 7 in
  let ci_store := PositiveMap.add 1 row (PositiveMap.empty CIRow) in
  let task_rows := PositiveMap.add 7 (ci_task_row sample_snapshot_old)
      (PositiveMap.empty TaskRow) in
  let '(ci_store', task_rows', lease') :=
    record_ci_resolved 1 ci_store task_rows (Some {| lease_task := 7 |}) in
  PositiveMap.find 1 ci_store' =
    Some {|
      ci_snapshot := Some sample_snapshot_old;
      ci_phase := CIResolved;
      ci_task := None;
      ci_attempts := O
    |} /\
  lease' = None /\
  (match PositiveMap.find 7 task_rows' with
  | Some row' =>
      match status row' with
      | StatusCompleted => true
      | _ => false
      end
  | None => false
  end) = true.
Proof.
  repeat split; reflexivity.
Qed.
