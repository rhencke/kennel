(** Task queue, rescope, and preemption coordination model.

    This model states the intended durable coordination contract for one repo's
    ordered task queue.  Tasks remain in an explicit durable order, at most one
    task may be actively leased for execution, and rescope applies explicit
    keep/rewrite/complete decisions to a previously snapped queue.  Today's
    Python still stores this through JSON plus state sidecars; the model is the
    intended SQLite-era contract and first serves as an oracle around the
    handwritten path. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import
  FSets.FMapPositive
  Lists.List
  Numbers.BinNums
  PArith.BinPos
  Strings.String.

Open Scope positive_scope.
Open Scope string_scope.
Import ListNotations.

(** [TaskKind] is the policy-level task kind.

    [TaskAsk] and [TaskDefer] are explicit non-executable kinds in the model,
    even though today's Python still derives them from title prefixes. *)
Inductive TaskKind : Type :=
| TaskCI
| TaskThread
| TaskSpec
| TaskAsk
| TaskDefer.

(** [TaskStatus] is the durable queue-row status. *)
Inductive TaskStatus : Type :=
| StatusPending
| StatusCompleted
| StatusBlocked.

(** [TaskRow] carries the durable task metadata relevant to D3. *)
Record TaskRow : Type := {
  task_title : string;
  task_description : string;
  task_kind : TaskKind;
  task_status : TaskStatus;
  task_source_comment : option positive
}.

(** [ExecutionLease] represents the single task currently being executed for a
    repo.  Ownership is implicit because there is at most one repo worker. *)
Record ExecutionLease : Type := {
  lease_task : positive
}.

(** [RescopeOp] is the explicit normalized rescope decision for one snapped
    task.  The handwritten adapter can derive these from today's omission-based
    provider output before comparing against the model. *)
Inductive RescopeOp : Type :=
| KeepTask : positive -> RescopeOp
| RewriteTask : positive -> string -> string -> RescopeOp
| CompleteTask : positive -> RescopeOp.

(** [task_executable] says whether a task kind may be selected for execution. *)
Definition task_executable (kind : TaskKind) : bool :=
  match kind with
  | TaskCI => true
  | TaskThread => true
  | TaskSpec => true
  | TaskAsk => false
  | TaskDefer => false
  end.

(** [task_row_executable] says whether one row is eligible for execution now. *)
Definition task_row_executable (row : TaskRow) : bool :=
  match task_status row with
  | StatusPending => task_executable (task_kind row)
  | _ => false
  end.

Definition task_kind_is_ci (kind : TaskKind) : bool :=
  match kind with
  | TaskCI => true
  | _ => false
  end.

Definition task_kind_is_non_ci (kind : TaskKind) : bool :=
  negb (task_kind_is_ci kind).

(** [task_preempt_rank] encodes D3's new-task preemption policy. *)
Definition task_preempt_rank (kind : TaskKind) : option nat :=
  match kind with
  | TaskCI => Some O
  | TaskThread => Some (S O)
  | TaskSpec => Some (S (S O))
  | TaskAsk => None
  | TaskDefer => None
  end.

(** [task_requires_abort] says whether a newly enqueued task outranks the
    current active lease and therefore should interrupt it. *)
Definition task_requires_abort
    (new_row current_row : TaskRow) : bool :=
  match task_preempt_rank (task_kind new_row), task_preempt_rank (task_kind current_row) with
  | Some new_rank, Some current_rank => Nat.ltb new_rank current_rank
  | _, _ => false
  end.

Definition positive_eqb (left right : positive) : bool :=
  Pos.eqb left right.

Fixpoint positive_mem (target : positive) (items : list positive) : bool :=
  match items with
  | [] => false
  | item :: rest => if positive_eqb target item then true else positive_mem target rest
  end.

Fixpoint find_comment_duplicate
    (comment : positive)
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : option positive :=
  match order with
  | [] => None
  | task :: rest =>
      match PositiveMap.find task rows with
      | Some row =>
          match task_source_comment row with
          | Some existing => if positive_eqb existing comment then Some task else find_comment_duplicate comment rest rows
          | None => find_comment_duplicate comment rest rows
          end
      | None => find_comment_duplicate comment rest rows
      end
  end.

Fixpoint find_pending_title_duplicate
    (title : string)
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : option positive :=
  match order with
  | [] => None
  | task :: rest =>
      match PositiveMap.find task rows with
      | Some row =>
          match task_status row with
          | StatusPending =>
              if String.eqb (task_title row) title then Some task
              else find_pending_title_duplicate title rest rows
          | _ => find_pending_title_duplicate title rest rows
          end
      | None => find_pending_title_duplicate title rest rows
      end
  end.

(** [enqueue_task] applies the current handwritten dedup rules to a task add.

    Thread/review-originated tasks dedup by source comment id regardless of
    status.  Non-thread tasks dedup by pending title. *)
Definition enqueue_task
    (task : positive)
    (row : TaskRow)
    (order : list positive)
    (rows : PositiveMap.t TaskRow)
    : list positive * PositiveMap.t TaskRow * positive :=
  match task_source_comment row with
  | Some comment =>
      match find_comment_duplicate comment order rows with
      | Some existing => (order, rows, existing)
      | None => (List.app order [task], PositiveMap.add task row rows, task)
      end
  | None =>
      match find_pending_title_duplicate (task_title row) order rows with
      | Some existing => (order, rows, existing)
      | None => (List.app order [task], PositiveMap.add task row rows, task)
      end
  end.

Fixpoint pick_first_ci
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : option positive :=
  match order with
  | [] => None
  | task :: rest =>
      match PositiveMap.find task rows with
      | Some row =>
          if andb (task_row_executable row) (task_kind_is_ci (task_kind row))
          then Some task
          else pick_first_ci rest rows
      | None => pick_first_ci rest rows
      end
  end.

Fixpoint pick_first_non_ci
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : option positive :=
  match order with
  | [] => None
  | task :: rest =>
      match PositiveMap.find task rows with
      | Some row =>
          if andb (task_row_executable row) (task_kind_is_non_ci (task_kind row))
          then Some task
          else pick_first_non_ci rest rows
      | None => pick_first_non_ci rest rows
      end
  end.

(** [pick_next_task] returns the next eligible task id, if any.

    CI tasks win over all other executable kinds; otherwise durable list order
    wins. *)
Definition pick_next_task
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : option positive :=
  match pick_first_ci order rows with
  | Some task => Some task
  | None => pick_first_non_ci order rows
  end.

(** [begin_task] acquires the repo's single execution lease for one task. *)
Definition begin_task
    (task : positive)
    (lease : option ExecutionLease)
    (rows : PositiveMap.t TaskRow) : option ExecutionLease :=
  match lease with
  | Some _ => None
  | None =>
      match PositiveMap.find task rows with
      | Some row => if task_row_executable row then Some {| lease_task := task |} else None
      | None => None
      end
  end.

(** [complete_task] marks one task completed and clears the lease if it owned
    that task. *)
Definition clear_matching_lease
    (task : positive)
    (lease : option ExecutionLease) : option ExecutionLease :=
  match lease with
  | Some active => if positive_eqb (lease_task active) task then None else lease
  | None => None
  end.

Definition complete_task
    (task : positive)
    (lease : option ExecutionLease)
    (rows : PositiveMap.t TaskRow) : option ExecutionLease * PositiveMap.t TaskRow :=
  let lease' := clear_matching_lease task lease in
  match PositiveMap.find task rows with
  | None => (lease', rows)
  | Some row =>
      let row' := {|
        task_title := task_title row;
        task_description := task_description row;
        task_kind := task_kind row;
        task_status := StatusCompleted;
        task_source_comment := task_source_comment row
      |} in
      (lease', PositiveMap.add task row' rows)
  end.

(** [abort_task] clears the active lease without discarding the queue row. *)
Definition abort_task
    (task : positive)
    (lease : option ExecutionLease) : option ExecutionLease :=
  match lease with
  | Some active => if positive_eqb (lease_task active) task then None else lease
  | None => None
  end.

Definition row_with_status (row : TaskRow) (status : TaskStatus) : TaskRow :=
  {|
    task_title := task_title row;
    task_description := task_description row;
    task_kind := task_kind row;
    task_status := status;
    task_source_comment := task_source_comment row
  |}.

Definition unblock_task_row
    (task : positive)
    (row : TaskRow)
    (rows : PositiveMap.t TaskRow) : PositiveMap.t TaskRow :=
  match task_status row with
  | StatusBlocked => PositiveMap.add task (row_with_status row StatusPending) rows
  | _ => rows
  end.

Definition unblock_task_if_present
    (task : positive)
    (rows : PositiveMap.t TaskRow) : PositiveMap.t TaskRow :=
  match PositiveMap.find task rows with
  | Some row => unblock_task_row task row rows
  | None => rows
  end.

Fixpoint unblock_task_rows
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : PositiveMap.t TaskRow :=
  match order with
  | [] => rows
  | task :: rest => unblock_task_rows rest (unblock_task_if_present task rows)
  end.

(** [unblock_tasks] transitions every blocked task back to pending. *)
Definition unblock_tasks
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : PositiveMap.t TaskRow :=
  unblock_task_rows order rows.

Definition rescope_task_id (op : RescopeOp) : positive :=
  match op with
  | KeepTask task => task
  | RewriteTask task _ _ => task
  | CompleteTask task => task
  end.

Fixpoint op_covers_task (task : positive) (ops : list RescopeOp) : bool :=
  match ops with
  | [] => false
  | op :: rest =>
      if positive_eqb (rescope_task_id op) task then true else op_covers_task task rest
  end.

(** [rescope_ops_cover_snapshot] checks that every snapped task has an explicit
    keep/rewrite/complete decision. *)
Fixpoint rescope_ops_cover_snapshot
    (snapshot_order : list positive)
    (ops : list RescopeOp) : bool :=
  match snapshot_order with
  | [] => true
  | task :: rest =>
      if op_covers_task task ops then rescope_ops_cover_snapshot rest ops else false
  end.

Fixpoint apply_rescope_ops
    (ops : list RescopeOp)
    (rows : PositiveMap.t TaskRow)
    (pending_ids : list positive)
    (completed_ids : list positive)
    : PositiveMap.t TaskRow * list positive * list positive :=
  match ops with
  | [] => (rows, pending_ids, completed_ids)
  | op :: rest =>
      let task := rescope_task_id op in
      match PositiveMap.find task rows with
      | None => apply_rescope_ops rest rows pending_ids completed_ids
      | Some row =>
          match op with
          | KeepTask _ =>
              match task_status row with
              | StatusCompleted => apply_rescope_ops rest rows pending_ids completed_ids
              | _ => apply_rescope_ops rest rows (List.app pending_ids [task]) completed_ids
              end
          | RewriteTask _ title description =>
              match task_status row with
              | StatusCompleted => apply_rescope_ops rest rows pending_ids completed_ids
              | _ =>
                  let row' := {|
                    task_title := title;
                    task_description := description;
                    task_kind := task_kind row;
                    task_status := task_status row;
                    task_source_comment := task_source_comment row
                  |} in
                  apply_rescope_ops rest
                    (PositiveMap.add task row' rows)
                    (List.app pending_ids [task])
                    completed_ids
              end
          | CompleteTask _ =>
              match task_status row with
              | StatusCompleted => apply_rescope_ops rest rows pending_ids completed_ids
              | _ =>
                  let row' := {|
                    task_title := task_title row;
                    task_description := task_description row;
                    task_kind := task_kind row;
                    task_status := StatusCompleted;
                    task_source_comment := task_source_comment row
                  |} in
                  apply_rescope_ops rest
                    (PositiveMap.add task row' rows)
                    pending_ids
                    (List.app completed_ids [task])
              end
          end
      end
  end.

Fixpoint completed_tasks_in_order
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : list positive :=
  match order with
  | [] => []
  | task :: rest =>
      match PositiveMap.find task rows with
      | Some row =>
          match task_status row with
          | StatusCompleted => task :: completed_tasks_in_order rest rows
          | _ => completed_tasks_in_order rest rows
          end
      | None => completed_tasks_in_order rest rows
      end
  end.

Fixpoint preserve_newly_added
    (snapshot_order : list positive)
    (current_order : list positive)
    (rows : PositiveMap.t TaskRow) : list positive :=
  match current_order with
  | [] => []
  | task :: rest =>
      let rest' := preserve_newly_added snapshot_order rest rows in
      if positive_mem task snapshot_order then rest'
      else
        match PositiveMap.find task rows with
        | Some row =>
            match task_status row with
            | StatusCompleted => rest'
            | _ => task :: rest'
            end
        | None => rest'
        end
  end.

Definition task_is_ci
    (rows : PositiveMap.t TaskRow)
    (task : positive) : bool :=
  match PositiveMap.find task rows with
  | Some row => task_kind_is_ci (task_kind row)
  | None => false
  end.

Definition task_is_non_ci
    (rows : PositiveMap.t TaskRow)
    (task : positive) : bool :=
  match PositiveMap.find task rows with
  | Some row => task_kind_is_non_ci (task_kind row)
  | None => false
  end.

Fixpoint collect_ci_tasks
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : list positive :=
  match order with
  | [] => []
  | task :: rest =>
      let rest' := collect_ci_tasks rest rows in
      if task_is_ci rows task then task :: rest' else rest'
  end.

Fixpoint collect_non_ci_tasks
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : list positive :=
  match order with
  | [] => []
  | task :: rest =>
      let rest' := collect_non_ci_tasks rest rows in
      if task_is_non_ci rows task then task :: rest' else rest'
  end.

Definition stable_ci_first
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : list positive :=
  let ci := collect_ci_tasks order rows in
  let non_ci := collect_non_ci_tasks order rows in
  List.app ci non_ci.

Definition task_title_changed (before_row after_row : TaskRow) : bool :=
  negb (String.eqb (task_title before_row) (task_title after_row)).

Definition task_description_changed (before_row after_row : TaskRow) : bool :=
  negb (String.eqb (task_description before_row) (task_description after_row)).

Definition task_metadata_changed (before_row after_row : TaskRow) : bool :=
  if task_title_changed before_row after_row then
    true
  else
    task_description_changed before_row after_row.

(** [apply_rescope] applies explicit keep/rewrite/complete decisions to a
    snapped queue while preserving post-snapshot additions and completed rows. *)
Definition apply_rescope
    (snapshot_order : list positive)
    (current_order : list positive)
    (rows : PositiveMap.t TaskRow)
    (ops : list RescopeOp)
    : list positive * PositiveMap.t TaskRow :=
  let '(rows', pending_ids, newly_completed) := apply_rescope_ops ops rows [] [] in
  let completed_existing := completed_tasks_in_order current_order rows in
  let newly_added := preserve_newly_added snapshot_order current_order rows' in
  let pending_order := stable_ci_first pending_ids rows' in
  (List.app pending_order
      (List.app completed_existing (List.app newly_completed newly_added)), rows').

(** [rescope_affects_active_task] says whether the active lease must abort after
    rescope because its task was completed or rewritten. *)
Definition rescope_affects_active_task
    (lease : option ExecutionLease)
    (rows_before rows_after : PositiveMap.t TaskRow) : bool :=
  match lease with
  | None => false
  | Some active =>
      match PositiveMap.find (lease_task active) rows_before,
            PositiveMap.find (lease_task active) rows_after with
      | Some before_row, Some after_row =>
          match task_status after_row with
          | StatusCompleted => true
          | _ => task_metadata_changed before_row after_row
          end
      | Some _, None => true
      | _, _ => false
      end
  end.

(** [should_abort_for_new_task] lifts the preemption rule to a durable active
    lease plus queue rows. *)
Definition should_abort_for_new_task
    (new_task : positive)
    (lease : option ExecutionLease)
    (rows : PositiveMap.t TaskRow) : bool :=
  match lease with
  | None => false
  | Some active =>
      if positive_eqb new_task (lease_task active) then false
      else
        match PositiveMap.find new_task rows, PositiveMap.find (lease_task active) rows with
        | Some new_row, Some current_row => task_requires_abort new_row current_row
        | _, _ => false
        end
  end.

Python File Extraction task_queue_rescope
  "task_executable task_row_executable enqueue_task pick_next_task begin_task complete_task abort_task unblock_tasks rescope_ops_cover_snapshot apply_rescope rescope_affects_active_task should_abort_for_new_task".
