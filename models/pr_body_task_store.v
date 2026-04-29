(** PR-body work-queue projection vs durable task-store model.

    This model states the D10 coordination contract: after any authoritative
    task-store write path terminates, GitHub's PR-body work-queue section is
    the projection of the durable task store.  The intended authority is the
    SQLite-era task queue/store boundary; today's JSON task file and PR body
    markdown are migration surfaces around that contract.

    The model is deliberately structured around [TaskWrite] transitions rather
    than around markdown editing.  A transition may start only from a state
    whose visible PR body already matches the durable store; it then applies
    one durable mutation and returns a state whose PR body is freshly projected
    from that new store.  That captures the bug-mined invariant from cluster E:
    [task_add], [task_complete], and [rescope_tasks] must not expose a
    terminated state where the queue in the PR body disagrees with the durable
    task rows.

    E1 flip point: when the scheduler/reducer boundary becomes authoritative,
    the extracted transition can replace the handwritten sync choreography.
    Before that, the extracted Python should run as an oracle around the
    existing [Tasks.add], [Tasks.complete_with_resolve], [reorder_tasks], and
    [sync_tasks] paths and crash loudly when a path returns with a stale
    PR-body projection. *)

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

Definition model_version : nat := O.

(** [TaskStore] is the durable queue state that future SQLite rows represent.
    [task_store_order] is the stable work-queue order; [task_store_rows] holds
    the row data keyed by durable task id. *)
Record TaskStore : Type := {
  task_store_order : list positive;
  task_store_rows : PositiveMap.t TaskRow
}.

(** [PRBodyStatus] is the visible checkbox state in the PR work queue.  Blocked
    rows are not rendered in today's PR body and therefore have no visible
    status here. *)
Inductive PRBodyStatus : Type :=
| PRPending
| PRCompleted.

(** [PRBodyRow] is a structured version of one markdown work-queue line.  The
    actual markdown formatting belongs to Python; the invariant only needs the
    visible task identity, display metadata, kind, and checkbox state. *)
Record PRBodyRow : Type := {
  pr_body_task : positive;
  pr_body_title : string;
  pr_body_description : string;
  pr_body_kind : TaskKind;
  pr_body_status : PRBodyStatus
}.

Definition projected_row
    (task : positive)
    (row : TaskRow)
    (status : PRBodyStatus) : PRBodyRow := {|
  pr_body_task := task;
  pr_body_title := task_title row;
  pr_body_description := task_description row;
  pr_body_kind := task_kind row;
  pr_body_status := status
|}.

Definition pending_ci_projection
    (task : positive)
    (rows : PositiveMap.t TaskRow) : list PRBodyRow :=
  match PositiveMap.find task rows with
  | Some row =>
      match task_status row with
      | StatusPending =>
          if task_kind_is_ci (task_kind row)
          then [projected_row task row PRPending]
          else []
      | _ => []
      end
  | None => []
  end.

Definition pending_non_ci_projection
    (task : positive)
    (rows : PositiveMap.t TaskRow) : list PRBodyRow :=
  match PositiveMap.find task rows with
  | Some row =>
      match task_status row with
      | StatusPending =>
          if negb (task_kind_is_ci (task_kind row))
          then [projected_row task row PRPending]
          else []
      | _ => []
      end
  | None => []
  end.

Definition completed_projection
    (task : positive)
    (rows : PositiveMap.t TaskRow) : list PRBodyRow :=
  match PositiveMap.find task rows with
  | Some row =>
      match task_status row with
      | StatusCompleted => [projected_row task row PRCompleted]
      | _ => []
      end
  | None => []
  end.

Fixpoint project_pending_ci
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : list PRBodyRow :=
  match order with
  | [] => []
  | task :: rest =>
      List.app (pending_ci_projection task rows) (project_pending_ci rest rows)
  end.

Fixpoint project_pending_non_ci
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : list PRBodyRow :=
  match order with
  | [] => []
  | task :: rest =>
      List.app (pending_non_ci_projection task rows)
        (project_pending_non_ci rest rows)
  end.

Fixpoint project_completed
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : list PRBodyRow :=
  match order with
  | [] => []
  | task :: rest =>
      List.app (completed_projection task rows) (project_completed rest rows)
  end.

(** [project_task_store] is the model of the PR-body work-queue section.

    It mirrors the current rendered order: pending CI rows first, then other
    pending rows in durable order, then completed rows.  Blocked rows are
    durable state but are not visible work-queue lines. *)
Definition project_task_store (store : TaskStore) : list PRBodyRow :=
  let order := task_store_order store in
  let rows := task_store_rows store in
  let pending_ci := project_pending_ci order rows in
  let pending_non_ci := project_pending_non_ci order rows in
  let completed := project_completed order rows in
  List.app pending_ci (List.app pending_non_ci completed).

Definition task_kind_eqb (left right : TaskKind) : bool :=
  match left, right with
  | TaskCI, TaskCI => true
  | TaskThread, TaskThread => true
  | TaskSpec, TaskSpec => true
  | TaskAsk, TaskAsk => true
  | TaskDefer, TaskDefer => true
  | _, _ => false
  end.

Definition pr_body_status_eqb (left right : PRBodyStatus) : bool :=
  match left, right with
  | PRPending, PRPending => true
  | PRCompleted, PRCompleted => true
  | _, _ => false
  end.

Definition pr_body_row_eqb (left right : PRBodyRow) : bool :=
  let same_task := Pos.eqb (pr_body_task left) (pr_body_task right) in
  let same_title := String.eqb (pr_body_title left) (pr_body_title right) in
  let same_description :=
    String.eqb (pr_body_description left) (pr_body_description right) in
  let same_kind := task_kind_eqb (pr_body_kind left) (pr_body_kind right) in
  let same_status :=
    pr_body_status_eqb (pr_body_status left) (pr_body_status right) in
  let same_text := andb same_title same_description in
  let same_metadata := andb same_kind same_status in
  andb same_task (andb same_text same_metadata).

Extract Constant pr_body_row_eqb => "__import__(""operator"").eq".

Fixpoint pr_body_eqb (left right : list PRBodyRow) : bool :=
  match left, right with
  | [], [] => true
  | left_row :: left_rest, right_row :: right_rest =>
      andb (pr_body_row_eqb left_row right_row)
        (pr_body_eqb left_rest right_rest)
  | _, _ => false
  end.

(** [SystemState] carries both durable task state and the PR body visible to
    GitHub readers. *)
Record SystemState : Type := {
  durable_task_store : TaskStore;
  visible_pr_body : list PRBodyRow
}.

Definition pr_body_matches_store (state : SystemState) : Prop :=
  visible_pr_body state = project_task_store (durable_task_store state).

Definition pr_body_matches_store_bool (state : SystemState) : bool :=
  let visible := visible_pr_body state in
  let projected := project_task_store (durable_task_store state) in
  pr_body_eqb visible projected.

Definition synced_state (store : TaskStore) : SystemState := {|
  durable_task_store := store;
  visible_pr_body := project_task_store store
|}.

(** [TaskWrite] names every authoritative durable write covered by D10. *)
Inductive TaskWrite : Type :=
| WriteTaskAdd (task : positive) (row : TaskRow) : TaskWrite
| WriteTaskComplete (task : positive) : TaskWrite
| WriteTaskRescope (snapshot_order : list positive) (ops : list RescopeOp)
    : TaskWrite.

Definition apply_task_write
    (write : TaskWrite)
    (store : TaskStore) : TaskStore :=
  match write with
  | WriteTaskAdd task row =>
      let order := task_store_order store in
      let rows := task_store_rows store in
      let '(order', rows', _) :=
        enqueue_task task row order rows in
      {|
        task_store_order := order';
        task_store_rows := rows'
      |}
  | WriteTaskComplete task =>
      let '(rows', _) := complete_task_visible task (task_store_rows store) in
      {|
        task_store_order := task_store_order store;
        task_store_rows := rows'
      |}
  | WriteTaskRescope snapshot_order ops =>
      let order := task_store_order store in
      let rows := task_store_rows store in
      let '(order', rows') :=
        apply_rescope snapshot_order order rows ops in
      {|
        task_store_order := order';
        task_store_rows := rows'
      |}
  end.

(** [transition] rejects attempts to perform the next durable mutation from an
    already-diverged visible state.  From a synced state, it applies the write
    and immediately returns the freshly synced PR-body projection. *)
Definition transition
    (state : SystemState)
    (write : TaskWrite) : option SystemState :=
  if pr_body_matches_store_bool state then
    Some (synced_state (apply_task_write write (durable_task_store state)))
  else
    None.

(** * Proved invariants *)

Lemma synced_state_matches_store :
  forall store, pr_body_matches_store (synced_state store).
Proof.
  intros store.
  reflexivity.
Qed.

(** [rejects_unsynced_writes]: no durable write is accepted from a state whose
    PR-body projection is already stale.  This is the "before the next durable
    state change" half of D10. *)
Lemma rejects_unsynced_writes :
  forall state write,
    pr_body_matches_store_bool state = false ->
    transition state write = None.
Proof.
  intros state write H.
  unfold transition.
  rewrite H.
  reflexivity.
Qed.

(** [task_add_resyncs_pr_body]: after [task_add] returns, the visible PR body is
    projected from the new durable store. *)
Lemma task_add_resyncs_pr_body :
  forall state task row,
    pr_body_matches_store_bool state = true ->
    exists next,
      transition state (WriteTaskAdd task row) = Some next /\
      pr_body_matches_store next.
Proof.
  intros state task row H.
  unfold transition.
  rewrite H.
  exists (synced_state
    (apply_task_write (WriteTaskAdd task row) (durable_task_store state))).
  split.
  - reflexivity.
  - apply synced_state_matches_store.
Qed.

(** [task_complete_resyncs_pr_body]: after [task_complete] returns, the visible
    PR body is projected from the new durable store. *)
Lemma task_complete_resyncs_pr_body :
  forall state task,
    pr_body_matches_store_bool state = true ->
    exists next,
      transition state (WriteTaskComplete task) = Some next /\
      pr_body_matches_store next.
Proof.
  intros state task H.
  unfold transition.
  rewrite H.
  exists (synced_state
    (apply_task_write (WriteTaskComplete task) (durable_task_store state))).
  split.
  - reflexivity.
  - apply synced_state_matches_store.
Qed.

(** [rescope_tasks_resyncs_pr_body]: after [rescope_tasks] returns, the visible
    PR body is projected from the new durable store. *)
Lemma rescope_tasks_resyncs_pr_body :
  forall state snapshot_order ops,
    pr_body_matches_store_bool state = true ->
    exists next,
      transition state (WriteTaskRescope snapshot_order ops) = Some next /\
      pr_body_matches_store next.
Proof.
  intros state snapshot_order ops H.
  unfold transition.
  rewrite H.
  exists (synced_state
    (apply_task_write
      (WriteTaskRescope snapshot_order ops)
      (durable_task_store state))).
  split.
  - reflexivity.
  - apply synced_state_matches_store.
Qed.

Python File Extraction pr_body_task_store
  "model_version project_task_store pr_body_matches_store_bool synced_state apply_task_write transition".
