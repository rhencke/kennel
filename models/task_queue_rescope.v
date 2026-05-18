(** Task queue, rescope, and preemption coordination model.

    This model states the intended durable coordination contract for one repo's
    ordered task queue.  Tasks remain in an explicit durable order, at most one
    task may be actively leased for execution, and rescope applies explicit
    keep/rewrite/complete decisions to a previously snapped queue.  Today's
    Python still stores this through JSON plus state sidecars; the model is the
    intended SQLite-era contract and first serves as an oracle around the
    handwritten path.

    E1 flip point: while D11 is in oracle mode, Python translates the current
    omission-based Opus output into [RescopeRelease] rows, calls
    [apply_batched_rescope], and fails closed if the handwritten reorder result
    diverges.  When the E-band scheduler/reducer boundary becomes authoritative,
    the extracted transition becomes the durable rescope implementation itself:
    Python should only translate inbound ACT/DO releases into the modeled input,
    commit the returned order/rows, then run outbox effects after that commit. *)

From FidoModels Require Import preamble.

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

(** [TaskRow] carries the durable task metadata relevant to D3.

    [source_comment] is the primary anchor (#1714 made it mutable).
    [lineage_comments] is the ordered set of comments that contributed to
    this task — a superset that always includes [source_comment] when set.
    The plumbing for [lineage_comments] arrives ahead of the merge
    operation that will write to it (#1717 split: this leaf adds the
    field; the next leaf adds the [MergeTasks] reducer and the
    lineage-preservation theorem).  Today the existing reducer ops
    (KeepTask / RewriteTask / RewriteAnchor / CompleteTask) preserve
    [lineage_comments] unchanged. *)
Record TaskRow : Type := {
  title : string;
  description : string;
  kind : TaskKind;
  status : TaskStatus;
  source_comment : option positive;
  lineage_comments : list positive
}.

(** [ExecutionLease] represents the single task currently being executed for a
    repo.  Ownership is implicit because there is at most one repo worker. *)
Record ExecutionLease : Type := {
  lease_task : positive
}.

(** [RescopeOp] is the explicit normalized rescope decision for one snapped
    task.  The handwritten adapter can derive these from today's omission-based
    provider output before comparing against the model.

    D11 treats the task id as the only immutable identity for an existing
    snapped task — title (#1713) and source-comment anchor (#1714) are both
    mutable metadata.  [RewriteTask] applies title and description.
    [RewriteAnchor] applies a new source-comment anchor; the Python adapter
    is responsible for preserving the previous anchor in the task's
    [lineage_comment_ids] origin metadata so reply/resolve paths can still
    reach earlier commenters.

    [MergeTasks target sources new_title new_description] (#1717) folds
    every [sources] task's [lineage_comments] (and primary [source_comment]
    anchor) into [target]'s [lineage_comments], then rewrites [target]'s
    title and description.  The sources are NOT closed by this op — the
    adapter emits a separate [CompleteTask] per source so the per-task
    coverage invariant ([rescope_ops_cover_snapshot]) still holds.
    [merge_preserves_source_lineage] proves: after this op, every
    source's [lineage_comments] and [source_comment] are present in
    [target]'s [lineage_comments] — no origin is lost.

    [SplitTask source children] (#1718) closes [source] and inserts each
    [SplitChild] as a fresh pending row that inherits the source's
    [source_comment] anchor and [lineage_comments].  The children's
    [task] ids are pre-allocated by the Python adapter (timestamp-
    random, like all new tasks).  [split_preserves_source_lineage]
    proves every child carries the source's lineage_comments + anchor
    — no origin is lost on the inverse operation either. *)
Record SplitChild : Type := {
  child_task : positive;
  child_title : string;
  child_description : string
}.

Inductive RescopeOp : Type :=
| KeepTask (task : positive) : RescopeOp
| RewriteTask (task : positive) (new_title : string) (new_description : string) : RescopeOp
| RewriteAnchor (task : positive) (new_anchor : option positive) : RescopeOp
| MergeTasks (task : positive) (sources : list positive)
    (new_title : string) (new_description : string) : RescopeOp
| SplitTask (task : positive) (children : list SplitChild) : RescopeOp
| CompleteTask (task : positive) : RescopeOp.

(** [RescopeReleaseKind] distinguishes the accumulated worker releases that
    enter D11's batched rescope input.  Both ACT and DO releases are normalized
    into the same explicit [RescopeOp] algebra before the durable transition
    runs; their kind remains part of the input shape so the oracle can model
    the exact worker boundary. *)
Inductive RescopeReleaseKind : Type :=
| ReleaseACT
| ReleaseDO.

Record RescopeRelease : Type := {
  release_kind : RescopeReleaseKind;
  release_decision : RescopeOp
}.

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
  match status row with
  | StatusPending => task_executable (kind row)
  | _ => false
  end.

Definition task_kind_is_ci (kind : TaskKind) : bool :=
  match kind with
  | TaskCI => true
  | _ => false
  end.

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
  match task_preempt_rank (kind new_row), task_preempt_rank (kind current_row) with
  | Some new_rank, Some current_rank => Nat.ltb new_rank current_rank
  | _, _ => false
  end.

Fixpoint positive_mem (target : positive) (items : list positive) : bool :=
  match items with
  | [] => false
  | item :: rest => if Pos.eqb target item then true else positive_mem target rest
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
          match source_comment row with
          | Some existing => if Pos.eqb existing comment then Some task else find_comment_duplicate comment rest rows
          | None => find_comment_duplicate comment rest rows
          end
      | None => find_comment_duplicate comment rest rows
      end
  end.

Definition row_has_pending_title
    (candidate_title : string)
    (row : TaskRow) : bool :=
  match status row with
  | StatusPending => String.eqb (title row) candidate_title
  | _ => false
  end.

Definition task_has_pending_title
    (candidate_title : string)
    (task : positive)
    (rows : PositiveMap.t TaskRow) : bool :=
  match PositiveMap.find task rows with
  | Some row => row_has_pending_title candidate_title row
  | None => false
  end.

Fixpoint find_pending_title_duplicate
    (candidate_title : string)
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : option positive :=
  match order with
  | [] => None
  | task :: rest =>
      if task_has_pending_title candidate_title task rows
      then Some task
      else find_pending_title_duplicate candidate_title rest rows
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
  match source_comment row with
  | Some comment =>
      match find_comment_duplicate comment order rows with
      | Some existing => (order, rows, existing)
      | None => (List.app order [task], PositiveMap.add task row rows, task)
      end
  | None =>
      match find_pending_title_duplicate (title row) order rows with
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
          if andb (task_row_executable row) (task_kind_is_ci (kind row))
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
          if andb (task_row_executable row) (negb (task_kind_is_ci (kind row)))
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
  | Some active => if Pos.eqb (lease_task active) task then None else lease
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
        title := title row;
        description := description row;
        kind := kind row;
        status := StatusCompleted;
        source_comment := source_comment row;
        lineage_comments := lineage_comments row
      |} in
      (lease', PositiveMap.add task row' rows)
  end.

(** [abort_task] clears the active lease without discarding the queue row. *)
Definition abort_task
    (task : positive)
    (lease : option ExecutionLease) : option ExecutionLease :=
  match lease with
  | Some active => if Pos.eqb (lease_task active) task then None else lease
  | None => None
  end.

Definition task_visible_after_rescope (row : TaskRow) : bool :=
  match status row with
  | StatusCompleted => false
  | _ => true
  end.

Definition unblock_task_row
    (task : positive)
    (row : TaskRow)
    (rows : PositiveMap.t TaskRow) : PositiveMap.t TaskRow :=
  match status row with
  | StatusBlocked =>
      let row' := {|
        title := title row;
        description := description row;
        kind := kind row;
        status := StatusPending;
        source_comment := source_comment row;
        lineage_comments := lineage_comments row
      |} in
      PositiveMap.add task row' rows
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
  | RewriteAnchor task _ => task
  | MergeTasks task _ _ _ => task
  | SplitTask task _ => task
  | CompleteTask task => task
  end.

Definition release_task_id (release : RescopeRelease) : positive :=
  rescope_task_id (release_decision release).

Fixpoint op_covers_task (task : positive) (ops : list RescopeOp) : bool :=
  match ops with
  | [] => false
  | op :: rest =>
      if Pos.eqb (rescope_task_id op) task then true else op_covers_task task rest
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

Fixpoint release_for_task
    (task : positive)
    (releases : list RescopeRelease) : option RescopeOp :=
  match releases with
  | [] => None
  | release :: rest =>
      if Pos.eqb (release_task_id release) task then
        Some (release_decision release)
      else
        release_for_task task rest
  end.

(** [normalize_rescope_batch] is the D11 confluence boundary.  The worker may
    accumulate ACT/DO releases in any FIFO interleaving, but the durable
    transition consumes them in snapshot order.  Equivalent batches whose
    per-task releases agree therefore feed [apply_rescope] the same op list. *)
Fixpoint normalize_rescope_batch
    (snapshot_order : list positive)
    (releases : list RescopeRelease) : list RescopeOp :=
  match snapshot_order with
  | [] => []
  | task :: rest =>
      let rest' := normalize_rescope_batch rest releases in
      match release_for_task task releases with
      | Some op => op :: rest'
      | None => rest'
      end
  end.

(** Insertion-order set helpers used by [MergeTasks] to fold every source
    task's [lineage_comments] (and primary [source_comment]) into the
    target task's [lineage_comments] without duplicates. *)
Definition append_unique (cid : positive) (acc : list positive) : list positive :=
  if positive_mem cid acc then acc else List.app acc [cid].

Fixpoint append_unique_list (cids acc : list positive) : list positive :=
  match cids with
  | [] => acc
  | cid :: rest => append_unique_list rest (append_unique cid acc)
  end.

Definition append_unique_option
    (cid : option positive) (acc : list positive) : list positive :=
  match cid with
  | Some c => append_unique c acc
  | None => acc
  end.

(** Fold one source row's lineage + primary anchor into [acc]. *)
Definition fold_source_lineage_into
    (rows : PositiveMap.t TaskRow)
    (src : positive)
    (acc : list positive) : list positive :=
  match PositiveMap.find src rows with
  | Some row =>
      let with_lineage := append_unique_list (lineage_comments row) acc in
      append_unique_option (source_comment row) with_lineage
  | None => acc
  end.

Fixpoint collect_source_lineages
    (sources : list positive)
    (rows : PositiveMap.t TaskRow)
    (acc : list positive) : list positive :=
  match sources with
  | [] => acc
  | src :: rest =>
      collect_source_lineages rest rows
        (fold_source_lineage_into rows src acc)
  end.

(** [split_child_row] builds the row for one [SplitChild]: inherits the
    source row's anchor, lineage, and kind; carries the child's title
    and description; status is StatusPending so split children are
    immediately visible to the worker picker. *)
Definition split_child_row
    (source_row : TaskRow) (spec : SplitChild) : TaskRow := {|
  title := child_title spec;
  description := child_description spec;
  kind := kind source_row;
  status := StatusPending;
  source_comment := source_comment source_row;
  lineage_comments := lineage_comments source_row
|}.

(** [add_split_kids] folds every [SplitChild] into [rows] /
    [pending_ids].  Each child row inherits the source's anchor and
    lineage by construction (see [split_child_row]). *)
Fixpoint add_split_kids
    (children : list SplitChild)
    (source_row : TaskRow)
    (rows : PositiveMap.t TaskRow)
    (pending_ids : list positive)
    : PositiveMap.t TaskRow * list positive :=
  match children with
  | [] => (rows, pending_ids)
  | spec :: rest =>
      let child_row := split_child_row source_row spec in
      let rows' := PositiveMap.add (child_task spec) child_row rows in
      let pending' := List.app pending_ids [child_task spec] in
      add_split_kids rest source_row rows' pending'
  end.

Definition apply_rescope_op
    (op : RescopeOp)
    (task : positive)
    (row : TaskRow)
    (rows : PositiveMap.t TaskRow)
    (pending_ids completed_ids : list positive)
    : PositiveMap.t TaskRow * list positive * list positive :=
  if task_visible_after_rescope row then
    match op with
    | KeepTask _ =>
        (rows, List.app pending_ids [task], completed_ids)
    | RewriteTask _ new_title new_description =>
        let row' := {|
          title := new_title;
          description := new_description;
          kind := kind row;
          status := status row;
          source_comment := source_comment row;
          lineage_comments := lineage_comments row
        |} in
        (PositiveMap.add task row' rows,
          List.app pending_ids [task],
          completed_ids)
    | RewriteAnchor _ new_anchor =>
        (* Re-anchoring extends lineage_comments with the new anchor so
           the previous primary commenter (already in lineage) and the
           newly-targeted commenter both stay reachable from this row.
           Mirrors the materializer's previous _reanchored_thread logic
           but moves the rule into the model so lineage flows through
           apply_rescope_op exclusively (#1717). *)
        let extended := append_unique_option new_anchor (lineage_comments row) in
        let row' := {|
          title := title row;
          description := description row;
          kind := kind row;
          status := status row;
          source_comment := new_anchor;
          lineage_comments := extended
        |} in
        (PositiveMap.add task row' rows,
          List.app pending_ids [task],
          completed_ids)
    | MergeTasks _ sources new_title new_description =>
        let merged := collect_source_lineages sources rows (lineage_comments row) in
        let row' := {|
          title := new_title;
          description := new_description;
          kind := kind row;
          status := status row;
          source_comment := source_comment row;
          lineage_comments := merged
        |} in
        (PositiveMap.add task row' rows,
          List.app pending_ids [task],
          completed_ids)
    | SplitTask _ children =>
        let closed := {|
          title := title row;
          description := description row;
          kind := kind row;
          status := StatusCompleted;
          source_comment := source_comment row;
          lineage_comments := lineage_comments row
        |} in
        let rows_with_source := PositiveMap.add task closed rows in
        let '(rows', pending') :=
          add_split_kids children row rows_with_source pending_ids in
        (rows', pending', List.app completed_ids [task])
    | CompleteTask _ =>
        let row' := {|
          title := title row;
          description := description row;
          kind := kind row;
          status := StatusCompleted;
          source_comment := source_comment row;
          lineage_comments := lineage_comments row
        |} in
        (PositiveMap.add task row' rows,
          pending_ids,
          List.app completed_ids [task])
    end
  else
    (rows, pending_ids, completed_ids).

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
          let '(rows', pending_ids', completed_ids') :=
            apply_rescope_op op task row rows pending_ids completed_ids in
          apply_rescope_ops rest rows' pending_ids' completed_ids'
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
          match status row with
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
            match status row with
            | StatusCompleted => rest'
            | _ => task :: rest'
            end
        | None => rest'
        end
  end.

Definition task_matches_ci_filter
    (include_ci : bool)
    (rows : PositiveMap.t TaskRow)
    (task : positive) : bool :=
  match PositiveMap.find task rows with
  | Some row =>
      if include_ci
      then task_kind_is_ci (kind row)
      else negb (task_kind_is_ci (kind row))
  | None => false
  end.

Fixpoint collect_tasks
    (include_ci : bool)
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : list positive :=
  match order with
  | [] => []
  | task :: rest =>
      let rest' := collect_tasks include_ci rest rows in
      if task_matches_ci_filter include_ci rows task
      then task :: rest'
      else rest'
  end.

Definition stable_ci_first
    (order : list positive)
    (rows : PositiveMap.t TaskRow) : list positive :=
  let ci := collect_tasks true order rows in
  let non_ci := collect_tasks false order rows in
  List.app ci non_ci.

Definition task_title_changed (before_row after_row : TaskRow) : bool :=
  negb (String.eqb (title before_row) (title after_row)).

Definition task_description_changed (before_row after_row : TaskRow) : bool :=
  negb (String.eqb (description before_row) (description after_row)).

Definition task_metadata_changed (before_row after_row : TaskRow) : bool :=
  if task_title_changed before_row after_row then
    true
  else
    task_description_changed before_row after_row.

(** Identity is the durable task id (#1340).  Title (#1713) and source-comment
    anchor (#1714) are both mutable metadata.  The previous
    [task_identity_changed] / [rescope_preserves_task_identity] predicates are
    gone — id is preserved by construction (it's the lookup key), so there is
    no separate identity invariant for the reducer to enforce.  Lineage
    preservation when the anchor changes is the Python adapter's
    responsibility (it merges the previous anchor into [lineage_comment_ids]
    on the task dict, which is metadata outside the modeled [TaskRow]). *)

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

Definition apply_batched_rescope
    (snapshot_order : list positive)
    (current_order : list positive)
    (rows : PositiveMap.t TaskRow)
    (releases : list RescopeRelease)
    : list positive * PositiveMap.t TaskRow :=
  let ops := normalize_rescope_batch snapshot_order releases in
  apply_rescope snapshot_order current_order rows ops.

(** [merge_target_lineage_includes_source] says whether [target_row]'s
    [lineage_comments] (after a [MergeTasks]) contains every entry from a
    given source row's [lineage_comments] plus its [source_comment]. *)
Fixpoint list_subset (xs ys : list positive) : bool :=
  match xs with
  | [] => true
  | x :: rest => andb (positive_mem x ys) (list_subset rest ys)
  end.

Definition source_anchor_in_lineage
    (anchor : option positive) (lineage : list positive) : bool :=
  match anchor with
  | Some c => positive_mem c lineage
  | None => true
  end.

Definition merge_target_lineage_includes_source
    (target_row src_row : TaskRow) : bool :=
  let target_lineage := lineage_comments target_row in
  let src_lineage_ok := list_subset (lineage_comments src_row) target_lineage in
  let src_anchor_ok := source_anchor_in_lineage (source_comment src_row) target_lineage in
  andb src_lineage_ok src_anchor_ok.

Fixpoint merge_preserves_source_lineage
    (sources : list positive)
    (rows_before : PositiveMap.t TaskRow)
    (target_row_after : TaskRow) : bool :=
  match sources with
  | [] => true
  | src :: rest =>
      match PositiveMap.find src rows_before with
      | None => merge_preserves_source_lineage rest rows_before target_row_after
      | Some src_row =>
          if merge_target_lineage_includes_source target_row_after src_row
          then merge_preserves_source_lineage rest rows_before target_row_after
          else false
      end
  end.

(** [optional_anchors_match] — true iff two optional source-comment
    anchors are both [None] or both [Some] with the same id.  Spelled
    as nested single-option matches (rather than a dual-pattern
    [match a, b with ... end]) so the python.ml extractor produces
    flat if/return code instead of the nested-lambda form that PLC3002
    rejects (#1737 tracks the proper extractor fix). *)
Definition optional_anchors_match (a b : option positive) : bool :=
  match a with
  | None =>
      match b with
      | None => true
      | Some _ => false
      end
  | Some x =>
      match b with
      | None => false
      | Some y => Pos.eqb x y
      end
  end.

(** [split_child_inherits_source] says whether one [child_row] (after a
    [SplitTask]) carries the source row's [lineage_comments] verbatim
    and the same primary [source_comment].  The reducer constructs
    children via [split_child_row] which inherits both fields by
    construction, so this predicate is true for every well-formed
    split output and serves as the runtime oracle the Python adapter
    asserts on every emitted SplitTask (#1718).  The intermediate [let]
    bindings keep extracted Python lines under the ruff length limit
    (#1737 tracks the proper extractor-side fix). *)
Definition split_child_inherits_source
    (source_row child_row : TaskRow) : bool :=
  let src_lineage := lineage_comments source_row in
  let child_lineage := lineage_comments child_row in
  let lineage_ok := list_subset src_lineage child_lineage in
  let src_anchor := source_comment source_row in
  let child_anchor := source_comment child_row in
  let anchor_ok := optional_anchors_match src_anchor child_anchor in
  andb lineage_ok anchor_ok.

Fixpoint split_preserves_source_lineage
    (children : list positive)
    (source_row : TaskRow)
    (rows_after : PositiveMap.t TaskRow) : bool :=
  match children with
  | [] => true
  | child :: rest =>
      match PositiveMap.find child rows_after with
      | None => false
      | Some child_row =>
          if split_child_inherits_source source_row child_row
          then split_preserves_source_lineage rest source_row rows_after
          else false
      end
  end.

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
          match status after_row with
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
      if Pos.eqb new_task (lease_task active) then false
      else
        match PositiveMap.find new_task rows, PositiveMap.find (lease_task active) rows with
        | Some new_row, Some current_row => task_requires_abort new_row current_row
        | _, _ => false
        end
  end.

(** [complete_task_visible] marks a task completed and returns its source
    comment id on the first completion.

    This models the key invariant of [Tasks.complete_by_id]: the source
    comment is returned to the caller only on the first status transition from
    a non-completed state.  Subsequent calls on an already-completed task
    return [None] — the caller must not re-resolve the review thread.  The
    lease is not managed here; the worker clears active-task state separately
    after the completion is durable. *)
Definition complete_task_visible
    (task : positive)
    (rows : PositiveMap.t TaskRow) : PositiveMap.t TaskRow * option positive :=
  match PositiveMap.find task rows with
  | None => (rows, None)
  | Some row =>
      match status row with
      | StatusCompleted => (rows, None)
      | _ =>
          let row' := {|
            title := title row;
            description := description row;
            kind := kind row;
            status := StatusCompleted;
            source_comment := source_comment row;
            lineage_comments := lineage_comments row
          |} in
          (PositiveMap.add task row' rows, source_comment row)
      end
  end.

(** [TaskChange] records one change to a thread-originated task detected
    during rescope.

    [TaskCompleted task] means Opus explicitly marked the task as completed
    in its rescope result; the original commenter should be told their request
    is done because it was covered by recent work.

    [TaskCancelled task] means Opus omitted the task entirely from the rescope
    result; the original commenter should be told their task was removed from
    the queue.  This is semantically different from completion: the work was
    not done, it was deprioritised or deemed no longer relevant.

    The distinction matters because the reply body to the commenter differs:
    "completed" (work was done) vs "cancelled" (work was removed).

    [TaskModified task new_title new_description] means the task's title or
    description changed; the commenter should be told of the updated plan. *)
Inductive TaskChange : Type :=
| TaskCompleted (task : positive) : TaskChange
| TaskCancelled (task : positive) : TaskChange
| TaskModified (task : positive) (new_title : string) (new_description : string) : TaskChange.

(** [task_change] computes the change record, if any, for one snapped
    task given the queue state before and after rescope.

    Returns [None] when the task has no source comment, was already completed
    before rescope, or is unchanged. *)
Definition task_change
    (task : positive)
    (rows_before rows_after : PositiveMap.t TaskRow) : option TaskChange :=
  match PositiveMap.find task rows_before with
  | None => None
  | Some before_row =>
      match source_comment before_row with
      | None => None
      | Some _ =>
          match status before_row with
          | StatusCompleted => None
          | _ =>
              match PositiveMap.find task rows_after with
              | None => Some (TaskCancelled task)
              | Some after_row =>
                  match status after_row with
                  | StatusCompleted => Some (TaskCompleted task)
                  | _ =>
                      if task_metadata_changed before_row after_row then
                        Some (TaskModified task
                          (title after_row)
                          (description after_row))
                      else None
                  end
              end
          end
      end
  end.

(** [compute_task_changes] collects change records for all thread-originated
    tasks in the rescope snapshot that were completed, cancelled, or modified.

    Only tasks in [snapshot_order] (those Opus knew about at snap time) are
    checked.  Already-completed tasks and tasks without a source comment are
    skipped, matching [_compute_thread_changes] in [tasks.py]. *)
Fixpoint compute_task_changes
    (snapshot_order : list positive)
    (rows_before rows_after : PositiveMap.t TaskRow) : list TaskChange :=
  match snapshot_order with
  | [] => []
  | task :: rest =>
      let rest' := compute_task_changes rest rows_before rows_after in
      match task_change task rows_before rows_after with
      | None => rest'
      | Some change => change :: rest'
      end
  end.

Definition task_changes_materially_significant
    (changes : list TaskChange) : bool :=
  match changes with
  | [] => false
  | _ => true
  end.

Definition batched_rescope_materially_significant
    (snapshot_order : list positive)
    (current_order : list positive)
    (rows : PositiveMap.t TaskRow)
    (releases : list RescopeRelease) : bool :=
  let '(_, rows_after) :=
    apply_batched_rescope snapshot_order current_order rows releases in
  let changes := compute_task_changes snapshot_order rows rows_after in
  task_changes_materially_significant changes.

(** [remove_from_order] removes all occurrences of [task] from [order].

    At most one occurrence normally exists; the function is total on any list
    so the proof is structural without needing uniqueness. *)
Fixpoint remove_from_order
    (task : positive)
    (order : list positive) : list positive :=
  match order with
  | [] => []
  | t :: rest =>
      let rest' := remove_from_order task rest in
      if Pos.eqb t task then rest' else t :: rest'
  end.

(** [cleanup_aborted_task] removes an aborted task from the queue entirely and
    clears the active lease.

    This models [_cleanup_aborted_task] in [worker.py]: when the abort signal
    fires mid-execution the task is removed from both the durable order and the
    row map — not merely marked completed — and the lease is cleared.  The
    abort signal itself is outside the pure model; the caller signals abort by
    invoking this transition. *)
Definition cleanup_aborted_task
    (task : positive)
    (lease : option ExecutionLease)
    (order : list positive)
    (rows : PositiveMap.t TaskRow)
    : option ExecutionLease * list positive * PositiveMap.t TaskRow :=
  let lease' := clear_matching_lease task lease in
  let order' := remove_from_order task order in
  let rows'  := PositiveMap.remove task rows in
  (lease', order', rows').

(** [task_still_pending] says whether a task remains in PENDING status.

    This models the external-completion guard inside [execute_task]'s resume
    loop in [worker.py]: if the task is no longer PENDING (e.g. it was
    completed externally via ``fido task complete`` while the provider was
    running) the retry loop exits without further provider invocations. *)
Definition task_still_pending
    (task : positive)
    (rows : PositiveMap.t TaskRow) : bool :=
  match PositiveMap.find task rows with
  | Some row =>
      match status row with
      | StatusPending => true
      | _ => false
      end
  | None => false
  end.


(** ** Intents, ops, and reply-back routing (#1804 / INV-F)

    Epic #1798 routes PR-comment intent through rescope rather than
    materializing comments as tasks before rescope sees them.  Opus consumes
    a batch of [IntentRow] entries (raw text + author + arrival index) and
    emits a list of [OpAttribution] entries (each op tagged with the intents
    that drove it).

    Reply-back rule: for each intent [X] with author [A], fire a reply-back
    to [X] iff some op targeting a task [X] contributed to is attributed to
    a *strictly later* intent from a *different* author and that op
    materially affects what gets done.  ``Materially affects'' means
    [EffectChange] (rewrite title/description, retarget anchor) or
    [EffectDrop] (close the task).  [EffectReorganize] (merge into another
    task, split into children) does *not* trigger reply-back — the work
    still gets done, just under a different task id.  [EffectNoChange]
    (keep) likewise never triggers.

    Same-author later changes never notify — the commenter already knows
    they overrode their own earlier ask.  This rule replaces the verdict-
    shape supersedence heuristic and the diff-based disposition classifier
    the earlier iteration of this PR shipped: whether a reply is owed is
    a structural property of ops + author + arrival order, not of the
    verdict-shape Opus assigned.

    Merge sources / split sources are themselves closed by separate
    [CompleteTask] ops the Python adapter emits as part of the merge/split
    sequence.  Those closures are not standalone drops — the adapter tags
    them with [EffectReorganize] when constructing the [OpAttribution]
    list so the model sees one effect per task per op.

    [reply_back_intents] returns the intents that need a reply along with
    the strongest [NotifyKind] (drop > change), in batch arrival order. *)

Record IntentRow : Type := {
  intent_comment_id : positive;
  intent_author : positive;
  intent_index : nat
}.

Inductive OpEffectKind : Type :=
| EffectChange
| EffectDrop
| EffectReorganize
| EffectNoChange.

(** [op_effect] maps a [RescopeOp] to its [OpEffectKind] in the standalone
    case (op not part of a merge/split sequence).  The Python adapter
    overrides [EffectDrop] to [EffectReorganize] when the closed task is
    itself a merge source or split source — that closure is part of the
    reorganize, not a standalone drop. *)
Definition op_effect (op : RescopeOp) : OpEffectKind :=
  match op with
  | KeepTask _ => EffectNoChange
  | RewriteTask _ _ _ => EffectChange
  | RewriteAnchor _ _ => EffectChange
  | MergeTasks _ _ _ _ => EffectReorganize
  | SplitTask _ _ => EffectReorganize
  | CompleteTask _ => EffectDrop
  end.

(** [OpInput] is one parsed rescope op plus the intents that drove it
    — the shape the Python adapter hands over.  ``oi_intents'' is the
    op's ``contributing_intents'' field; the model computes the effect
    kind (with the merge/split override) so all reply-back logic lives
    here, not in the adapter. *)
Record OpInput : Type := {
  oi_op : RescopeOp;
  oi_intents : list positive
}.

(** [OpAttribution] is one op + the intents that drove it + the
    (possibly overridden) effect kind. *)
Record OpAttribution : Type := {
  oa_task : positive;
  oa_effect : OpEffectKind;
  oa_intents : list positive
}.

(** [op_input_task] is the task id this op targets — for [MergeTasks]
    the target, otherwise the standalone task. *)
Definition op_input_task (oi : OpInput) : positive :=
  rescope_task_id (oi_op oi).

(** [op_input_merge_sources] returns the merge source ids when this op
    is a [MergeTasks], else nil.  Used by [reorganize_source_ids] to
    detect [CompleteTask] ops emitted as part of a merge sequence. *)
Definition op_input_merge_sources (oi : OpInput) : list positive :=
  match oi_op oi with
  | MergeTasks _ sources _ _ => sources
  | _ => []
  end.

(** [op_input_split_source] returns the split source id when this op is
    a [SplitTask], else None. *)
Definition op_input_split_source (oi : OpInput) : option positive :=
  match oi_op oi with
  | SplitTask source _ => Some source
  | _ => None
  end.

(** [reorganize_source_ids] returns every task id that is itself a
    merge source or split source in the batch.  Standalone
    [CompleteTask] ops for these tasks are reclassified from
    [EffectDrop] to [EffectReorganize] — the closure is part of the
    merge/split, not a standalone drop. *)
Fixpoint reorganize_source_ids
    (ops : list OpInput) : list positive :=
  match ops with
  | [] => []
  | oi :: rest =>
      let tail := reorganize_source_ids rest in
      let merge_acc := List.app (op_input_merge_sources oi) tail in
      match op_input_split_source oi with
      | None => merge_acc
      | Some src => src :: merge_acc
      end
  end.

Definition op_input_effect
    (oi : OpInput) (reorg_ids : list positive) : OpEffectKind :=
  match op_effect (oi_op oi) with
  | EffectDrop =>
      if positive_mem (op_input_task oi) reorg_ids
      then EffectReorganize
      else EffectDrop
  | other => other
  end.

Definition op_input_to_attribution
    (oi : OpInput) (reorg_ids : list positive) : OpAttribution :=
  {| oa_task := op_input_task oi;
     oa_effect := op_input_effect oi reorg_ids;
     oa_intents := oi_intents oi |}.

Fixpoint op_attributions_in
    (cursor : list OpInput) (reorg_ids : list positive) : list OpAttribution :=
  match cursor with
  | [] => []
  | oi :: rest =>
      let head := op_input_to_attribution oi reorg_ids in
      let tail := op_attributions_in rest reorg_ids in
      head :: tail
  end.

(** [op_attributions] converts a list of [OpInput] entries into the
    [OpAttribution] list [reply_back_intents] consumes, applying the
    merge/split-source reorganize override. *)
Definition op_attributions (ops : list OpInput) : list OpAttribution :=
  op_attributions_in ops (reorganize_source_ids ops).

Inductive NotifyKind : Type :=
| NotifyChanged
| NotifyDropped.

Definition effect_notify_kind (e : OpEffectKind) : option NotifyKind :=
  match e with
  | EffectChange => Some NotifyChanged
  | EffectDrop => Some NotifyDropped
  | EffectReorganize => None
  | EffectNoChange => None
  end.

(** [TaskWithIntents] pairs a result-list task id with its
    ``contributing_intents'' field — the intents the result attributes
    to this task.  ``contributing_intents'' is the cumulative provenance
    across every op that touched the task during this rescope batch and
    every prior batch whose lineage carried through. *)
Record TaskWithIntents : Type := {
  twi_id : positive;
  twi_row : TaskRow;
  twi_contributing : list positive
}.

Fixpoint find_intent_by_id
    (cid : positive) (intents : list IntentRow) : option IntentRow :=
  match intents with
  | [] => None
  | r :: rest =>
      if Pos.eqb (intent_comment_id r) cid then Some r
      else find_intent_by_id cid rest
  end.

Definition intent_later_and_cross_author
    (x other : IntentRow) : bool :=
  if Nat.ltb (intent_index x) (intent_index other)
  then negb (Pos.eqb (intent_author x) (intent_author other))
  else false.

(** [any_later_cross_author] checks whether any intent id in [others]
    refers to an intent strictly later than [x] in arrival order AND
    from a different author. *)
Fixpoint any_later_cross_author
    (x : IntentRow)
    (intents : list IntentRow)
    (others : list positive) : bool :=
  match others with
  | [] => false
  | i :: rest =>
      match find_intent_by_id i intents with
      | None => any_later_cross_author x intents rest
      | Some other =>
          if intent_later_and_cross_author x other
          then true
          else any_later_cross_author x intents rest
      end
  end.

(** [intent_contributing_tasks] returns the result-task ids that list
    [intent] in their ``contributing_intents'' — the tasks [intent]
    shaped during this batch (or carried forward from prior batches). *)
Fixpoint intent_contributing_tasks
    (intent : positive)
    (tasks : list TaskWithIntents) : list positive :=
  match tasks with
  | [] => []
  | t :: rest =>
      let tail := intent_contributing_tasks intent rest in
      if positive_mem intent (twi_contributing t)
      then twi_id t :: tail
      else tail
  end.

(** [stronger_kind] picks the more-significant [NotifyKind] for
    aggregation across multiple matching ops.  [NotifyDropped] dominates
    [NotifyChanged] — losing the task is the more salient outcome. *)
Definition stronger_kind (a b : NotifyKind) : NotifyKind :=
  match a, b with
  | NotifyDropped, _ => NotifyDropped
  | _, NotifyDropped => NotifyDropped
  | _, _ => NotifyChanged
  end.

Definition combine_notify_kinds
    (a b : option NotifyKind) : option NotifyKind :=
  match a, b with
  | None, k => k
  | k, None => k
  | Some k1, Some k2 => Some (stronger_kind k1 k2)
  end.

(** [task_notify_kind_for_intent] returns the strongest [NotifyKind] the
    ops yield for a single task targeted by [task_id] on behalf of
    [intent], applying the later-cross-author filter. *)
Fixpoint task_notify_kind_for_intent
    (task_id : positive)
    (intent : IntentRow)
    (intents : list IntentRow)
    (ops : list OpAttribution) : option NotifyKind :=
  match ops with
  | [] => None
  | o :: rest =>
      let tail := task_notify_kind_for_intent task_id intent intents rest in
      if andb
           (Pos.eqb (oa_task o) task_id)
           (any_later_cross_author intent intents (oa_intents o))
      then combine_notify_kinds (effect_notify_kind (oa_effect o)) tail
      else tail
  end.

(** [any_task_notify_kind] aggregates [task_notify_kind_for_intent]
    across all of an intent's contributing tasks. *)
Fixpoint any_task_notify_kind
    (task_ids : list positive)
    (intent : IntentRow)
    (intents : list IntentRow)
    (ops : list OpAttribution) : option NotifyKind :=
  match task_ids with
  | [] => None
  | t :: rest =>
      let head := task_notify_kind_for_intent t intent intents ops in
      let tail := any_task_notify_kind rest intent intents ops in
      combine_notify_kinds head tail
  end.

(** [intent_notify_kind] is the per-intent decision: ``[Some kind]'' if
    a later cross-author op materially affected one of [intent]'s
    contributing tasks; ``None'' otherwise. *)
Definition intent_notify_kind
    (intent : IntentRow)
    (intents : list IntentRow)
    (tasks : list TaskWithIntents)
    (ops : list OpAttribution) : option NotifyKind :=
  let x_tasks := intent_contributing_tasks (intent_comment_id intent) tasks in
  any_task_notify_kind x_tasks intent intents ops.

Definition reply_back_entry
    (intent : IntentRow)
    (intents : list IntentRow)
    (tasks : list TaskWithIntents)
    (ops : list OpAttribution) : option (positive * NotifyKind) :=
  match intent_notify_kind intent intents tasks ops with
  | None => None
  | Some kind => Some (intent_comment_id intent, kind)
  end.

(** [reply_back_intents] walks the batch in arrival order and yields
    the ``(comment_id, NotifyKind)'' pairs that warrant a reply.
    Silent intents are omitted; the Python adapter posts one reply per
    returned entry. *)
Fixpoint reply_back_intents_in
    (cursor intents : list IntentRow)
    (tasks : list TaskWithIntents)
    (ops : list OpAttribution)
    : list (positive * NotifyKind) :=
  match cursor with
  | [] => []
  | r :: rest =>
      let tail := reply_back_intents_in rest intents tasks ops in
      match reply_back_entry r intents tasks ops with
      | None => tail
      | Some entry => entry :: tail
      end
  end.

Definition reply_back_intents
    (intents : list IntentRow)
    (tasks : list TaskWithIntents)
    (ops : list OpAttribution)
    : list (positive * NotifyKind) :=
  reply_back_intents_in intents intents tasks ops.

(** [reply_back_intents_for] is the all-in-one entry point: takes the
    raw parsed op inputs, applies [op_attributions] (with merge/split
    detection), and calls [reply_back_intents].  Python callers go
    through this so no notify-rule logic lives in the adapter. *)
Definition reply_back_intents_for
    (intents : list IntentRow)
    (tasks : list TaskWithIntents)
    (ops : list OpInput)
    : list (positive * NotifyKind) :=
  reply_back_intents intents tasks (op_attributions ops).

Python File Extraction task_queue_rescope
  "task_executable task_row_executable enqueue_task pick_next_task begin_task complete_task abort_task unblock_tasks rescope_ops_cover_snapshot normalize_rescope_batch apply_rescope apply_batched_rescope rescope_affects_active_task should_abort_for_new_task complete_task_visible task_change compute_task_changes task_changes_materially_significant batched_rescope_materially_significant remove_from_order cleanup_aborted_task task_still_pending merge_preserves_source_lineage merge_target_lineage_includes_source split_preserves_source_lineage split_child_inherits_source op_effect op_attributions effect_notify_kind intent_notify_kind reply_back_intents reply_back_intents_for".
