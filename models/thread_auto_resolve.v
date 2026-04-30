(** Review-thread auto-resolution coordination model.

    This model states the D13 contract for deciding when Fido may resolve a
    GitHub review thread.  Resolution is an outbox effect owned by the worker
    after durable task state has been committed; webhook-side review comments
    are translated into queued or dismissed feedback commands before the
    resolver acts.

    The core safety rule is deliberately small: Fido may resolve a thread only
    when the thread is still open, Fido authored the latest visible comment,
    and no pending follow-up task remains for any comment in that thread.  A
    concurrent human re-comment therefore keeps the thread open until that new
    input is queued or explicitly dismissed.

    E1 flip point: while D13 is in oracle mode, Python should translate the
    current [Tasks.complete_with_resolve], [Worker.resolve_addressed_threads],
    and resolved-thread duplicate suppression paths into these predicates and
    fail closed on divergence.  When the scheduler/reducer boundary becomes
    authoritative, the extracted transition should become the reducer decision:
    first commit the task/feedback command state, then emit
    [resolveReviewThread] only from the returned outbox. *)

From FidoModels Require Import preamble task_queue_rescope.

From Stdlib Require Import
  Lists.List
  Numbers.BinNums
  PArith.BinPos.

Open Scope positive_scope.
Import ListNotations.

(** [ThreadCommentAuthor] is the review-thread actor relevant to the
    auto-resolve decision.  The Python adapter maps all Fido bot logins to
    [CommentByFido] and every human reviewer to [CommentByHuman]. *)
Inductive ThreadCommentAuthor : Type :=
| CommentByFido
| CommentByHuman.

(** [ThreadComment] is one GitHub review-thread comment in chronological
    order.  [thread_comment_id] is the raw GitHub database id used by task
    metadata. *)
Record ThreadComment : Type := {
  thread_comment_id : positive;
  thread_comment_author : ThreadCommentAuthor
}.

(** [ReviewThread] is the visible GitHub thread state needed by D13. *)
Record ReviewThread : Type := {
  review_thread_resolved : bool;
  review_thread_comments : list ThreadComment
}.

(** [ThreadTask] is the durable task-store projection relevant to one review
    comment.  The full task row lives in [task_queue_rescope.v]; D13 only
    needs the source comment id and status. *)
Record ThreadTask : Type := {
  thread_task_comment : positive;
  thread_task_status : TaskStatus
}.

(** [ThreadResolutionDecision] is the outbox decision for the resolver path. *)
Inductive ThreadResolutionDecision : Type :=
| ResolveReviewThread
| KeepReviewThreadOpen.

(** [ResolvedThreadQueueDecision] is the webhook-side admission decision for a
    comment observed on a thread GitHub currently reports as resolved. *)
Inductive ResolvedThreadQueueDecision : Type :=
| QueueThreadTask
| DismissStaleResolvedThread.

Fixpoint thread_comment_ids (comments : list ThreadComment) : list positive :=
  match comments with
  | [] => []
  | comment :: rest => thread_comment_id comment :: thread_comment_ids rest
  end.

Fixpoint last_comment_author_from
    (current : option ThreadCommentAuthor)
    (comments : list ThreadComment) : option ThreadCommentAuthor :=
  match comments with
  | [] => current
  | comment :: rest =>
      last_comment_author_from (Some (thread_comment_author comment)) rest
  end.

Definition last_comment_author
    (comments : list ThreadComment) : option ThreadCommentAuthor :=
  last_comment_author_from None comments.

Definition comment_is_pending_task (task : ThreadTask) : bool :=
  match thread_task_status task with
  | StatusPending => true
  | _ => false
  end.

Definition task_blocks_thread_resolution
    (comment_ids : list positive)
    (task : ThreadTask) : bool :=
  match thread_task_status task with
  | StatusPending => positive_mem (thread_task_comment task) comment_ids
  | _ => false
  end.

Fixpoint has_pending_thread_task
    (comment_ids : list positive)
    (tasks : list ThreadTask) : bool :=
  match tasks with
  | [] => false
  | task :: rest =>
      if task_blocks_thread_resolution comment_ids task
      then true
      else has_pending_thread_task comment_ids rest
  end.

Definition latest_comment_is_fido (thread : ReviewThread) : bool :=
  match last_comment_author (review_thread_comments thread) with
  | Some CommentByFido => true
  | _ => false
  end.

(** [should_resolve_thread] is true exactly when the worker may emit a
    [resolveReviewThread] effect for the visible thread and current durable
    task state. *)
Definition should_resolve_thread
    (thread : ReviewThread)
    (tasks : list ThreadTask) : bool :=
  let comment_ids := thread_comment_ids (review_thread_comments thread) in
  let thread_open := negb (review_thread_resolved thread) in
  let fido_last := latest_comment_is_fido thread in
  let followup_done := negb (has_pending_thread_task comment_ids tasks) in
  andb thread_open (andb fido_last followup_done).

Definition resolution_decision
    (thread : ReviewThread)
    (tasks : list ThreadTask) : ThreadResolutionDecision :=
  if should_resolve_thread thread tasks
  then ResolveReviewThread
  else KeepReviewThreadOpen.

Fixpoint latest_human_comment (comments : list ThreadComment) : option positive :=
  match comments with
  | [] => None
  | comment :: rest =>
      match latest_human_comment rest with
      | Some later => Some later
      | None =>
          match thread_comment_author comment with
          | CommentByHuman => Some (thread_comment_id comment)
          | CommentByFido => None
          end
      end
  end.

(** [resolved_thread_queue_decision] is the race guard for webhook admission.
    If GitHub already reports the thread as resolved, only the latest human
    comment is fresh input that must queue work.  Older human comments on that
    resolved thread are stale duplicate deliveries and may be dismissed. *)
Definition resolved_thread_queue_decision
    (thread : ReviewThread)
    (incoming_comment : positive) : ResolvedThreadQueueDecision :=
  if negb (review_thread_resolved thread) then
    QueueThreadTask
  else
    match latest_human_comment (review_thread_comments thread) with
    | Some latest =>
        if Pos.eqb latest incoming_comment
        then QueueThreadTask
        else DismissStaleResolvedThread
    | None => DismissStaleResolvedThread
    end.

Python File Extraction thread_auto_resolve
  "thread_comment_ids last_comment_author_from last_comment_author comment_is_pending_task task_blocks_thread_resolution has_pending_thread_task latest_comment_is_fido should_resolve_thread resolution_decision latest_human_comment resolved_thread_queue_decision".

(** * Proved invariants *)

Definition sample_human_comment : ThreadComment := {|
  thread_comment_id := 1;
  thread_comment_author := CommentByHuman
|}.

Definition sample_fido_comment : ThreadComment := {|
  thread_comment_id := 2;
  thread_comment_author := CommentByFido
|}.

Definition sample_fresh_human_comment : ThreadComment := {|
  thread_comment_id := 3;
  thread_comment_author := CommentByHuman
|}.

Definition sample_thread_fido_last : ReviewThread := {|
  review_thread_resolved := false;
  review_thread_comments := [sample_human_comment; sample_fido_comment]
|}.

Definition sample_thread_human_last : ReviewThread := {|
  review_thread_resolved := false;
  review_thread_comments := [
    sample_human_comment;
    sample_fido_comment;
    sample_fresh_human_comment
  ]
|}.

Definition sample_resolved_thread : ReviewThread := {|
  review_thread_resolved := true;
  review_thread_comments := [
    sample_human_comment;
    sample_fido_comment;
    sample_fresh_human_comment
  ]
|}.

Definition sample_pending_task : ThreadTask := {|
  thread_task_comment := 1;
  thread_task_status := StatusPending
|}.

Definition sample_completed_task : ThreadTask := {|
  thread_task_comment := 1;
  thread_task_status := StatusCompleted
|}.

(** [resolve_requires_fido_last_comment]: a human re-comment racing after
    Fido's reply keeps the thread open, even when no pending task remains. *)
Lemma resolve_requires_fido_last_comment :
  resolution_decision sample_thread_human_last [] = KeepReviewThreadOpen.
Proof.
  reflexivity.
Qed.

(** [resolve_requires_no_pending_followup]: an unresolved thread where Fido
    posted last still cannot resolve while any same-thread comment has a
    pending durable task. *)
Lemma resolve_requires_no_pending_followup :
  resolution_decision sample_thread_fido_last [sample_pending_task] =
    KeepReviewThreadOpen.
Proof.
  reflexivity.
Qed.

(** [completed_followup_allows_resolve]: once durable follow-up work is
    completed and Fido remains the latest commenter, resolution may fire. *)
Lemma completed_followup_allows_resolve :
  resolution_decision sample_thread_fido_last [sample_completed_task] =
    ResolveReviewThread.
Proof.
  reflexivity.
Qed.

(** [fresh_recomment_on_resolved_thread_queues]: a latest human comment on a
    resolved thread is fresh feedback and must queue work instead of being
    suppressed as stale duplicate delivery. *)
Lemma fresh_recomment_on_resolved_thread_queues :
  resolved_thread_queue_decision sample_resolved_thread 3 = QueueThreadTask.
Proof.
  reflexivity.
Qed.

(** [stale_resolved_thread_delivery_dismisses]: an older human comment on a
    resolved thread is stale after auto-resolution and may be dismissed. *)
Lemma stale_resolved_thread_delivery_dismisses :
  resolved_thread_queue_decision sample_resolved_thread 1 =
    DismissStaleResolvedThread.
Proof.
  reflexivity.
Qed.
