(** Review-thread auto-resolution coordination model.

    This model states the D13 contract for deciding when Fido may resolve a
    GitHub review thread.  Resolution is an outbox effect owned by the worker
    after durable task state has been committed; webhook-side review comments
    are translated into queued or dismissed feedback commands before the
    resolver acts.

    The core safety rule is deliberately small: Fido may resolve a thread only
    when the thread is still open, Fido authored the latest modeled comment,
    and no pending follow-up task remains for any modeled comment in that
    thread.  A concurrent owner/collaborator or bot re-comment therefore keeps
    the thread open until that new input is queued or explicitly dismissed;
    comments from other users are ignored for now.

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
    auto-resolve decision.  The Python adapter maps Fido's own comments to
    [CommentByFido], owner/collaborator comments to [CommentByActionable],
    recognized bot comments to [CommentByBot], and all other comments to
    [CommentIgnored].  Ignored comments are visible in GitHub's thread but do
    not keep the thread open or queue work in D13.  Bots are modeled
    separately so the later DO/DUMP bot-feedback reducer can distinguish them
    from owner/collaborator review input. *)
Inductive ThreadCommentAuthor : Type :=
| CommentByFido
| CommentByActionable
| CommentByBot
| CommentIgnored.

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

(** [BotFeedbackOutcome] is the triage result for recognized bot comments.
    Bots do not share the owner/collaborator review model: DO means Fido
    accepts the suggestion with a reply and queues work, while DUMP means Fido
    replies with the reason for declining and treats that feedback as closed. *)
Inductive BotFeedbackOutcome : Type :=
| BotFeedbackDo
| BotFeedbackDump.

(** [BotFeedbackDecision] is the D13 projection of bot triage onto task and
    thread side effects.  Python still writes the visible reply; the oracle
    states the durable task and close-thread obligations. *)
Inductive BotFeedbackDecision : Type :=
| TakeBotSuggestion
| DumpBotSuggestionAndClose.

Fixpoint thread_comment_ids (comments : list ThreadComment) : list positive :=
  match comments with
  | [] => []
  | comment :: rest => thread_comment_id comment :: thread_comment_ids rest
  end.

Fixpoint modeled_thread_comment_ids
    (comments : list ThreadComment) : list positive :=
  match comments with
  | [] => []
  | comment :: rest =>
      match thread_comment_author comment with
      | CommentIgnored => modeled_thread_comment_ids rest
      | _ => thread_comment_id comment :: modeled_thread_comment_ids rest
      end
  end.

Fixpoint last_modeled_author_from
    (current : option ThreadCommentAuthor)
    (comments : list ThreadComment) : option ThreadCommentAuthor :=
  match comments with
  | [] => current
  | comment :: rest =>
      match thread_comment_author comment with
      | CommentIgnored => last_modeled_author_from current rest
      | author => last_modeled_author_from (Some author) rest
      end
  end.

Definition last_modeled_author
    (comments : list ThreadComment) : option ThreadCommentAuthor :=
  last_modeled_author_from None comments.

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
  match last_modeled_author (review_thread_comments thread) with
  | Some CommentByFido => true
  | _ => false
  end.

(** [should_resolve_thread] is true exactly when the worker may emit a
    [resolveReviewThread] effect for the visible thread and current durable
    task state. *)
Definition should_resolve_thread
    (thread : ReviewThread)
    (tasks : list ThreadTask) : bool :=
  let comment_ids := modeled_thread_comment_ids
    (review_thread_comments thread) in
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

Fixpoint latest_queueable_comment
    (comments : list ThreadComment) : option positive :=
  match comments with
  | [] => None
  | comment :: rest =>
      match latest_queueable_comment rest with
      | Some later => Some later
      | None =>
          match thread_comment_author comment with
          | CommentByActionable | CommentByBot =>
              Some (thread_comment_id comment)
          | CommentByFido | CommentIgnored => None
          end
      end
  end.

(** [resolved_thread_queue_decision] is the race guard for webhook admission.
    If GitHub already reports the thread as resolved, only the latest
    queueable owner/collaborator or bot comment is fresh input that must queue
    work.  Older queueable comments and ignored outsider comments on that
    resolved thread are stale deliveries and may be dismissed. *)
Definition resolved_thread_queue_decision
    (thread : ReviewThread)
    (incoming_comment : positive) : ResolvedThreadQueueDecision :=
  if negb (review_thread_resolved thread) then
    QueueThreadTask
  else
    match latest_queueable_comment (review_thread_comments thread) with
    | Some latest =>
        if Pos.eqb latest incoming_comment
        then QueueThreadTask
        else DismissStaleResolvedThread
    | None => DismissStaleResolvedThread
    end.

Definition bot_feedback_decision
    (outcome : BotFeedbackOutcome) : BotFeedbackDecision :=
  match outcome with
  | BotFeedbackDo => TakeBotSuggestion
  | BotFeedbackDump => DumpBotSuggestionAndClose
  end.

Definition bot_feedback_creates_task
    (outcome : BotFeedbackOutcome) : bool :=
  match bot_feedback_decision outcome with
  | TakeBotSuggestion => true
  | DumpBotSuggestionAndClose => false
  end.

Definition bot_feedback_resolves_thread
    (outcome : BotFeedbackOutcome) : bool :=
  match bot_feedback_decision outcome with
  | TakeBotSuggestion => false
  | DumpBotSuggestionAndClose => true
  end.

Python File Extraction thread_auto_resolve
  "thread_comment_ids modeled_thread_comment_ids last_modeled_author_from last_modeled_author comment_is_pending_task task_blocks_thread_resolution has_pending_thread_task latest_comment_is_fido should_resolve_thread resolution_decision latest_queueable_comment resolved_thread_queue_decision bot_feedback_decision".

(** * Proved invariants *)

Definition sample_human_comment : ThreadComment := {|
  thread_comment_id := 1;
  thread_comment_author := CommentByActionable
|}.

Definition sample_fido_comment : ThreadComment := {|
  thread_comment_id := 2;
  thread_comment_author := CommentByFido
|}.

Definition sample_fresh_human_comment : ThreadComment := {|
  thread_comment_id := 3;
  thread_comment_author := CommentByActionable
|}.

Definition sample_ignored_comment : ThreadComment := {|
  thread_comment_id := 4;
  thread_comment_author := CommentIgnored
|}.

Definition sample_bot_comment : ThreadComment := {|
  thread_comment_id := 5;
  thread_comment_author := CommentByBot
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

Definition sample_thread_ignored_last : ReviewThread := {|
  review_thread_resolved := false;
  review_thread_comments := [
    sample_human_comment;
    sample_fido_comment;
    sample_ignored_comment
  ]
|}.

Definition sample_thread_bot_last : ReviewThread := {|
  review_thread_resolved := false;
  review_thread_comments := [
    sample_human_comment;
    sample_fido_comment;
    sample_bot_comment
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

Definition sample_resolved_thread_ignored_last : ReviewThread := {|
  review_thread_resolved := true;
  review_thread_comments := [
    sample_human_comment;
    sample_fido_comment;
    sample_ignored_comment
  ]
|}.

Definition sample_resolved_thread_bot_last : ReviewThread := {|
  review_thread_resolved := true;
  review_thread_comments := [
    sample_human_comment;
    sample_fido_comment;
    sample_bot_comment
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

(** [ignored_comment_after_fido_does_not_block_resolve]: comments from outside
    the owner/collaborator/bot set are ignored by the D13 resolver. *)
Lemma ignored_comment_after_fido_does_not_block_resolve :
  resolution_decision sample_thread_ignored_last [] = ResolveReviewThread.
Proof.
  reflexivity.
Qed.

(** [bot_comment_after_fido_blocks_resolve]: bot comments are separate from
    owner/collaborator comments, but still modeled thread input. *)
Lemma bot_comment_after_fido_blocks_resolve :
  resolution_decision sample_thread_bot_last [] = KeepReviewThreadOpen.
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

(** [ignored_resolved_thread_delivery_dismisses]: an ignored outsider comment
    on an already resolved thread does not queue work. *)
Lemma ignored_resolved_thread_delivery_dismisses :
  resolved_thread_queue_decision sample_resolved_thread_ignored_last 4 =
    DismissStaleResolvedThread.
Proof.
  reflexivity.
Qed.

(** [bot_resolved_thread_delivery_queues]: bot feedback is not collapsed into
    owner/collaborator feedback, but it is still fresh queueable input in D13. *)
Lemma bot_resolved_thread_delivery_queues :
  resolved_thread_queue_decision sample_resolved_thread_bot_last 5 =
    QueueThreadTask.
Proof.
  reflexivity.
Qed.

(** [bot_do_takes_suggestion]: DO feedback from a bot queues work and leaves
    the thread open until that work is completed. *)
Lemma bot_do_takes_suggestion :
  bot_feedback_decision BotFeedbackDo = TakeBotSuggestion /\
  bot_feedback_creates_task BotFeedbackDo = true /\
  bot_feedback_resolves_thread BotFeedbackDo = false.
Proof.
  repeat split; reflexivity.
Qed.

(** [bot_dump_closes_feedback]: DUMP feedback from a bot does not queue work;
    Fido's reply explains why and the feedback can be considered closed. *)
Lemma bot_dump_closes_feedback :
  bot_feedback_decision BotFeedbackDump = DumpBotSuggestionAndClose /\
  bot_feedback_creates_task BotFeedbackDump = false /\
  bot_feedback_resolves_thread BotFeedbackDump = true.
Proof.
  repeat split; reflexivity.
Qed.
