(** Turn-outcome sentinel types for the worker task protocol.

    Every provider turn must end with a JSON sentinel on its final
    non-empty line.  The sentinel declares what the harness should do
    next: stage and commit the working tree then mark the task completed,
    stage and commit but keep the task pending for another turn, skip
    the commit and record a reason, or signal that it is stuck and needs
    human intervention.

    The LLM declares intent; Python acts on it.  Git operations are
    never the LLM's responsibility.

    Four constructors:

    [CommitTaskComplete]   — stage + commit, then mark the task completed.
                             [summary] becomes the git commit message.
    [CommitTaskInProgress]  — stage + commit, but keep the task pending for
                             another turn.  The harness commits partial work
                             so progress is durable, then re-enters the task
                             on the next iteration.
    [SkipTaskWithReason]   — do not commit; record [reason] instead.  If
                             the task is genuinely complete (already covered
                             by a prior commit, consolidated into another
                             task, or proved a no-op), the reason is
                             sufficient grounding for completion.  Otherwise
                             the task remains pending with the reason logged.
    [StuckOnTask]          — the LLM is blocked and cannot make further
                             progress without human guidance.  The harness
                             posts a BLOCKED comment on the PR and parks the
                             task — no further work until the human provides
                             direction (skip, iterate to fix, etc). *)

From FidoModels Require Import preamble.

From Stdlib Require Import Strings.String.
Open Scope string_scope.

Inductive TurnOutcome : Type :=
| CommitTaskComplete   (summary : string) : TurnOutcome
| CommitTaskInProgress (summary : string) : TurnOutcome
| SkipTaskWithReason   (reason  : string) : TurnOutcome
| StuckOnTask          (reason  : string) : TurnOutcome.

(** [outcome_summary] extracts the summary string from commit outcomes.
    Returns the empty string for [SkipTaskWithReason] since skip outcomes
    carry a reason, not a commit message. *)
Definition outcome_summary (o : TurnOutcome) : string :=
  match o with
  | CommitTaskComplete   s => s
  | CommitTaskInProgress s => s
  | SkipTaskWithReason   _ => ""%string
  | StuckOnTask          _ => ""%string
  end.

(** [outcome_is_commit] returns [true] when the outcome calls for a
    stage-and-commit cycle. *)
Definition outcome_is_commit (o : TurnOutcome) : bool :=
  match o with
  | CommitTaskComplete   _ => true
  | CommitTaskInProgress _ => true
  | SkipTaskWithReason   _ => false
  | StuckOnTask          _ => false
  end.

(** [outcome_is_terminal] returns [true] when the outcome marks the task
    as finished — either successfully committed and complete, or skipped
    with a reason that grounds completion. *)
Definition outcome_is_terminal (o : TurnOutcome) : bool :=
  match o with
  | CommitTaskComplete   _ => true
  | CommitTaskInProgress _ => false
  | SkipTaskWithReason   _ => true
  | StuckOnTask          _ => true
  end.

(** [parse_sentinel] models the dispatch logic of the Python
    [parse_turn_outcome] parser.  Given the [turn_outcome] kind string
    and the payload string (the "summary" or "reason" value), it
    returns [Some] of the matching constructor when the kind is
    recognised and the payload is non-empty, or [None] otherwise.

    The Python parser handles JSON decoding, field extraction, and error
    messages — this function captures only the pure dispatch invariant
    that the oracle asserts at runtime. *)
Definition parse_sentinel (kind : string) (payload : string) : option TurnOutcome :=
  if String.eqb payload "" then None
  else if String.eqb kind "commit-task-complete" then
    Some (CommitTaskComplete payload)
  else if String.eqb kind "commit-task-in-progress" then
    Some (CommitTaskInProgress payload)
  else if String.eqb kind "skip-task-with-reason" then
    Some (SkipTaskWithReason payload)
  else if String.eqb kind "stuck-on-task" then
    Some (StuckOnTask payload)
  else None.

(** [parse_sentinel] is total: every recognised kind with a non-empty
    payload produces a result. *)
Lemma parse_sentinel_known_kinds :
  forall payload,
    payload <> ""%string ->
    parse_sentinel "commit-task-complete" payload = Some (CommitTaskComplete payload) /\
    parse_sentinel "commit-task-in-progress" payload = Some (CommitTaskInProgress payload) /\
    parse_sentinel "skip-task-with-reason" payload = Some (SkipTaskWithReason payload) /\
    parse_sentinel "stuck-on-task" payload = Some (StuckOnTask payload).
Proof.
  intros payload Hne.
  unfold parse_sentinel.
  destruct (String.eqb payload "") eqn:Heq.
  - apply String.eqb_eq in Heq. contradiction.
  - repeat split; reflexivity.
Qed.

(** Empty payload always yields [None], regardless of kind. *)
Lemma parse_sentinel_empty_payload :
  forall kind, parse_sentinel kind "" = None.
Proof.
  intros kind. unfold parse_sentinel. simpl. reflexivity.
Qed.

(** When [parse_sentinel] returns [Some o], [outcome_is_commit o] agrees
    with the kind being one of the two commit kinds. *)
Lemma parse_sentinel_commit_iff :
  forall kind payload o,
    parse_sentinel kind payload = Some o ->
    outcome_is_commit o = true <->
    (String.eqb kind "commit-task-complete" = true \/
     String.eqb kind "commit-task-in-progress" = true).
Proof.
  intros kind payload o Hparse.
  unfold parse_sentinel in Hparse.
  destruct (String.eqb payload "") eqn:Hempty; [discriminate|].
  destruct (String.eqb kind "commit-task-complete") eqn:Hctc.
  - injection Hparse as <-. simpl. split; intros; auto.
  - destruct (String.eqb kind "commit-task-in-progress") eqn:Hcip.
    + injection Hparse as <-. simpl. split; intros; auto.
    + destruct (String.eqb kind "skip-task-with-reason") eqn:Hskip.
      * injection Hparse as <-. simpl. split.
        -- intros Habs; discriminate.
        -- intros [H1 | H2]; discriminate.
      * destruct (String.eqb kind "stuck-on-task") eqn:Hstuck.
        -- injection Hparse as <-. simpl. split.
           ++ intros Habs; discriminate.
           ++ intros [H1 | H2]; discriminate.
        -- discriminate.
Qed.

Python File Extraction turn_outcome
  "outcome_summary outcome_is_commit outcome_is_terminal parse_sentinel".
