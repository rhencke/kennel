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

Python File Extraction turn_outcome
  "outcome_summary outcome_is_commit outcome_is_terminal".
