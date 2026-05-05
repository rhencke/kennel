(** Nudge-kind enumeration and selection logic.

    When the harness needs to ask the LLM to try again, there are three
    distinct nudge situations:

    [NudgeMissingSentinel]  — the LLM's output did not end with a valid
                              turn_outcome JSON object.  The harness cannot
                              proceed without a sentinel.
    [NudgeNothingStaged]    — the turn_outcome declared a commit, but
                              [git add -u] staged nothing.  The working
                              tree has no tracked modifications.
    [NudgeHookFailure]      — the pre-commit hook (or [git commit] itself)
                              rejected the commit.  The LLM must fix the
                              issues and emit a new sentinel.

    The nudge-selection function maps a [CommitResult] to an optional
    [NudgeKind].  The [MissingSentinel] case is selected separately —
    it fires before any commit attempt, when sentinel parsing fails.

    Python keeps the string templates and context formatting but delegates
    the "which nudge" decision to the extracted Rocq function. *)

From FidoModels Require Import preamble commit_result.

From Stdlib Require Import Strings.String.
Open Scope string_scope.

Inductive NudgeKind : Type :=
| NudgeMissingSentinel : NudgeKind
| NudgeNothingStaged   : NudgeKind
| NudgeHookFailure     : NudgeKind.

(** [commit_result_nudge] maps a [CommitResult] to the nudge that should
    be sent, or [None] if no nudge is needed (success or skipped results
    do not trigger nudges). *)
Definition commit_result_nudge (r : CommitResult) : option NudgeKind :=
  match r with
  | CommitSuccess _       => None
  | CommitSkipped _       => None
  | CommitNothingStaged   => Some NudgeNothingStaged
  | CommitHookFailure _   => Some NudgeHookFailure
  end.

(* ----------------------------------------------------------------------- *)
(** * Lemmas                                                                *)
(* ----------------------------------------------------------------------- *)

(** Results that need retry always produce a nudge. *)
Lemma retry_implies_nudge :
  forall r, result_needs_retry r = true ->
    exists k, commit_result_nudge r = Some k.
Proof.
  intros r H.
  destruct r; simpl in *; try discriminate.
  - exists NudgeHookFailure. reflexivity.
  - exists NudgeNothingStaged. reflexivity.
Qed.

(** Results that don't need retry produce no nudge. *)
Lemma no_retry_implies_no_nudge :
  forall r, result_needs_retry r = false ->
    commit_result_nudge r = None.
Proof.
  intros r H.
  destruct r; simpl in *; try discriminate; reflexivity.
Qed.

(** Successful commits never produce a nudge. *)
Lemma success_no_nudge :
  forall sha, commit_result_nudge (CommitSuccess sha) = None.
Proof. reflexivity. Qed.

(** Skipped commits never produce a nudge. *)
Lemma skipped_no_nudge :
  forall reason, commit_result_nudge (CommitSkipped reason) = None.
Proof. reflexivity. Qed.

(** Nothing-staged always produces the nothing-staged nudge. *)
Lemma nothing_staged_nudge_kind :
  commit_result_nudge CommitNothingStaged = Some NudgeNothingStaged.
Proof. reflexivity. Qed.

(** Hook failure always produces the hook-failure nudge. *)
Lemma hook_failure_nudge_kind :
  forall output,
    commit_result_nudge (CommitHookFailure output) = Some NudgeHookFailure.
Proof. reflexivity. Qed.

(** [commit_result_nudge] returns [Some _] iff [result_needs_retry] is [true]. *)
Lemma nudge_iff_retry :
  forall r,
    (exists k, commit_result_nudge r = Some k) <-> result_needs_retry r = true.
Proof.
  intros r. split.
  - intros [k H]. destruct r; simpl in *; try discriminate; reflexivity.
  - intros H. destruct r; simpl in *; try discriminate.
    + exists NudgeHookFailure. reflexivity.
    + exists NudgeNothingStaged. reflexivity.
Qed.

(* ----------------------------------------------------------------------- *)
(** * Examples                                                               *)
(* ----------------------------------------------------------------------- *)

Example success_example :
  commit_result_nudge (CommitSuccess "abc123") = None.
Proof. reflexivity. Qed.

Example skipped_example :
  commit_result_nudge (CommitSkipped "already done") = None.
Proof. reflexivity. Qed.

Example nothing_staged_example :
  commit_result_nudge CommitNothingStaged = Some NudgeNothingStaged.
Proof. reflexivity. Qed.

Example hook_failure_example :
  commit_result_nudge (CommitHookFailure "ruff check failed") = Some NudgeHookFailure.
Proof. reflexivity. Qed.

Python File Extraction nudge_kind
  "commit_result_nudge".
