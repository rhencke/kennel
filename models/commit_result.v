(** Commit-result types for the harness commit protocol.

    After the harness processes a turn-outcome sentinel and runs
    ``git add -u`` / ``git commit``, the outcome is one of four
    constructors:

    [CommitSuccess]        — a new commit was created.  [sha] is the
                             full hex SHA of the commit.
    [CommitHookFailure]    — the pre-commit hook (or ``git commit``
                             itself) rejected the commit.  [output] is
                             the combined stdout + stderr for the LLM.
    [CommitNothingStaged]  — ``git add -u`` staged nothing (worktree
                             clean or only untracked files remain).
    [CommitSkipped]        — the turn outcome was [SkipTaskWithReason]
                             so no git operations were run.  [reason]
                             is forwarded from the sentinel. *)

From FidoModels Require Import preamble.

From Stdlib Require Import Strings.String.
Open Scope string_scope.

Inductive CommitResult : Type :=
| CommitSuccess       (sha    : string) : CommitResult
| CommitHookFailure   (output : string) : CommitResult
| CommitNothingStaged                   : CommitResult
| CommitSkipped       (reason : string) : CommitResult.

(** [result_is_success] returns [true] when the result represents a
    successfully created commit. *)
Definition result_is_success (r : CommitResult) : bool :=
  match r with
  | CommitSuccess       _ => true
  | CommitHookFailure   _ => false
  | CommitNothingStaged   => false
  | CommitSkipped       _ => false
  end.

(** [result_needs_retry] returns [true] when the result indicates the
    LLM should be nudged to fix something and try again — currently
    only hook failures and nothing-staged. *)
Definition result_needs_retry (r : CommitResult) : bool :=
  match r with
  | CommitSuccess       _ => false
  | CommitHookFailure   _ => true
  | CommitNothingStaged   => true
  | CommitSkipped       _ => false
  end.

(* ----------------------------------------------------------------------- *)
(** * Lemmas: success and retry are mutually exclusive                       *)
(* ----------------------------------------------------------------------- *)

(** A successful commit never needs retry. *)
Lemma success_implies_not_retry :
  forall r, result_is_success r = true -> result_needs_retry r = false.
Proof.
  intros r H. destruct r; simpl in *; reflexivity || discriminate.
Qed.

(** A result that needs retry is never a success. *)
Lemma retry_implies_not_success :
  forall r, result_needs_retry r = true -> result_is_success r = false.
Proof.
  intros r H. destruct r; simpl in *; reflexivity || discriminate.
Qed.

(* ----------------------------------------------------------------------- *)
(** * Examples: each constructor through both predicates                     *)
(* ----------------------------------------------------------------------- *)

Example success_is_success :
  result_is_success (CommitSuccess "abc123") = true.
Proof. reflexivity. Qed.

Example success_not_retry :
  result_needs_retry (CommitSuccess "abc123") = false.
Proof. reflexivity. Qed.

Example hook_failure_not_success :
  result_is_success (CommitHookFailure "ruff failed") = false.
Proof. reflexivity. Qed.

Example hook_failure_needs_retry :
  result_needs_retry (CommitHookFailure "ruff failed") = true.
Proof. reflexivity. Qed.

Example nothing_staged_not_success :
  result_is_success CommitNothingStaged = false.
Proof. reflexivity. Qed.

Example nothing_staged_needs_retry :
  result_needs_retry CommitNothingStaged = true.
Proof. reflexivity. Qed.

Example skipped_not_success :
  result_is_success (CommitSkipped "already done") = false.
Proof. reflexivity. Qed.

Example skipped_not_retry :
  result_needs_retry (CommitSkipped "already done") = false.
Proof. reflexivity. Qed.

Python File Extraction commit_result
  "result_is_success result_needs_retry".

