(** Harness commit decision logic — the pure function that maps a
    turn-outcome sentinel plus observable git environment to a commit
    result constructor.

    The LLM declares intent via a [TurnOutcome] sentinel; the harness
    stages tracked files, observes the git state, and commits.  This
    model captures the branching decision that determines which
    [CommitResult] constructor the Python harness returns.  Git I/O
    is not modeled — only the decision table.

    Five branches:

    [SkipTaskWithReason reason] → [CommitSkipped reason]
    [StuckOnTask reason]        → [CommitSkipped reason]
    [CommitTaskComplete/InProgress summary], nothing staged
                                → [CommitNothingStaged]
    [CommitTaskComplete/InProgress summary], staged, commit fails
                                → [CommitHookFailure output]
    [CommitTaskComplete/InProgress summary], staged, commit succeeds
                                → [CommitSuccess sha]

    The model is oracle-only: it cross-references [turn_outcome.v] and
    [commit_result.v], so the extracted Python is self-contained (not
    imported by runtime code).  The runtime [HarnessCommitter.commit]
    must produce the same constructor for the same inputs — divergence
    means a bug in the Python. *)

From FidoModels Require Import preamble turn_outcome commit_result.

From Stdlib Require Import Strings.String.
Open Scope string_scope.

(** [GitEnv] captures the observable git environment after staging.
    The harness runs [git add -u] then [git diff --cached --quiet]:

    - [has_staged] is [true] when the diff exits non-zero (changes staged).
    - [commit_ok] is the exit status of [git commit]: [true] for success,
      [false] for hook failure.  Only meaningful when [has_staged = true].
    - [commit_sha] is the HEAD sha after a successful commit.
    - [commit_output] is the combined stdout+stderr on commit failure. *)
Record GitEnv : Type := mkGitEnv {
  has_staged     : bool;
  commit_ok      : bool;
  commit_sha     : string;
  commit_output  : string;
}.

(** [harness_commit_decision] is the pure decision function.  Given a
    [TurnOutcome] and the [GitEnv] observed after staging, it returns
    the [CommitResult] that the harness must produce.

    The [GitEnv] fields are only consulted for commit-bearing outcomes
    (where [outcome_is_commit] is [true]).  For skip/stuck outcomes,
    the environment is irrelevant. *)
Definition harness_commit_decision (o : TurnOutcome) (env : GitEnv) : CommitResult :=
  match o with
  | SkipTaskWithReason reason   => CommitSkipped reason
  | StuckOnTask reason          => CommitSkipped reason
  | CommitTaskComplete _
  | CommitTaskInProgress _      =>
      if negb (has_staged env) then
        CommitNothingStaged
      else if commit_ok env then
        CommitSuccess (commit_sha env)
      else
        CommitHookFailure (commit_output env)
  end.

(* ----------------------------------------------------------------------- *)
(** * Lemmas: decision properties                                           *)
(* ----------------------------------------------------------------------- *)

(** Skip outcomes never touch git — result is always [CommitSkipped]. *)
Lemma skip_always_skipped :
  forall reason env,
    harness_commit_decision (SkipTaskWithReason reason) env = CommitSkipped reason.
Proof. reflexivity. Qed.

(** Stuck outcomes produce the same result as skip. *)
Lemma stuck_always_skipped :
  forall reason env,
    harness_commit_decision (StuckOnTask reason) env = CommitSkipped reason.
Proof. reflexivity. Qed.

(** When nothing is staged, commit outcomes always produce [CommitNothingStaged]. *)
Lemma nothing_staged_implies_nothing_staged_result :
  forall o env,
    outcome_is_commit o = true ->
    has_staged env = false ->
    harness_commit_decision o env = CommitNothingStaged.
Proof.
  intros o env Hcommit Hstaged.
  destruct o; simpl in *; try discriminate;
    unfold harness_commit_decision; rewrite Hstaged; reflexivity.
Qed.

(** When staged and commit succeeds, the result is [CommitSuccess]. *)
Lemma staged_commit_ok_implies_success :
  forall o env,
    outcome_is_commit o = true ->
    has_staged env = true ->
    commit_ok env = true ->
    harness_commit_decision o env = CommitSuccess (commit_sha env).
Proof.
  intros o env Hcommit Hstaged Hok.
  destruct o; simpl in *; try discriminate;
    unfold harness_commit_decision; rewrite Hstaged; simpl; rewrite Hok; reflexivity.
Qed.

(** When staged and commit fails, the result is [CommitHookFailure]. *)
Lemma staged_commit_fail_implies_hook_failure :
  forall o env,
    outcome_is_commit o = true ->
    has_staged env = true ->
    commit_ok env = false ->
    harness_commit_decision o env = CommitHookFailure (commit_output env).
Proof.
  intros o env Hcommit Hstaged Hfail.
  destruct o; simpl in *; try discriminate;
    unfold harness_commit_decision; rewrite Hstaged; simpl; rewrite Hfail; reflexivity.
Qed.

(** The decision for skip/stuck outcomes is independent of the git environment. *)
Lemma skip_stuck_env_independent :
  forall o env1 env2,
    outcome_is_commit o = false ->
    harness_commit_decision o env1 = harness_commit_decision o env2.
Proof.
  intros o env1 env2 H.
  destruct o; simpl in *; try discriminate; reflexivity.
Qed.

(** Non-commit outcomes always produce a skipped result. *)
Lemma non_commit_is_skipped :
  forall o env,
    outcome_is_commit o = false ->
    exists reason, harness_commit_decision o env = CommitSkipped reason.
Proof.
  intros o env H.
  destruct o; simpl in *; try discriminate.
  - exists reason. reflexivity.
  - exists reason. reflexivity.
Qed.

(** Commit outcomes never produce [CommitSkipped]. *)
Lemma commit_outcome_never_skipped :
  forall o env,
    outcome_is_commit o = true ->
    (forall reason, harness_commit_decision o env <> CommitSkipped reason).
Proof.
  intros o env Hcommit reason.
  destruct o; simpl in Hcommit; try discriminate.
  - unfold harness_commit_decision.
    destruct (has_staged env); simpl;
      [ destruct (commit_ok env); discriminate | discriminate ].
  - unfold harness_commit_decision.
    destruct (has_staged env); simpl;
      [ destruct (commit_ok env); discriminate | discriminate ].
Qed.

(* ----------------------------------------------------------------------- *)
(** * Examples: concrete decision table                                      *)
(* ----------------------------------------------------------------------- *)

Definition sample_env_clean : GitEnv := {|
  has_staged    := false;
  commit_ok     := false;
  commit_sha    := "";
  commit_output := "";
|}.

Definition sample_env_staged_ok : GitEnv := {|
  has_staged    := true;
  commit_ok     := true;
  commit_sha    := "abc123def";
  commit_output := "";
|}.

Definition sample_env_staged_fail : GitEnv := {|
  has_staged    := true;
  commit_ok     := false;
  commit_sha    := "";
  commit_output := "ruff check failed";
|}.

Example skip_produces_skipped :
  harness_commit_decision (SkipTaskWithReason "already done") sample_env_staged_ok
    = CommitSkipped "already done".
Proof. reflexivity. Qed.

Example stuck_produces_skipped :
  harness_commit_decision (StuckOnTask "need human input") sample_env_staged_ok
    = CommitSkipped "need human input".
Proof. reflexivity. Qed.

Example complete_clean_produces_nothing_staged :
  harness_commit_decision (CommitTaskComplete "Add feature") sample_env_clean
    = CommitNothingStaged.
Proof. reflexivity. Qed.

Example in_progress_clean_produces_nothing_staged :
  harness_commit_decision (CommitTaskInProgress "wip") sample_env_clean
    = CommitNothingStaged.
Proof. reflexivity. Qed.

Example complete_staged_ok_produces_success :
  harness_commit_decision (CommitTaskComplete "Add feature") sample_env_staged_ok
    = CommitSuccess "abc123def".
Proof. reflexivity. Qed.

Example in_progress_staged_ok_produces_success :
  harness_commit_decision (CommitTaskInProgress "wip") sample_env_staged_ok
    = CommitSuccess "abc123def".
Proof. reflexivity. Qed.

Example complete_staged_fail_produces_hook_failure :
  harness_commit_decision (CommitTaskComplete "Add feature") sample_env_staged_fail
    = CommitHookFailure "ruff check failed".
Proof. reflexivity. Qed.

Example in_progress_staged_fail_produces_hook_failure :
  harness_commit_decision (CommitTaskInProgress "wip") sample_env_staged_fail
    = CommitHookFailure "ruff check failed".
Proof. reflexivity. Qed.

Python Import "fido.rocq.turn_outcome" TurnOutcome.
Python Import "fido.rocq.commit_result" CommitResult.

Python File Extraction harness_commit_decision
  "harness_commit_decision".

(* ----------------------------------------------------------------------- *)
(** * Consumer-side dispatch: CommitAction                                   *)
(* ----------------------------------------------------------------------- *)

(** [CommitAction] names the logical action the sentinel loop takes after
    the harness observes a [CommitResult].  There is a 1:1 correspondence
    between these constructors and the arms of the [match commit_result:]
    block in [Worker._run_sentinel_loop] in [worker.py]:

    [ActionSkip]               ↔ [case CommitSkipped]: complete, advance queue.
    [ActionNudgeNothingStaged] ↔ [case CommitNothingStaged]: write nudge prompt.
    [ActionNudgeHookFailure]   ↔ [case CommitHookFailure]: write hook nudge.
    [ActionPushAndComplete]    ↔ [case CommitSuccess] + [CommitTaskComplete]:
                                  push branch, complete task, advance queue.
    [ActionContinueSession]    ↔ [case CommitSuccess] + [CommitTaskInProgress]:
                                  partial commit accepted, loop for another turn. *)
Inductive CommitAction : Type :=
  | ActionSkip
  | ActionNudgeNothingStaged
  | ActionNudgeHookFailure
  | ActionPushAndComplete
  | ActionContinueSession.

(** [outcome_is_complete] returns [true] only for [CommitTaskComplete].
    Mirrors the [isinstance(outcome, CommitTaskComplete)] guard that
    disambiguates the [CommitSuccess] arm in [worker.py]. *)
Definition outcome_is_complete (o : TurnOutcome) : bool :=
  match o with
  | CommitTaskComplete _ => true
  | _                    => false
  end.

(** [commit_result_action] maps a [(TurnOutcome, CommitResult)] pair to the
    [CommitAction] the sentinel loop must take.

    The [TurnOutcome] is only consulted for [CommitSuccess] — skip/stuck
    outcomes cannot produce [CommitSuccess] (see [commit_outcome_never_skipped]),
    so [outcome_is_complete] is [false] only for [CommitTaskInProgress] in
    well-formed executions. *)
Definition commit_result_action (o : TurnOutcome) (r : CommitResult) : CommitAction :=
  match r with
  | CommitSkipped _     => ActionSkip
  | CommitNothingStaged => ActionNudgeNothingStaged
  | CommitHookFailure _ => ActionNudgeHookFailure
  | CommitSuccess _     =>
      if outcome_is_complete o then ActionPushAndComplete
      else ActionContinueSession
  end.

(* ----------------------------------------------------------------------- *)
(** * Lemmas: action mapping is total and deterministic                      *)
(* ----------------------------------------------------------------------- *)

(** [CommitSkipped] always maps to [ActionSkip], regardless of outcome. *)
Lemma skipped_maps_to_skip :
  forall o reason,
    commit_result_action o (CommitSkipped reason) = ActionSkip.
Proof. reflexivity. Qed.

(** [CommitNothingStaged] always maps to [ActionNudgeNothingStaged]. *)
Lemma nothing_staged_maps_to_nudge :
  forall o,
    commit_result_action o CommitNothingStaged = ActionNudgeNothingStaged.
Proof. reflexivity. Qed.

(** [CommitHookFailure] always maps to [ActionNudgeHookFailure]. *)
Lemma hook_failure_maps_to_nudge :
  forall o output,
    commit_result_action o (CommitHookFailure output) = ActionNudgeHookFailure.
Proof. reflexivity. Qed.

(** [CommitSuccess] with [CommitTaskComplete] maps to [ActionPushAndComplete]. *)
Lemma success_complete_maps_to_push :
  forall summary sha,
    commit_result_action (CommitTaskComplete summary) (CommitSuccess sha)
      = ActionPushAndComplete.
Proof. reflexivity. Qed.

(** [CommitSuccess] with [CommitTaskInProgress] maps to [ActionContinueSession]. *)
Lemma success_in_progress_maps_to_continue :
  forall summary sha,
    commit_result_action (CommitTaskInProgress summary) (CommitSuccess sha)
      = ActionContinueSession.
Proof. reflexivity. Qed.

(** [ActionPushAndComplete] arises exactly when the result is [CommitSuccess]
    and the outcome is [CommitTaskComplete]. *)
Lemma push_complete_iff :
  forall o r,
    commit_result_action o r = ActionPushAndComplete <->
    exists sha, r = CommitSuccess sha /\ outcome_is_complete o = true.
Proof.
  intros o r. split.
  - intros H.
    destruct r as [sha | output | | reason]; simpl in H; try discriminate.
    exists sha. split; [reflexivity|].
    destruct (outcome_is_complete o) eqn:Hoc; [reflexivity | discriminate].
  - intros [sha [Hr Hoc]].
    subst r. unfold commit_result_action. rewrite Hoc. reflexivity.
Qed.

(** [ActionContinueSession] arises exactly when the result is [CommitSuccess]
    and the outcome is not [CommitTaskComplete] — in well-formed executions,
    only [CommitTaskInProgress] can reach this branch. *)
Lemma continue_session_iff :
  forall o r,
    commit_result_action o r = ActionContinueSession <->
    exists sha, r = CommitSuccess sha /\ outcome_is_complete o = false.
Proof.
  intros o r. split.
  - intros H.
    destruct r as [sha | output | | reason]; simpl in H; try discriminate.
    exists sha. split; [reflexivity|].
    destruct (outcome_is_complete o) eqn:Hoc; [discriminate | reflexivity].
  - intros [sha [Hr Hoc]].
    subst r. unfold commit_result_action. rewrite Hoc. reflexivity.
Qed.

(* ----------------------------------------------------------------------- *)
(** * Examples: concrete action table                                         *)
(* ----------------------------------------------------------------------- *)

Example skip_action :
  commit_result_action (SkipTaskWithReason "done") (CommitSkipped "done")
    = ActionSkip.
Proof. reflexivity. Qed.

Example nothing_staged_action :
  commit_result_action (CommitTaskComplete "Add feature") CommitNothingStaged
    = ActionNudgeNothingStaged.
Proof. reflexivity. Qed.

Example hook_failure_action :
  commit_result_action (CommitTaskComplete "Add feature")
    (CommitHookFailure "ruff failed")
    = ActionNudgeHookFailure.
Proof. reflexivity. Qed.

Example push_and_complete_action :
  commit_result_action (CommitTaskComplete "Add feature")
    (CommitSuccess "abc123def")
    = ActionPushAndComplete.
Proof. reflexivity. Qed.

Example continue_session_action :
  commit_result_action (CommitTaskInProgress "wip")
    (CommitSuccess "abc123def")
    = ActionContinueSession.
Proof. reflexivity. Qed.

Python Import "fido.rocq.turn_outcome" TurnOutcome.
Python Import "fido.rocq.commit_result" CommitResult.

Python File Extraction commit_result_action
  "commit_result_action".
