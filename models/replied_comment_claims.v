(** Replied-comment claim and promise coordination model.

    The raw GitHub comment database id is the claim key.  The model covers all
    actionable PR feedback surfaces: top-level PR comments, review comments,
    file-level comments, and line/range review-thread comments.  One reply
    promise may cover many comment ids, and acknowledging that promise
    completes every covered id. *)

From FidoModels Require Import preamble.

From Stdlib Require Import
  FSets.FMapPositive
  Lists.List
  Numbers.BinNums.

Open Scope positive_scope.
Import ListNotations.

(** [ClaimState] is the durable ownership state for one raw GitHub comment id. *)
Inductive ClaimState : Type :=
| ClaimInProgress
| ClaimCompleted
| ClaimRetryableFailed.

(** [PromiseState] is the durable lifecycle for one outbound reply attempt. *)
Inductive PromiseState : Type :=
| PromisePrepared
| PromisePosted
| PromiseAcked
| PromiseFailed.

(** [RecoveryObservation] is the live GitHub fact observed when replaying or
    reconciling a durable promise after a crash, redelivery, or startup scan. *)
Inductive RecoveryObservation : Type :=
| SeenPromiseMarker
| AnchorDeleted
| WrongPullRequest
| ReplayPosted
| ReplayFailed.

(** [ClaimOwner] records the runtime path that owns an in-progress claim. *)
Inductive ClaimOwner : Type :=
| OwnerWebhook
| OwnerWorker
| OwnerRecovery.

(** [Attempt] is the shared durable metadata carried by both indexes.
    Retry fields are abstract counters/ticks so D1 can state retryability
    without owning the full D22 scheduler. *)
Record Attempt : Type := {
  attempt_owner : ClaimOwner;
  attempt_retry_count : nat;
  attempt_next_retry_after : nat
}.

(** [ClaimRow] is the comment-keyed exclusion index.  It points at the outbound
    promise that owns or completed the raw GitHub comment id. *)
Record ClaimRow : Type := {
  claim_attempt : Attempt;
  claim_state : ClaimState;
  claim_promise : positive
}.

(** [PromiseRow] is the promise-keyed outbound reply attempt.  One promise may
    cover many comment-keyed claims. *)
Record PromiseRow : Type := {
  promise_attempt : Attempt;
  promise_state : PromiseState;
  promise_anchor_comment : positive;
  promise_covered_comments : list positive
}.

(** [ConversationLane] is the batch boundary for visible replies.
    Review comments aggregate per root thread; top-level PR comments aggregate
    in one PR-level lane. *)
Inductive ConversationLane : Type :=
| ReviewThreadLane (thread_id : positive) : ConversationLane
| PullRequestLane (pr_number : positive) : ConversationLane.

(** [ReplyArtifact] is one visible GitHub comment that may discharge many
    promises in the same lane. *)
Record ReplyArtifact : Type := {
  artifact_comment : positive;
  artifact_lane : ConversationLane;
  artifact_promises : list positive
}.

(** [ReviewReplyOutcome] is the policy-level result of triaging one review
    comment or unresolved review thread.  Python still performs the actual
    GitHub and task side effects; the model states which obligations apply. *)
Inductive ReviewReplyOutcome : Type :=
| ReviewAct
| ReviewDo
| ReviewAsk
| ReviewAnswer
| ReviewDefer
| ReviewDump.

(** [new_attempt] creates the initial shared metadata for a prepared promise. *)
Definition new_attempt (owner : ClaimOwner) : Attempt :=
  {| attempt_owner := owner;
     attempt_retry_count := 0;
     attempt_next_retry_after := 0 |}.

(** [retry_attempt] advances retry metadata after a failed attempt. *)
Definition retry_attempt (attempt : Attempt) : Attempt :=
  {| attempt_owner := attempt_owner attempt;
     attempt_retry_count := S (attempt_retry_count attempt);
     attempt_next_retry_after := S (attempt_next_retry_after attempt) |}.

(** [reset_attempt_retry] clears backoff metadata after an acked attempt. *)
Definition reset_attempt_retry (attempt : Attempt) : Attempt :=
  {| attempt_owner := attempt_owner attempt;
     attempt_retry_count := attempt_retry_count attempt;
     attempt_next_retry_after := 0 |}.

(** [claim_is_blocking_anchor] says whether an existing claim prevents the
    *anchor* — the comment that triggered this reply — from being claimed
    again.  An anchor is the comment whose own reply we are now generating;
    its dedup must catch both "someone is currently replying" (in-progress)
    and "we already replied to this exact comment" (completed, e.g. a
    webhook redelivery). *)
Definition claim_is_blocking_anchor (row : ClaimRow) : bool :=
  match claim_state row with
  | ClaimInProgress => true
  | ClaimCompleted => true
  | ClaimRetryableFailed => false
  end.

(** [claim_is_blocking_lineage] says whether an existing claim prevents a
    *non-anchor* lineage member from being co-covered by a new reply.  The
    only blocking state for non-anchor members is [ClaimInProgress] — a
    sibling handler is currently coalescing the same thread.  A
    [ClaimCompleted] non-anchor is a *prior* comment with its own
    already-delivered reply: it must not block a new comment in the same
    thread from getting its own reply (closes the dropped-comment-on-replied-
    thread bug). *)
Definition claim_is_blocking_lineage (row : ClaimRow) : bool :=
  match claim_state row with
  | ClaimInProgress => true
  | ClaimCompleted => false
  | ClaimRetryableFailed => false
  end.

(** [anchor_claimable] checks the new reply's anchor comment id against the
    durable claim table.  Stricter than [comment_claimable]: a completed
    claim on the anchor itself is treated as "already replied". *)
Definition anchor_claimable (claims : PositiveMap.t ClaimRow) (anchor : positive) : bool :=
  match PositiveMap.find anchor claims with
  | None => true
  | Some row => negb (claim_is_blocking_anchor row)
  end.

(** [comment_claimable] checks one *non-anchor* raw comment id against the
    durable claim table.  Retryable failures and completed prior replies
    are claimable (they don't block a new sibling); in-progress rows are
    not. *)
Definition comment_claimable (claims : PositiveMap.t ClaimRow) (comment : positive) : bool :=
  match PositiveMap.find comment claims with
  | None => true
  | Some row => negb (claim_is_blocking_lineage row)
  end.

(** [all_claimable] checks the aggregate-reply precondition for the
    *non-anchor* covered comment ids: every member must be claimable
    (i.e. not currently in-progress on a sibling handler) before one
    promise can co-cover them. *)
Fixpoint all_claimable (claims : PositiveMap.t ClaimRow) (comments : list positive) : bool :=
  match comments with
  | [] => true
  | comment :: rest =>
      if comment_claimable claims comment then all_claimable claims rest else false
  end.

(** [in_progress_row] builds the durable row installed by a prepared promise. *)
Definition in_progress_row (owner : ClaimOwner) (promise : positive) : ClaimRow :=
  {|
    claim_attempt := new_attempt owner;
    claim_state := ClaimInProgress;
    claim_promise := promise
  |}.

(** [claim_all] marks every covered comment id in progress for the same
    promise. *)
Fixpoint claim_all
    (owner : ClaimOwner)
    (promise : positive)
    (comments : list positive)
    (claims : PositiveMap.t ClaimRow) : PositiveMap.t ClaimRow :=
  match comments with
  | [] => claims
  | comment :: rest =>
      claim_all owner promise rest
        (PositiveMap.add comment (in_progress_row owner promise) claims)
  end.

(** [prepare_claims] atomically prepares one promise for a nonempty set of
    covered raw comment ids.  [None] represents the loser of a race. *)
Definition prepare_claims
    (owner : ClaimOwner)
    (promise : positive)
    (anchor : positive)
    (covered : list positive)
    (claims : PositiveMap.t ClaimRow)
    (promises : PositiveMap.t PromiseRow) : option (PositiveMap.t ClaimRow * PositiveMap.t PromiseRow) :=
  let comments := anchor :: covered in
  if andb (anchor_claimable claims anchor) (all_claimable claims covered) then
    let claims' := claim_all owner promise comments claims in
    let promise_row := {|
      promise_attempt := new_attempt owner;
      promise_state := PromisePrepared;
      promise_anchor_comment := anchor;
      promise_covered_comments := comments
    |} in
    Some (claims', PositiveMap.add promise promise_row promises)
  else
    None.

(** [mark_promise_posted] records that GitHub accepted the outbound reply
    attempt. *)
Definition mark_promise_posted
    (promise : positive)
    (promises : PositiveMap.t PromiseRow) : PositiveMap.t PromiseRow :=
  match PositiveMap.find promise promises with
  | None => promises
  | Some row =>
      PositiveMap.add promise
        {| promise_attempt := promise_attempt row;
           promise_state := PromisePosted;
           promise_anchor_comment := promise_anchor_comment row;
           promise_covered_comments := promise_covered_comments row |}
        promises
  end.

(** [promise_recoverable] says whether recovery should still inspect a durable
    promise.  Acked promises are terminal; all earlier states still need
    reconciliation against live GitHub state. *)
Definition promise_recoverable (state : PromiseState) : bool :=
  match state with
  | PromiseAcked => false
  | _ => true
  end.

(** [complete_comment] marks one raw comment id as completed by an acked
    promise. *)
Definition complete_comment
    (promise : positive)
    (comment : positive)
    (claims : PositiveMap.t ClaimRow) : PositiveMap.t ClaimRow :=
  match PositiveMap.find comment claims with
  | None => claims
  | Some row =>
      PositiveMap.add comment
        {| claim_attempt := reset_attempt_retry (claim_attempt row);
           claim_state := ClaimCompleted;
           claim_promise := promise |}
        claims
  end.

(** [complete_all] completes every raw comment id covered by a promise. *)
Fixpoint complete_all
    (promise : positive)
    (comments : list positive)
    (claims : PositiveMap.t ClaimRow) : PositiveMap.t ClaimRow :=
  match comments with
  | [] => claims
  | comment :: rest => complete_all promise rest (complete_comment promise comment claims)
  end.

(** [ack_promise] marks the promise acked and completes every covered comment
    id. *)
Definition ack_promise
    (promise : positive)
    (claims : PositiveMap.t ClaimRow)
    (promises : PositiveMap.t PromiseRow) : PositiveMap.t ClaimRow * PositiveMap.t PromiseRow :=
  match PositiveMap.find promise promises with
  | None => (claims, promises)
  | Some row =>
      let claims' := complete_all promise (promise_covered_comments row) claims in
      let promises' := PositiveMap.add promise
        {| promise_attempt := reset_attempt_retry (promise_attempt row);
           promise_state := PromiseAcked;
           promise_anchor_comment := promise_anchor_comment row;
           promise_covered_comments := promise_covered_comments row |}
        promises in
      (claims', promises')
  end.

(** [retryable_row] is the failure row installed after a pre-post failure. *)
Definition retryable_row (row : ClaimRow) : ClaimRow :=
  {| claim_attempt := retry_attempt (claim_attempt row);
     claim_state := ClaimRetryableFailed;
     claim_promise := claim_promise row |}.

(** [fail_comment] marks one claimed comment retryable. *)
Definition fail_comment (comment : positive) (claims : PositiveMap.t ClaimRow) : PositiveMap.t ClaimRow :=
  match PositiveMap.find comment claims with
  | None => claims
  | Some row => PositiveMap.add comment (retryable_row row) claims
  end.

(** [fail_all] marks all comments covered by a failed promise retryable. *)
Fixpoint fail_all (comments : list positive) (claims : PositiveMap.t ClaimRow) : PositiveMap.t ClaimRow :=
  match comments with
  | [] => claims
  | comment :: rest => fail_all rest (fail_comment comment claims)
  end.

(** [fail_promise] records a retryable promise failure and releases every
    covered comment id for a later owner. *)
Definition fail_promise
    (promise : positive)
    (claims : PositiveMap.t ClaimRow)
    (promises : PositiveMap.t PromiseRow) : PositiveMap.t ClaimRow * PositiveMap.t PromiseRow :=
  match PositiveMap.find promise promises with
  | None => (claims, promises)
  | Some row =>
      let claims' := fail_all (promise_covered_comments row) claims in
      let promises' := PositiveMap.add promise
        {| promise_attempt := retry_attempt (promise_attempt row);
           promise_state := PromiseFailed;
           promise_anchor_comment := promise_anchor_comment row;
           promise_covered_comments := promise_covered_comments row |}
        promises in
      (claims', promises')
  end.

(** [recover_promise] applies one live recovery observation to a durable
    promise.  Marker sightings and successful replay complete the lifecycle;
    deleted anchors and replay failures return the promise to retryable
    failure; comments that belong to another PR are left untouched for a later
    pass on the correct PR. *)
Definition recover_promise
    (promise : positive)
    (observation : RecoveryObservation)
    (claims : PositiveMap.t ClaimRow)
    (promises : PositiveMap.t PromiseRow) : PositiveMap.t ClaimRow * PositiveMap.t PromiseRow :=
  match observation with
  | SeenPromiseMarker => ack_promise promise claims promises
  | AnchorDeleted => fail_promise promise claims promises
  | WrongPullRequest => (claims, promises)
  | ReplayPosted =>
      let promises' := mark_promise_posted promise promises in
      let '(claims', promises'') := ack_promise promise claims promises' in
      (claims', promises'')
  | ReplayFailed => fail_promise promise claims promises
  end.

(** [record_reply_artifact] records one visible GitHub reply artifact and the
    promises it covers.  Runtime recovery uses hidden markers to recover this
    same many-promises-to-one-artifact relation after crashes. *)
Definition record_reply_artifact
    (artifact_comment : positive)
    (lane : ConversationLane)
    (covered_promises : list positive)
    (artifacts : PositiveMap.t ReplyArtifact) : PositiveMap.t ReplyArtifact :=
  PositiveMap.add artifact_comment
    {| artifact_comment := artifact_comment;
       artifact_lane := lane;
       artifact_promises := covered_promises |}
    artifacts.

(** [review_outcome_creates_tasks] states whether a review reply outcome owes
    one or more durable work-queue entries after the reply is posted. *)
Definition review_outcome_creates_tasks (outcome : ReviewReplyOutcome) : bool :=
  match outcome with
  | ReviewAct => true
  | ReviewDo => true
  | _ => false
  end.

(** [review_outcome_resolves_thread] states whether a review reply outcome
    should close the review thread after the reply is posted. *)
Definition review_outcome_resolves_thread (outcome : ReviewReplyOutcome) : bool :=
  match outcome with
  | ReviewDefer => true
  | ReviewDump => true
  | _ => false
  end.

(** [claim_completed] observes whether one raw comment id is completed. *)
Definition claim_state_completed (state : ClaimState) : bool :=
  match state with
  | ClaimCompleted => true
  | _ => false
  end.

(** [claim_completed] observes whether one raw comment id is completed. *)
Definition claim_completed (claims : PositiveMap.t ClaimRow) (comment : positive) : bool :=
  match PositiveMap.find comment claims with
  | Some row => claim_state_completed (claim_state row)
  | None => false
  end.

Python File Extraction replied_comment_claims
  "promise_recoverable anchor_claimable comment_claimable all_claimable prepare_claims mark_promise_posted ack_promise fail_promise recover_promise record_reply_artifact review_outcome_creates_tasks review_outcome_resolves_thread claim_completed".
