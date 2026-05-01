(** Durable reply/outbox idempotence protocol.

    D14 composes the earlier replied-comment claim/promise model with the
    outbox side-effect boundary.  The durable keys are deliberately explicit:
    webhook delivery ids, semantic origin comment ids, reply-promise ids, and
    outbox effect ids all participate in deciding whether Python may generate
    or post visible work.

    The protocol has three safety gates:

    - [prepare_reply] claims the semantic origin before any reply generation
      may begin.  A second delivery for the same origin cannot create another
      live promise, even if it has a different GitHub delivery id.
    - [claim_outbox_effect] claims a committed outbox effect before Python may
      call GitHub.  Duplicate/retried executions of the same effect id observe
      the durable row instead of emitting another visible side effect.
    - [record_reply_posted], [record_deferred_issue_opened], and
      [record_outbox_failure] release the claim only after success or failure
      has been committed durably.

    This model is still an oracle around the current Python path.  At the E1
    flip point, the scheduler/reducer should commit this transition first and
    run GitHub replies, deferred issue creation, task creation, reactions, and
    thread resolution only from the returned outbox effects. *)

From FidoModels Require Import preamble.

From Stdlib Require Import
  FSets.FMapPositive
  Lists.List
  Numbers.BinNums
  PArith.BinPos.

Open Scope positive_scope.
Import ListNotations.

(** [OriginKind] names the GitHub surface that produced a semantic reply
    origin.  Review-thread origins are keyed by the root thread comment;
    top-level PR comments are keyed by the issue-comment id. *)
Inductive OriginKind : Type :=
| ReviewThreadOrigin
| IssueCommentOrigin.

(** [OutboxKind] distinguishes visible reply posts from deferred issue
    creation.  Both are externally visible effects and both require durable
    idempotence. *)
Inductive OutboxKind : Type :=
| ReplyPostEffect
| DeferredIssueEffect.

(** [ReplyClaimState] tracks the semantic origin, not just an in-memory claim
    set.  [ReplyClaimRetryable] means a pre-post failure released the origin
    for a later durable promise. *)
Inductive ReplyClaimState : Type :=
| ReplyClaimInProgress
| ReplyClaimCompleted
| ReplyClaimRetryable.

(** [EffectState] is the durable lifecycle for one outbox effect id. *)
Inductive EffectState : Type :=
| EffectPrepared
| EffectClaimed
| EffectDelivered
| EffectFailed.

(** [EffectDecision] is the adapter-visible answer to "may I call GitHub for
    this effect now?". *)
Inductive EffectDecision : Type :=
| EmitEffect
| ReuseDeliveredEffect
| WaitForInFlightEffect
| RetryLaterEffect
| RejectMissingEffect.

(** [OriginClaim] is keyed by semantic origin comment id. *)
Record OriginClaim : Type := {
  origin_claim_kind : OriginKind;
  origin_claim_state : ReplyClaimState;
  origin_claim_promise : positive;
  origin_claim_delivery : positive
}.

(** [OutboxEffect] is keyed by durable effect id.  [effect_external_id] is the
    visible GitHub comment id or tracking issue id after delivery succeeds. *)
Record OutboxEffect : Type := {
  effect_kind : OutboxKind;
  effect_origin : positive;
  effect_promise : positive;
  effect_state : EffectState;
  effect_external_id : option positive
}.

(** [ProtocolState] is the durable projection needed for D14. *)
Record ProtocolState : Type := {
  protocol_origins : PositiveMap.t OriginClaim;
  protocol_deliveries : PositiveMap.t positive;
  protocol_effects : PositiveMap.t OutboxEffect;
  protocol_deferred_effects : PositiveMap.t positive;
  protocol_live_replies : PositiveMap.t positive;
  protocol_live_issues : PositiveMap.t positive
}.

Definition empty_protocol_state : ProtocolState := {|
  protocol_origins := PositiveMap.empty OriginClaim;
  protocol_deliveries := PositiveMap.empty positive;
  protocol_effects := PositiveMap.empty OutboxEffect;
  protocol_deferred_effects := PositiveMap.empty positive;
  protocol_live_replies := PositiveMap.empty positive;
  protocol_live_issues := PositiveMap.empty positive
|}.

Definition claim_blocks_generation (claim : OriginClaim) : bool :=
  match origin_claim_state claim with
  | ReplyClaimInProgress => true
  | ReplyClaimCompleted => true
  | ReplyClaimRetryable => false
  end.

Definition origin_claimable (state : ProtocolState) (origin : positive) : bool :=
  match PositiveMap.find origin (protocol_origins state) with
  | None => true
  | Some claim => negb (claim_blocks_generation claim)
  end.

Definition origin_completed (state : ProtocolState) (origin : positive) : bool :=
  match PositiveMap.find origin (protocol_origins state) with
  | Some claim =>
      match origin_claim_state claim with
      | ReplyClaimCompleted => true
      | _ => false
      end
  | None => false
  end.

Definition delivery_origin
    (state : ProtocolState)
    (delivery : positive) : option positive :=
  PositiveMap.find delivery (protocol_deliveries state).

Definition promise_owns_origin
    (state : ProtocolState)
    (promise origin : positive) : bool :=
  match PositiveMap.find origin (protocol_origins state) with
  | Some claim =>
      if Pos.eqb (origin_claim_promise claim) promise then
        match origin_claim_state claim with
        | ReplyClaimInProgress => true
        | _ => false
        end
      else
        false
  | None => false
  end.

Definition can_generate_reply
    (state : ProtocolState)
    (promise origin : positive) : bool :=
  promise_owns_origin state promise origin.

Definition in_progress_claim
    (kind : OriginKind)
    (promise delivery : positive) : OriginClaim :=
  {| origin_claim_kind := kind;
     origin_claim_state := ReplyClaimInProgress;
     origin_claim_promise := promise;
     origin_claim_delivery := delivery |}.

Definition prepared_effect
    (kind : OutboxKind)
    (origin promise : positive) : OutboxEffect :=
  {| effect_kind := kind;
     effect_origin := origin;
     effect_promise := promise;
     effect_state := EffectPrepared;
     effect_external_id := None |}.

(** [prepare_reply] durably claims the origin and records the delivery before
    generation.  A duplicate semantic origin returns [None] unless the prior
    claim is retryable. *)
Definition prepare_reply
    (delivery origin promise reply_effect : positive)
    (kind : OriginKind)
    (state : ProtocolState) : option ProtocolState :=
  if origin_claimable state origin then
    Some {|
      protocol_origins := PositiveMap.add origin
        (in_progress_claim kind promise delivery)
        (protocol_origins state);
      protocol_deliveries := PositiveMap.add delivery origin
        (protocol_deliveries state);
      protocol_effects := PositiveMap.add reply_effect
        (prepared_effect ReplyPostEffect origin promise)
        (protocol_effects state);
      protocol_deferred_effects := protocol_deferred_effects state;
      protocol_live_replies := protocol_live_replies state;
      protocol_live_issues := protocol_live_issues state
    |}
  else
    None.

Definition effect_claimable (effect : OutboxEffect) : bool :=
  match effect_state effect with
  | EffectPrepared => true
  | EffectFailed => true
  | _ => false
  end.

Definition effect_decision_for (effect : OutboxEffect) : EffectDecision :=
  match effect_state effect with
  | EffectPrepared => EmitEffect
  | EffectClaimed => WaitForInFlightEffect
  | EffectDelivered => ReuseDeliveredEffect
  | EffectFailed => RetryLaterEffect
  end.

Definition outbox_decision
    (state : ProtocolState)
    (effect_id : positive) : EffectDecision :=
  match PositiveMap.find effect_id (protocol_effects state) with
  | None => RejectMissingEffect
  | Some effect => effect_decision_for effect
  end.

Definition claimed_effect (effect : OutboxEffect) : OutboxEffect :=
  {| effect_kind := effect_kind effect;
     effect_origin := effect_origin effect;
     effect_promise := effect_promise effect;
     effect_state := EffectClaimed;
     effect_external_id := effect_external_id effect |}.

(** [claim_outbox_effect] is the claim-before-post gate. *)
Definition claim_outbox_effect
    (effect_id : positive)
    (state : ProtocolState) : option ProtocolState :=
  match PositiveMap.find effect_id (protocol_effects state) with
  | Some effect =>
      if effect_claimable effect then
        Some {|
          protocol_origins := protocol_origins state;
          protocol_deliveries := protocol_deliveries state;
          protocol_effects := PositiveMap.add effect_id
            (claimed_effect effect)
            (protocol_effects state);
          protocol_deferred_effects := protocol_deferred_effects state;
          protocol_live_replies := protocol_live_replies state;
          protocol_live_issues := protocol_live_issues state
        |}
      else
        None
  | None => None
  end.

Definition delivered_effect
    (external_id : positive)
    (effect : OutboxEffect) : OutboxEffect :=
  {| effect_kind := effect_kind effect;
     effect_origin := effect_origin effect;
     effect_promise := effect_promise effect;
     effect_state := EffectDelivered;
     effect_external_id := Some external_id |}.

Definition completed_claim (claim : OriginClaim) : OriginClaim :=
  {| origin_claim_kind := origin_claim_kind claim;
     origin_claim_state := ReplyClaimCompleted;
     origin_claim_promise := origin_claim_promise claim;
     origin_claim_delivery := origin_claim_delivery claim |}.

Definition retryable_claim (claim : OriginClaim) : OriginClaim :=
  {| origin_claim_kind := origin_claim_kind claim;
     origin_claim_state := ReplyClaimRetryable;
     origin_claim_promise := origin_claim_promise claim;
     origin_claim_delivery := origin_claim_delivery claim |}.

Definition failed_effect (effect : OutboxEffect) : OutboxEffect :=
  {| effect_kind := effect_kind effect;
     effect_origin := effect_origin effect;
     effect_promise := effect_promise effect;
     effect_state := EffectFailed;
     effect_external_id := effect_external_id effect |}.

(** [record_reply_posted] commits the visible reply artifact and completes the
    origin.  Calling it again for the same effect id is idempotent: a delivered
    effect is returned unchanged and no second live reply is recorded. *)
Definition record_reply_posted
    (effect_id artifact_id : positive)
    (state : ProtocolState) : option ProtocolState :=
  match PositiveMap.find effect_id (protocol_effects state) with
  | Some effect =>
      match effect_kind effect, effect_state effect with
      | ReplyPostEffect, EffectClaimed =>
          let origin := effect_origin effect in
          match PositiveMap.find origin (protocol_origins state) with
          | Some claim =>
              Some {|
                protocol_origins := PositiveMap.add origin
                  (completed_claim claim)
                  (protocol_origins state);
                protocol_deliveries := protocol_deliveries state;
                protocol_effects := PositiveMap.add effect_id
                  (delivered_effect artifact_id effect)
                  (protocol_effects state);
                protocol_deferred_effects := protocol_deferred_effects state;
                protocol_live_replies := PositiveMap.add origin
                  artifact_id
                  (protocol_live_replies state);
                protocol_live_issues := protocol_live_issues state
              |}
          | None => None
          end
      | ReplyPostEffect, EffectDelivered => Some state
      | _, _ => None
      end
  | None => None
  end.

(** [prepare_deferred_issue] commits the intent to create a tracking issue as a
    distinct outbox effect id, but tied to the same promise and origin. *)
Definition prepare_deferred_issue
    (issue_effect origin promise : positive)
    (state : ProtocolState) : option ProtocolState :=
  match PositiveMap.find origin (protocol_origins state) with
  | Some claim =>
      if Pos.eqb (origin_claim_promise claim) promise then
        match PositiveMap.find origin (protocol_deferred_effects state) with
        | Some existing_effect =>
            if Pos.eqb existing_effect issue_effect then Some state else None
        | None =>
            match PositiveMap.find issue_effect (protocol_effects state) with
            | None =>
                Some {|
                  protocol_origins := protocol_origins state;
                  protocol_deliveries := protocol_deliveries state;
                  protocol_effects := PositiveMap.add issue_effect
                    (prepared_effect DeferredIssueEffect origin promise)
                    (protocol_effects state);
                  protocol_deferred_effects := PositiveMap.add origin
                    issue_effect
                    (protocol_deferred_effects state);
                  protocol_live_replies := protocol_live_replies state;
                  protocol_live_issues := protocol_live_issues state
                |}
            | Some _ => Some state
            end
        end
      else
        None
  | None => None
  end.

(** [record_deferred_issue_opened] commits the issue id for a deferred effect.
    Replays of the same effect id reuse the first recorded issue. *)
Definition record_deferred_issue_opened
    (effect_id issue_id : positive)
    (state : ProtocolState) : option ProtocolState :=
  match PositiveMap.find effect_id (protocol_effects state) with
  | Some effect =>
      match effect_kind effect, effect_state effect with
      | DeferredIssueEffect, EffectClaimed =>
          Some {|
            protocol_origins := protocol_origins state;
            protocol_deliveries := protocol_deliveries state;
            protocol_effects := PositiveMap.add effect_id
              (delivered_effect issue_id effect)
              (protocol_effects state);
            protocol_deferred_effects := protocol_deferred_effects state;
            protocol_live_replies := protocol_live_replies state;
            protocol_live_issues := PositiveMap.add effect_id issue_id
              (protocol_live_issues state)
          |}
      | DeferredIssueEffect, EffectDelivered => Some state
      | _, _ => None
      end
  | None => None
  end.

(** [record_outbox_failure] releases the semantic origin after a failed reply
    post, and records retryable failure for issue effects without unclaiming
    the already-claimed reply origin. *)
Definition record_outbox_failure
    (effect_id : positive)
    (state : ProtocolState) : option ProtocolState :=
  match PositiveMap.find effect_id (protocol_effects state) with
  | Some effect =>
      match effect_state effect with
      | EffectClaimed =>
          let effects' := PositiveMap.add effect_id
            (failed_effect effect)
            (protocol_effects state) in
          match effect_kind effect with
          | ReplyPostEffect =>
              let origin := effect_origin effect in
              match PositiveMap.find origin (protocol_origins state) with
              | Some claim =>
                  Some {|
                    protocol_origins := PositiveMap.add origin
                      (retryable_claim claim)
                      (protocol_origins state);
                    protocol_deliveries := protocol_deliveries state;
                    protocol_effects := effects';
                    protocol_deferred_effects := protocol_deferred_effects state;
                    protocol_live_replies := protocol_live_replies state;
                    protocol_live_issues := protocol_live_issues state
                  |}
              | None => None
              end
          | DeferredIssueEffect =>
              Some {|
                protocol_origins := protocol_origins state;
                protocol_deliveries := protocol_deliveries state;
                protocol_effects := effects';
                protocol_deferred_effects := protocol_deferred_effects state;
                protocol_live_replies := protocol_live_replies state;
                protocol_live_issues := protocol_live_issues state
              |}
          end
      | _ => None
      end
  | None => None
  end.

Definition live_reply_for_origin
    (state : ProtocolState)
    (origin : positive) : option positive :=
  PositiveMap.find origin (protocol_live_replies state).

Definition live_issue_for_effect
    (state : ProtocolState)
    (effect_id : positive) : option positive :=
  PositiveMap.find effect_id (protocol_live_issues state).

Definition effect_external
    (state : ProtocolState)
    (effect_id : positive) : option positive :=
  match PositiveMap.find effect_id (protocol_effects state) with
  | Some effect => effect_external_id effect
  | None => None
  end.

Python File Extraction reply_outbox_protocol
  "empty_protocol_state claim_blocks_generation origin_claimable origin_completed delivery_origin promise_owns_origin can_generate_reply prepare_reply outbox_decision claim_outbox_effect record_reply_posted prepare_deferred_issue record_deferred_issue_opened record_outbox_failure live_reply_for_origin live_issue_for_effect effect_external".

(** * Proved invariants *)

Definition sample_prepared : ProtocolState :=
  match prepare_reply 10 20 30 40 ReviewThreadOrigin empty_protocol_state with
  | Some state => state
  | None => empty_protocol_state
  end.

Definition sample_claimed : ProtocolState :=
  match claim_outbox_effect 40 sample_prepared with
  | Some state => state
  | None => sample_prepared
  end.

Definition sample_posted : ProtocolState :=
  match record_reply_posted 40 50 sample_claimed with
  | Some state => state
  | None => sample_claimed
  end.

(** [claim_before_generate]: a reply can be generated only after the semantic
    origin has been durably claimed by that promise. *)
Lemma claim_before_generate :
  can_generate_reply empty_protocol_state 30 20 = false /\
  can_generate_reply sample_prepared 30 20 = true /\
  can_generate_reply sample_prepared 31 20 = false.
Proof.
  repeat split; reflexivity.
Qed.

(** [duplicate_origin_rejected]: a second delivery for the same semantic
    origin cannot prepare another live reply while the first claim is active. *)
Lemma duplicate_origin_rejected :
  prepare_reply 11 20 31 41 ReviewThreadOrigin sample_prepared = None.
Proof.
  reflexivity.
Qed.

(** [claim_before_post]: an outbox effect must be claimed before a visible
    reply can be recorded. *)
Lemma claim_before_post :
  record_reply_posted 40 50 sample_prepared = None /\
  outbox_decision sample_prepared 40 = EmitEffect /\
  outbox_decision sample_claimed 40 = WaitForInFlightEffect.
Proof.
  repeat split; reflexivity.
Qed.

(** [reply_post_is_idempotent]: once a reply artifact is delivered, replaying
    the same effect id keeps the first artifact and marks the origin completed. *)
Lemma reply_post_is_idempotent :
  live_reply_for_origin sample_posted 20 = Some 50 /\
  origin_completed sample_posted 20 = true /\
  record_reply_posted 40 51 sample_posted = Some sample_posted /\
  outbox_decision sample_posted 40 = ReuseDeliveredEffect.
Proof.
  repeat split; reflexivity.
Qed.

Definition sample_with_issue : ProtocolState :=
  match prepare_deferred_issue 60 20 30 sample_posted with
  | Some state => state
  | None => sample_posted
  end.

Definition sample_issue_claimed : ProtocolState :=
  match claim_outbox_effect 60 sample_with_issue with
  | Some state => state
  | None => sample_with_issue
  end.

Definition sample_issue_opened : ProtocolState :=
  match record_deferred_issue_opened 60 70 sample_issue_claimed with
  | Some state => state
  | None => sample_issue_claimed
  end.

(** [deferred_issue_creation_is_idempotent]: a retried deferred issue outbox
    effect cannot create two live issue ids for the same committed intent. *)
Lemma deferred_issue_creation_is_idempotent :
  live_issue_for_effect sample_issue_opened 60 = Some 70 /\
  record_deferred_issue_opened 60 71 sample_issue_opened = Some sample_issue_opened /\
  outbox_decision sample_issue_opened 60 = ReuseDeliveredEffect.
Proof.
  repeat split; reflexivity.
Qed.

(** [one_deferred_issue_effect_per_intent]: one origin/promise intent cannot
    prepare a second tracking-issue effect under a different effect id. *)
Lemma one_deferred_issue_effect_per_intent :
  prepare_deferred_issue 61 20 30 sample_with_issue = None.
Proof.
  reflexivity.
Qed.

(** [failure_releases_origin]: a failed reply post releases its semantic
    origin for a later durable retry. *)
Lemma failure_releases_origin :
  match record_outbox_failure 40 sample_claimed with
  | Some failed_state => origin_claimable failed_state 20 = true
  | None => False
  end.
Proof.
  reflexivity.
Qed.
