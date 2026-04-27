(** Webhook ingress deduplication FSM: at-least-once delivery vs
    exactly-once dispatch.

    Models GitHub's at-least-once delivery guarantee against Fido's
    exactly-once dispatch obligation.  GitHub may redeliver any webhook
    (same [X-GitHub-Delivery] header, same delivery id) if Fido's HTTP
    response was missing or took too long.  Additionally, a single inline
    review comment triggers two GitHub events — [pull_request_review_comment
    / created] and [pull_request_review / submitted] — with different
    delivery ids but the same semantic comment.  The FSM tracks one
    delivery slot through the ingress pipeline.

    [Fresh]      — this delivery has never been seen; dispatch is permitted.
                   The initial state for every new (delivery-id, comment-id)
                   pair entering the ingress pipeline.
    [Dispatched] — this delivery was seen and dispatch fired exactly once.
                   GitHub may redeliver the same delivery id; those
                   redeliveries are suppressed.
    [Collapsed]  — this delivery is a known double-fire: a
                   [pull_request_review / submitted] event whose inline
                   comment was already handled via the
                   [pull_request_review_comment / created] path.  No
                   dispatch fires for the [Collapsed] delivery; the
                   [review_comment] delivery is the canonical one.

    Three events name every ingress transition.

    [Arrive]         — a new delivery arrives for this slot.  The delivery
                       has not been seen before.  [Fresh] → [Dispatched]
                       (dispatch fires exactly once).
    [Redeliver]      — GitHub resends the same delivery id.  [Dispatched] →
                       [Dispatched] (suppressed; dispatch does not fire
                       again).
    [CollapseReview] — a [pull_request_review / submitted] event arrives
                       whose inline comment was already dispatched via the
                       [review_comment] path.  [Fresh] → [Collapsed] (no
                       dispatch for this delivery).

    Four proved invariants capture the exactly-once and collapse guarantees:

      [exactly_once_dispatch]        — [Arrive] is accepted only from
                                       [Fresh]; from [Dispatched] or
                                       [Collapsed] it returns [None].
                                       Dispatch fires at most once per slot.
      [redeliver_is_suppressed]      — [Redeliver] from [Dispatched] stays
                                       [Dispatched]; no second dispatch fires.
      [collapse_is_terminal]         — from [Collapsed], every event returns
                                       [None]; no transition leads back to
                                       [Fresh] or forward to [Dispatched].
      [redeliver_requires_prior_dispatch] — [Redeliver] from [Fresh] or
                                       [Collapsed] is rejected; a delivery
                                       must have been dispatched before a
                                       redelivery can be suppressed.

    E1 flip point: when the E-band work lands, the extracted [transition]
    function replaces the hand-written dedup logic in [events.py] (currently
    based on the SQLite [comment_claims] table and promise-marker recovery).
    The oracle path first wraps the current code and crashes loudly — naming
    the violated invariant — on any divergence.  At E1 the oracle becomes
    the control flow. *)

From FidoModels Require Import preamble.

(** * State

    Three phases of one webhook delivery slot in the ingress pipeline. *)
Inductive State : Type :=
| Fresh      : State
| Dispatched : State
| Collapsed  : State.

(** * Event

    [Arrive]         — first arrival of a new delivery; [Fresh] →
                       [Dispatched] (dispatch fires).
    [Redeliver]      — GitHub resends the same delivery id; [Dispatched] →
                       [Dispatched] (suppressed).
    [CollapseReview] — review-submitted double-fire suppressed in favour of
                       the review-comment delivery; [Fresh] → [Collapsed]
                       (no dispatch). *)
Inductive Event : Type :=
| Arrive         : Event
| Redeliver      : Event
| CollapseReview : Event.

(** * Transition function

    [transition current event] returns [Some new_state] when [event] is
    valid in [current], or [None] when it is rejected.

    Key rejection groups:
    - [Arrive] is rejected from [Dispatched] and [Collapsed] — a slot
      that has already been dispatched or collapsed cannot dispatch again.
    - [Redeliver] is rejected from [Fresh] and [Collapsed] — suppression
      requires a prior dispatch; a fresh slot has not dispatched yet and
      a collapsed slot was never dispatched via this delivery id.
    - [CollapseReview] is rejected from [Dispatched] and [Collapsed] — the
      collapse path is only valid before any dispatch has fired for this slot.
    - All events are rejected from [Collapsed] — [Collapsed] is a terminal
      absorbing state; no further transitions are possible once collapsed. *)
Definition transition (current : State) (event : Event) : option State :=
  match current, event with
  | Fresh,      Arrive         => Some Dispatched
  | Fresh,      CollapseReview => Some Collapsed
  | Dispatched, Redeliver      => Some Dispatched
  | _,          _              => None
  end.

Python File Extraction webhook_ingress_dedupe "transition".

(** * Proved invariants

    All lemmas follow by computation ([reflexivity]): [transition] reduces
    to concrete values on each listed combination, and Rocq's kernel verifies
    the equalities by normalisation.  No induction is needed.

    These names surface in the runtime oracle: when the Python dedup logic
    in [events.py] diverges from [transition], the crash message names the
    violated invariant so the engineer knows exactly which guarantee was
    broken. *)

(** [exactly_once_dispatch]: [Arrive] is accepted only from [Fresh].
    From [Dispatched] or [Collapsed] it returns [None], ensuring that
    dispatch fires at most once per delivery slot.  The second arrival of
    a delivery id never re-dispatches — the slot is already [Dispatched]
    and [Arrive] is refused. *)
Lemma exactly_once_dispatch :
  transition Fresh      Arrive = Some Dispatched /\
  transition Dispatched Arrive = None /\
  transition Collapsed  Arrive = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [redeliver_is_suppressed]: [Redeliver] from [Dispatched] stays
    [Dispatched].  GitHub's redelivery does not re-fire dispatch — the
    slot acknowledges the redelivery and remains [Dispatched].  This is
    the machine-checked statement of the at-least-once / exactly-once
    bridge: receiving the same delivery id a second time is harmless. *)
Lemma redeliver_is_suppressed :
  transition Dispatched Redeliver = Some Dispatched.
Proof.
  reflexivity.
Qed.

(** [collapse_is_terminal]: from [Collapsed], every event returns [None].
    Once a delivery slot is marked as a collapsed double-fire it is an
    absorbing state — no event can move it to [Fresh] (re-enable dispatch)
    or to [Dispatched] (claim dispatch fired).  The review-submitted
    delivery that was collapsed will never produce a second handler
    invocation regardless of future redeliveries. *)
Lemma collapse_is_terminal :
  transition Collapsed Arrive         = None /\
  transition Collapsed Redeliver      = None /\
  transition Collapsed CollapseReview = None.
Proof.
  repeat split; reflexivity.
Qed.

(** [redeliver_requires_prior_dispatch]: [Redeliver] from [Fresh] or
    [Collapsed] is rejected.  Suppressing a redelivery requires that a
    dispatch already fired for this slot ([Dispatched]); a [Fresh] slot
    has not dispatched yet and a [Collapsed] slot was suppressed before
    dispatch.  Accepting [Redeliver] from either would be a logic error —
    the slot cannot acknowledge a redelivery it never dispatched. *)
Lemma redeliver_requires_prior_dispatch :
  transition Fresh     Redeliver = None /\
  transition Collapsed Redeliver = None.
Proof.
  split; reflexivity.
Qed.
