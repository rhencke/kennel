(** Webhook-to-command translation types and FIFO bridge.

    Models GitHub webhook payloads as typed event descriptors and the typed
    commands they translate to, then bridges each command to the D4
    session-ownership FIFO contender kind it should be [Enqueue]d as.

    Design boundary: raw GitHub webhook JSON enters [dispatch] in
    [events.py] and exits as a typed [WebhookCommand].  [WebhookCommand]
    is what the D4 session-ownership FIFO [Enqueue]s via [cmd_to_contender];
    its payload is what the handler acts on when it [Dequeue]s.

    D5 lands as specification ahead of implementation.  Today's [dispatch]
    returns a loosely-typed [Action] dataclass; the model is the intended
    typed-command contract and will later serve as an oracle around the
    current Fido path, crashing with theorem names on divergence.

    Six handled (event-type, action) pairs map to five command kinds.
    Unknown or silently-filtered events — wrong action, missing required
    fields, self-comment, non-allowed author, non-PR issue comments,
    non-failure CI conclusion, non-merged PR close — cannot be constructed
    as [WebhookEvent] values at all, so [translate] is total (no [None]).
    This is the fail-closed contract: unrecognised shapes never yield
    commands.

    This is the second cross-[.v] import in the repo (after D4's
    [session_ownership_fifo] importing [replied_comment_claims]): D5
    imports [session_ownership_fifo] to access [Contender] for
    [cmd_to_contender]. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From FidoModels Require Import session_ownership_fifo.

From Stdlib Require Import
  Lists.List
  Numbers.BinNums
  PArith.BinPos
  Strings.String.

Open Scope string_scope.
Import ListNotations.

(* Prevent sort-polymorphism so nullary-constructor extraction is clean.
   See the note in [rocq-python-extraction/test/datatypes.v] for context. *)
Unset Universe Polymorphism.

(* Remap [option] so [Some x] erases to [x] and [None] stays [None].
   This makes [translate] return the [WebhookCommand] directly on success
   and [None] on rejection — a natural Python return type. *)
Extract Inductive option => ""
  [ "" "None" ]
  "(lambda fSome, fNone, x: fNone() if x is None else fSome(x))".

(** * DeliveryId

    Opaque identifier for one webhook delivery.  GitHub sends a unique
    [X-GitHub-Delivery] UUID header per attempt; Fido models it as a
    [positive] for dedup comparisons.  Two arrivals with the same
    [DeliveryId] are the same delivery redelivered — the translate oracle
    and handler must produce identical commands for identical delivery ids.
    The dedup predicate is [same_delivery] below. *)
Definition DeliveryId := positive.

(** * CheckConclusion

    The CI check outcomes that Fido acts on.  GitHub also sends [success],
    [neutral], [cancelled], [skipped], [stale], and [action_required];
    those conclusions never cross the translation boundary — only
    [CIFailure] and [CITimedOut] produce a [WebhookCommand].
    ([events.py:617-619]: conclusion not in ("failure", "timed_out") → None.) *)
Inductive CheckConclusion : Type :=
| CIFailure  : CheckConclusion
| CITimedOut : CheckConclusion.

(** * CommentKind

    Distinguishes the two PR comment surfaces Fido handles.  [ReviewLine]
    comments attach to specific file lines and are posted via the pulls
    review API; [TopLevelPR] comments appear on the issue thread and are
    posted via the issues API.  Both produce a [CmdComment] — the kind
    field routes the reply to the correct GitHub endpoint.

    ([events.py:535-611]: review comment → reply_to with comment_type "pulls";
     issue comment → thread with comment_type "issues".) *)
Inductive CommentKind : Type :=
| ReviewLine : CommentKind
| TopLevelPR : CommentKind.

(** * WebhookEvent

    Typed event descriptors extracted from raw GitHub webhook JSON.  One
    constructor per (event-type, action) pair that Fido may act on.

    The [wev_delivery] field is the idempotence key shared by every
    constructor.  It uniquely identifies one GitHub delivery attempt and
    is the dedup anchor for detecting redeliveries.

    Constructors match the six branches of [events.py:dispatch]:
      [WevReviewComment]    — pull_request_review_comment / created
      [WevIssueComment]     — issue_comment / created (on PR)
      [WevCIFailure]        — check_run / completed (failure or timed_out)
      [WevPRMerged]         — pull_request / closed (merged)
      [WevIssueAssigned]    — issues / assigned
      [WevReviewSubmitted]  — pull_request_review / submitted *)
Inductive WebhookEvent : Type :=

| WevReviewComment
    (wev_delivery   : DeliveryId)
    (wev_pr         : positive)
    (wev_comment_id : positive)
    (wev_author     : string)
    (wev_is_bot     : bool)
    : WebhookEvent

| WevIssueComment
    (wev_delivery   : DeliveryId)
    (wev_pr         : positive)
    (wev_comment_id : positive)
    (wev_author     : string)
    (wev_is_bot     : bool)
    : WebhookEvent

| WevCIFailure
    (wev_delivery   : DeliveryId)
    (wev_check_name : string)
    (wev_conclusion : CheckConclusion)
    (wev_pr_numbers : list positive)
    : WebhookEvent

| WevPRMerged
    (wev_delivery : DeliveryId)
    (wev_pr       : positive)
    : WebhookEvent

| WevIssueAssigned
    (wev_delivery : DeliveryId)
    (wev_issue    : positive)
    (wev_assignee : string)
    : WebhookEvent

| WevReviewSubmitted
    (wev_delivery  : DeliveryId)
    (wev_pr        : positive)
    (wev_review_id : positive)
    (wev_author    : string)
    : WebhookEvent.

(** * WebhookCommand

    Typed commands produced by translating a [WebhookEvent].  Each
    constructor carries the payload the session-ownership FIFO handler
    needs after [Dequeue].

    [cmd_delivery] is the idempotence key inherited from the originating
    [WebhookEvent].  A handler that sees the same [cmd_delivery] twice
    must not double-act — the dedup predicate (Task 2) formalises this
    obligation.

    [CmdComment] covers both [WevReviewComment] and [WevIssueComment];
    [cmd_kind] routes the reply to the correct GitHub API surface.

    [CmdReviewSubmitted] is kept separate from [CmdComment] because the
    review-level event is handled per-comment today ([events.py:528-533])
    and is a stub ([reply_to_review] is a no-op); the command exists to
    record the arrival and support future per-review triage.

    The mapping from [WebhookCommand] to a D4 [Contender] (for [Enqueue])
    is [cmd_to_contender] below. *)
Inductive WebhookCommand : Type :=

| CmdComment
    (cmd_delivery   : DeliveryId)
    (cmd_pr         : positive)
    (cmd_comment_id : positive)
    (cmd_author     : string)
    (cmd_is_bot     : bool)
    (cmd_kind       : CommentKind)
    : WebhookCommand

| CmdCIFailure
    (cmd_delivery   : DeliveryId)
    (cmd_check_name : string)
    (cmd_conclusion : CheckConclusion)
    (cmd_pr_numbers : list positive)
    : WebhookCommand

| CmdPRMerged
    (cmd_delivery : DeliveryId)
    (cmd_pr       : positive)
    : WebhookCommand

| CmdIssueAssigned
    (cmd_delivery : DeliveryId)
    (cmd_issue    : positive)
    (cmd_assignee : string)
    : WebhookCommand

| CmdReviewSubmitted
    (cmd_delivery  : DeliveryId)
    (cmd_pr        : positive)
    (cmd_review_id : positive)
    (cmd_author    : string)
    : WebhookCommand.

(** * cmd_delivery_id

    Accessor that extracts the [DeliveryId] from any [WebhookCommand].
    Used by the dedup predicate (Task 2) to compare two commands for
    idempotence without pattern-matching on the full command shape. *)
Definition cmd_delivery_id (cmd : WebhookCommand) : DeliveryId :=
  match cmd with
  | CmdComment         d _ _ _ _ _ => d
  | CmdCIFailure       d _ _ _     => d
  | CmdPRMerged        d _         => d
  | CmdIssueAssigned   d _ _       => d
  | CmdReviewSubmitted d _ _ _     => d
  end.

(** * translate

    Total function from [WebhookEvent] to [WebhookCommand].  Every
    constructor of [WebhookEvent] maps to exactly one constructor of
    [WebhookCommand]; there is no [None] branch.

    This totality is the Rocq statement of the fail-closed guarantee:
    a [WebhookEvent] can only be constructed from a recognised
    (event-type, action) pair, so [translate] never silently drops an
    event that crossed the JSON boundary.  Unknown shapes are excluded
    earlier, at the raw-JSON parser (not yet modelled); the absence of a
    [None] branch here is the formal witness.

    [translate_total] (proved below) states the existence witness.

    Delivery id threading: [translate] preserves [wev_delivery]
    unchanged into [cmd_delivery_id (translate ev) = wev_delivery_id ev].
    This is the idempotence anchor — a redelivered event with the same
    [DeliveryId] yields a command with the same [cmd_delivery_id], so
    the dedup check in [same_delivery] is delivery-stable.  Proved
    below as [translate_preserves_delivery]. *)
Definition translate (ev : WebhookEvent) : WebhookCommand :=
  match ev with

  | WevReviewComment d pr cid author is_bot =>
      CmdComment d pr cid author is_bot ReviewLine

  | WevIssueComment d pr cid author is_bot =>
      CmdComment d pr cid author is_bot TopLevelPR

  | WevCIFailure d name conclusion pr_nums =>
      CmdCIFailure d name conclusion pr_nums

  | WevPRMerged d pr =>
      CmdPRMerged d pr

  | WevIssueAssigned d issue assignee =>
      CmdIssueAssigned d issue assignee

  | WevReviewSubmitted d pr rid author =>
      CmdReviewSubmitted d pr rid author

  end.

(** * same_delivery

    Idempotence predicate: [same_delivery c1 c2] is [true] when [c1] and
    [c2] carry the same [DeliveryId].  Two commands with the same delivery
    id originated from the same physical GitHub webhook delivery (possibly
    redelivered); a handler that has already acted on one must skip the
    other.

    Implemented as [Pos.eqb] on [cmd_delivery_id] — decidable, pure, and
    extractable to Python [==] on integers.  [same_delivery_refl] and
    [same_delivery_sym] are proved below. *)
Definition same_delivery (c1 c2 : WebhookCommand) : bool :=
  Pos.eqb (cmd_delivery_id c1) (cmd_delivery_id c2).

(** * cmd_to_contender

    Total bridge from [WebhookCommand] to the D4 [Contender] kind that
    should be passed to [Enqueue] for this command.

    Every webhook-originated command maps to [Handler].  From
    [session_ownership_fifo]:
      [Handler] — covers webhook-handler and CI-fix turns; CI failures are
                  folded into [Handler] (no rank ordering inside the FIFO
                  queue — arrival timestamp is the only ordering).
      [CronSweep] — reserved for the periodic idle-drain sweep; never
                    produced from a webhook.

    This function is total — every constructor maps to exactly one
    [Contender].  The exhaustive match ensures that adding a new
    [WebhookCommand] constructor without updating [cmd_to_contender]
    is a compile-time error.

    [cmd_to_contender_is_handler] (proved below) is the formal statement
    that webhooks never enqueue a [CronSweep] contender:
    [∀ cmd, cmd_to_contender cmd = Handler]. *)
Definition cmd_to_contender (cmd : WebhookCommand) : Contender :=
  match cmd with
  | CmdComment         _ _ _ _ _ _ => Handler
  | CmdCIFailure       _ _ _ _     => Handler
  | CmdPRMerged        _ _         => Handler
  | CmdIssueAssigned   _ _ _       => Handler
  | CmdReviewSubmitted _ _ _ _     => Handler
  end.

(** * wev_delivery_id

    Uniform accessor that extracts the [DeliveryId] from any [WebhookEvent].
    The symmetric counterpart to [cmd_delivery_id]; used to state
    [translate_preserves_delivery] without pattern-matching on each
    constructor individually. *)
Definition wev_delivery_id (ev : WebhookEvent) : DeliveryId :=
  match ev with
  | WevReviewComment   d _ _ _ _ => d
  | WevIssueComment    d _ _ _ _ => d
  | WevCIFailure       d _ _ _   => d
  | WevPRMerged        d _       => d
  | WevIssueAssigned   d _ _     => d
  | WevReviewSubmitted d _ _ _   => d
  end.

(** * Proved invariants

    All lemmas follow by computation ([reflexivity], [destruct] +
    [reflexivity], or [Pos.eqb] lemmas): the functions reduce on concrete
    constructor shapes, and Rocq's kernel verifies the equalities by
    normalisation.

    These names surface in the runtime oracle: when the Python implementation
    diverges from the Rocq definitions, the crash message includes the theorem
    name so the engineer knows exactly which invariant was violated. *)

(** [translate_total]: every [WebhookEvent] yields a [WebhookCommand].
    [translate] is a total function with no [None] branch — the formal
    witness that all recognised events produce commands and no recognised
    event is silently dropped. *)
Lemma translate_total :
  forall ev : WebhookEvent, exists cmd : WebhookCommand, translate ev = cmd.
Proof.
  intro ev; exists (translate ev); reflexivity.
Qed.

(** [translate_preserves_delivery]: [translate] threads the [DeliveryId]
    from the event into the command unchanged.  This is the idempotence
    anchor: a redelivered event (same [wev_delivery_id]) yields a command
    with the same [cmd_delivery_id], so [same_delivery] is delivery-stable. *)
Lemma translate_preserves_delivery :
  forall ev : WebhookEvent,
    cmd_delivery_id (translate ev) = wev_delivery_id ev.
Proof.
  intro ev; destruct ev; reflexivity.
Qed.

(** [same_delivery_refl]: every command is a duplicate of itself.  A handler
    that compares its own command against a stored record using [same_delivery]
    always sees [true]. *)
Lemma same_delivery_refl :
  forall cmd : WebhookCommand, same_delivery cmd cmd = true.
Proof.
  intro cmd; unfold same_delivery; apply Pos.eqb_refl.
Qed.

(** [same_delivery_sym]: [same_delivery] is symmetric.  Dedup checks may
    compare in either order and get the same result. *)
Lemma same_delivery_sym :
  forall c1 c2 : WebhookCommand, same_delivery c1 c2 = same_delivery c2 c1.
Proof.
  intros c1 c2; unfold same_delivery; apply Pos.eqb_sym.
Qed.

(** [cmd_to_contender_is_handler]: every webhook command maps to [Handler].
    The formal statement that webhooks never enqueue a [CronSweep] contender
    — [CronSweep] is reserved exclusively for the periodic idle-drain sweep. *)
Lemma cmd_to_contender_is_handler :
  forall cmd : WebhookCommand, cmd_to_contender cmd = Handler.
Proof.
  intro cmd; destruct cmd; reflexivity.
Qed.
