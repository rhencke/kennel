(** Keyed coordination-state vocabulary.

    This model exercises backend-owned finite-map and finite-set extraction for
    the identifier shapes Fido uses most often: numeric GitHub IDs and string
    repo/provider/task IDs.

    The primitives here underpin two invariant clusters from the bug-mined
    survey ([BUG_MINED_INVARIANTS.md]):

      - Cluster F (reply/claim dedup, D1 #739): [add_claim] / [remove_claim] /
        [has_claim] — the finite-set operations that [replied_comment_claims.v]
        uses to enforce exactly-once replies.

      - Cluster G (picker eligibility, D16 #888): [assign_issue] /
        [unassign_issue] / [issue_owner] — the finite-map operations that
        track which GitHub issue each repo's worker is responsible for.

    Neither cluster is proved here; this file provides the extracted primitives
    on which the proving models build. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import
  FSets.FMapPositive
  MSets.MSetPositive
  Lists.List
  Numbers.BinNums
  Strings.String.

Open Scope positive_scope.
Open Scope string_scope.
Import ListNotations.

(** * Claim-set primitives (cluster F — reply/claim dedup, D1 #739)

    [add_claim] records that a numeric thread/claim id is currently owned.
    The caller supplies the current set; no claim ids are known at compile
    time.  The full dedup model — which proves at-most-one-reply per anchor —
    lives in [replied_comment_claims.v]; these three operations are its
    extracted building blocks. *)
Definition add_claim (thread : positive) (claims : PositiveSet.t) : PositiveSet.t :=
  PositiveSet.add thread claims.

(** [remove_claim] clears a numeric thread/claim id from a caller-provided
    claim set.  This covers the finite-set remove operation in the model
    surface Fido will use for coordination state. *)
Definition remove_claim (thread : positive) (claims : PositiveSet.t) : PositiveSet.t :=
  PositiveSet.remove thread claims.

(** [has_claim] tests membership in the runtime claim set for one positive id. *)
Definition has_claim (claims : PositiveSet.t) (thread : positive) : bool :=
  PositiveSet.mem thread claims.

(** * Issue-owner map primitives (cluster G — picker eligibility, D16 #888)

    [assign_issue] associates a runtime GitHub issue id with a runtime owner
    string in the caller-provided issue-owner map.  PR task/checklist state is
    deliberately separate from this GitHub issue coordination index.  The
    picker-eligibility model (D16 #888) uses these operations to verify that
    fresh-retry re-checks live assignment rather than a cached snapshot. *)
Definition assign_issue (issue : positive) (owner : String.string)
    (owners : PositiveMap.t String.string) : PositiveMap.t String.string :=
  PositiveMap.add issue owner owners.

(** [unassign_issue] removes a runtime GitHub issue id from the caller-provided
    issue-owner map. *)
Definition unassign_issue (issue : positive)
    (owners : PositiveMap.t String.string) : PositiveMap.t String.string :=
  PositiveMap.remove issue owners.

(** [issue_owner] looks up the owner string for a runtime GitHub issue id. *)
Definition issue_owner (owners : PositiveMap.t String.string)
    (issue : positive) : option String.string :=
  PositiveMap.find issue owners.

(** * Repo-list primitives

    [repo_entry] is one CLI-provided repo tuple: owner/repo, path on disk, and
    provider.  The repo collection is supplied at runtime, so this type is
    only the shape of one entry, not a compile-time repo list. *)
Definition repo_entry : Type :=
  (String.string * (String.string * String.string))%type.

(** [repo_provider] projects the provider field from one runtime repo entry. *)
Definition repo_provider (repo : repo_entry) : String.string :=
  match repo with
  | (_, (_, provider)) => provider
  end.

(** [repo_providers] projects providers from the runtime repo entries without
    baking any repo or provider names into generated code.  It traverses the
    input list as given; callers should not treat that as a semantic order. *)
Fixpoint repo_providers (repos : list repo_entry) : list String.string :=
  match repos with
  | [] => []
  | repo :: rest => repo_provider repo :: repo_providers rest
  end.

(** [repo_count] counts the runtime repo list provided by the CLI. *)
Definition repo_count (repos : list repo_entry) : nat :=
  List.length repos.

(** [CoordIndex] groups the runtime coordination inputs that otherwise repeat
    across most operations.  It is the source-level object boundary we want the
    Python backend to recognize as a class-shaped API. *)
Record CoordIndex : Type := {
  coord_claims : PositiveSet.t;
  coord_issue_owners : PositiveMap.t String.string;
  coord_repos : list repo_entry
}.

(** [empty_coord_index] creates an empty runtime coordination index. *)
Definition empty_coord_index : CoordIndex := {|
  coord_claims := PositiveSet.empty;
  coord_issue_owners := PositiveMap.empty String.string;
  coord_repos := []
|}.

(** [coord_add_claim] records an owned claim in a whole coordination index. *)
Definition coord_add_claim (index : CoordIndex) (thread : positive) : CoordIndex := {|
  coord_claims := add_claim thread index.(coord_claims);
  coord_issue_owners := index.(coord_issue_owners);
  coord_repos := index.(coord_repos)
|}.

(** [coord_remove_claim] clears an owned claim in a whole coordination index. *)
Definition coord_remove_claim (index : CoordIndex) (thread : positive) : CoordIndex := {|
  coord_claims := remove_claim thread index.(coord_claims);
  coord_issue_owners := index.(coord_issue_owners);
  coord_repos := index.(coord_repos)
|}.

(** [coord_has_claim] checks claim ownership through the grouped index. *)
Definition coord_has_claim (index : CoordIndex) (thread : positive) : bool :=
  has_claim index.(coord_claims) thread.

(** [coord_assign_issue] records a GitHub issue owner in the grouped index. *)
Definition coord_assign_issue (index : CoordIndex) (issue : positive)
    (owner : String.string) : CoordIndex := {|
  coord_claims := index.(coord_claims);
  coord_issue_owners := assign_issue issue owner index.(coord_issue_owners);
  coord_repos := index.(coord_repos)
|}.

(** [coord_unassign_issue] clears a GitHub issue owner in the grouped index. *)
Definition coord_unassign_issue (index : CoordIndex) (issue : positive) : CoordIndex := {|
  coord_claims := index.(coord_claims);
  coord_issue_owners := unassign_issue issue index.(coord_issue_owners);
  coord_repos := index.(coord_repos)
|}.

(** [coord_issue_owner] looks up a GitHub issue owner through the grouped index. *)
Definition coord_issue_owner (index : CoordIndex) (issue : positive)
    : option String.string :=
  issue_owner index.(coord_issue_owners) issue.

(** [coord_repo_providers] projects runtime repo providers through the grouped
    index. *)
Definition coord_repo_providers (index : CoordIndex) : list String.string :=
  repo_providers index.(coord_repos).

(** [coord_repo_count] counts runtime repos through the grouped index. *)
Definition coord_repo_count (index : CoordIndex) : nat :=
  repo_count index.(coord_repos).

Python File Extraction coord_index
  "add_claim remove_claim has_claim assign_issue unassign_issue issue_owner repo_providers repo_count empty_coord_index coord_add_claim coord_remove_claim coord_has_claim coord_assign_issue coord_unassign_issue coord_issue_owner coord_repo_providers coord_repo_count".
