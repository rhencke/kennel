(** Built-in finite collection remapping acceptance tests.

    This file intentionally has no local [Extract Inductive] or
    [Extract Constant] pragmas for [option], [list], [prod], finite maps, or
    finite sets.  The Python backend owns those Stdlib mappings directly. *)

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

(** [FiniteCollectionFixtures] keeps the finite collection acceptance fixtures in
    one extracted module so callers import a namespace instead of many
    one-value modules. *)
Module FiniteCollectionFixtures.

(** [p1] is the smallest positive key used in map fixtures. *)
Definition p1 : positive := xH.
(** [p2] is the even positive key used in set fixtures. *)
Definition p2 : positive := xO xH.
(** [p3] is the odd positive key used as the second map key. *)
Definition p3 : positive := xI xH.
(** [p5] is a larger odd positive key used in set fixtures. *)
Definition p5 : positive := xI (xO xH).
(** [p7] is the extra key used to test set union and difference. *)
Definition p7 : positive := xI (xI xH).

(** [positive_task_map] builds a positive-keyed finite map with two string
    values, exercising native Python dict extraction. *)
Definition positive_task_map : PositiveMap.t String.string :=
  PositiveMap.add p3 "ci" (PositiveMap.add p1 "plan" (@PositiveMap.empty String.string)).

(** [positive_task_hit] is the expected successful lookup result for [p1]. *)
Definition positive_task_hit : option String.string :=
  Some "plan".

(** [positive_task_missing] is the expected missing lookup result. *)
Definition positive_task_missing : option String.string :=
  None.

(** [positive_task_removed] removes [p1] from [positive_task_map], directly
    covering [PositiveMap.remove]. *)
Definition positive_task_removed : PositiveMap.t String.string :=
  PositiveMap.remove p1 positive_task_map.

(** [positive_task_has_3] is the expected membership result for [p3]. *)
Definition positive_task_has_3 : bool :=
  true.

(** [positive_task_count] is the expected map cardinality. *)
Definition positive_task_count : nat :=
  S (S O).

(** [positive_task_elements] is the expected ordered view of the map. *)
Definition positive_task_elements : list (positive * String.string) :=
  [(p1, "plan"); (p3, "ci")].

(** [positive_claim_set] builds a positive-keyed finite set with two keys. *)
Definition positive_claim_set : PositiveSet.t :=
  PositiveSet.add p5 (PositiveSet.add p2 PositiveSet.empty).

(** [positive_claim_union] adds [p7] to the base claim set. *)
Definition positive_claim_union : PositiveSet.t :=
  PositiveSet.add p7 positive_claim_set.

(** [positive_claim_inter] is the expected intersection fixture. *)
Definition positive_claim_inter : PositiveSet.t :=
  PositiveSet.add p5 PositiveSet.empty.

(** [positive_claim_diff] is the expected set difference fixture. *)
Definition positive_claim_diff : PositiveSet.t :=
  PositiveSet.add p7 PositiveSet.empty.

(** [positive_claim_union_expr] directly covers native set union lowering. *)
Definition positive_claim_union_expr : PositiveSet.t :=
  PositiveSet.union positive_claim_set positive_claim_diff.

(** [positive_claim_inter_expr] directly covers native set intersection
    lowering. *)
Definition positive_claim_inter_expr : PositiveSet.t :=
  PositiveSet.inter positive_claim_set positive_claim_union.

(** [positive_claim_diff_expr] directly covers native set difference lowering. *)
Definition positive_claim_diff_expr : PositiveSet.t :=
  PositiveSet.diff positive_claim_union positive_claim_set.

(** [positive_claim_nested_expr] keeps nested lowered set operations under
    source assertion coverage so later precedence-aware printing can preserve
    the intended grouping. *)
Definition positive_claim_nested_expr : PositiveSet.t :=
  PositiveSet.inter
    (PositiveSet.union positive_claim_set positive_claim_diff)
    positive_claim_union.

(** [positive_claim_removed] removes [p2] from the base set, directly
    covering [PositiveSet.remove]. *)
Definition positive_claim_removed : PositiveSet.t :=
  PositiveSet.remove p2 positive_claim_set.

(** [positive_claim_has_2] is the expected membership result for [p2]. *)
Definition positive_claim_has_2 : bool :=
  true.

(** [positive_claim_count] is the expected set cardinality. *)
Definition positive_claim_count : nat :=
  S (S O).

(** [positive_claim_elements] is the expected ordered view of the set. *)
Definition positive_claim_elements : list positive :=
  [p2; p5].

(** [string_label_map] is a neutral string-key list fixture for ordered views. *)
Definition string_label_map : list (String.string * nat) :=
  [("alpha", 1); ("beta", 2)].

(** [string_label_hit] is the expected successful lookup result for a label. *)
Definition string_label_hit : option nat :=
  Some 1.

(** [string_label_elements] aliases the expected string-key element view. *)
Definition string_label_elements : list (String.string * nat) :=
  string_label_map.

(** [string_label_set] keeps caller-provided labels as runtime input rather
    than baking a repo list into extracted code. *)
Definition string_label_set (labels : list String.string) : list String.string :=
  labels.

(** [string_label_set_elements] exposes a runtime-provided label list as the
    expected ordered element view. *)
Definition string_label_set_elements (labels : list String.string) : list String.string :=
  string_label_set labels.

End FiniteCollectionFixtures.

Python Module Extraction FiniteCollectionFixtures.
