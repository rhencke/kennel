(** Record-projection acceptance tests (issue #720).

    Exercises [MLcase] on a record inductive, which the new optimisation
    converts to a lambda-lifted attribute-access form instead of a
    [match]/[case] statement:

      (lambda f0, f1, …: body)(scrutinee.f0, scrutinee.f1, …)

    Coverage:
      2-field record [pair_r]: projection of first field, second field, and
        a swap function that reads both fields and constructs a new record.
      5-field record [point5]: individual projection of each of the 5 named
        fields — this is the acceptance criterion from issue #720: "a Rocq
        record with 5 named fields extracts with readable Python field names
        and attribute access". *)

Declare ML Module "rocq-python-extraction".

(* [Extract Inductive] and related vernaculars need the extraction plugin. *)
Declare ML Module "rocq-runtime.plugins.extraction".

(* Remap nat → int so round-trip assertions operate on plain Python ints. *)
Extract Inductive nat => "int"
  [ "0" "(lambda x: x + 1)" ]
  "(lambda fO, fS, n: fO() if n == 0 else fS(n - 1))".

(* ------------------------------------------------------------------ *)
(*  1. 2-field record — basic projection and swap                      *)
(* ------------------------------------------------------------------ *)

(** [pair_r] is a two-field record used to test named-field access and record
    construction in generated Python. *)
Record pair_r := MkPairR { pfst_r : nat ; psnd_r : nat }.

(** [proj_first]: project the first field. The single-branch MLcase triggers
    the record-projection optimisation, yielding attribute access [p.pfst_r]. *)
Definition proj_first (p : pair_r) : nat :=
  match p with MkPairR f _ => f end.

(** [proj_second]: project the second field. *)
Definition proj_second (p : pair_r) : nat :=
  match p with MkPairR _ s => s end.

(** [swap_pair_r]: swap the two fields.  The match reads both via attribute
    access and the record-construction notation emits keyword arguments, so
    the combined output exercises both the projection path and the existing
    [MLcons Record] keyword-argument path. *)
Definition swap_pair_r (p : pair_r) : pair_r :=
  match p with
  | MkPairR f s => {| pfst_r := s ; psnd_r := f |}
  end.

(** [set_second_direct]: record update shape that keeps one field from an
    existing dataclass and changes another.  The extractor should emit
    [dataclasses.replace] rather than reconstructing every field manually. *)
Definition set_second_direct (p : pair_r) (n : nat) : pair_r :=
  {| pfst_r := pfst_r p ; psnd_r := n |}.

Axiom pair_r_eq : pair_r -> pair_r -> bool.
Extract Constant pair_r_eq => "__PY_NATIVE_EQ__".

Definition pair_r_same (left right : pair_r) : bool :=
  pair_r_eq left right.

(* ------------------------------------------------------------------ *)
(*  2. 5-field record — acceptance criterion for issue #720            *)
(*                                                                     *)
(*  A Rocq record with 5 named fields must extract with readable       *)
(*  Python field names and attribute access (not positional _0/_1/…). *)
(* ------------------------------------------------------------------ *)

(** [point5] is a five-field record acceptance fixture for readable generated
    field names. *)
Record point5 := MkPoint5 {
  p5_x : nat ; p5_y : nat ; p5_z : nat ; p5_w : nat ; p5_v : nat
}.

(** Individual field projections — each triggers one single-branch
    MLcase → lambda-lifted attribute-access transformation. *)
Definition get_p5_x (p : point5) : nat :=
  match p with MkPoint5 x _ _ _ _ => x end.

(** [get_p5_y] projects the second field from [point5]. *)
Definition get_p5_y (p : point5) : nat :=
  match p with MkPoint5 _ y _ _ _ => y end.

(** [get_p5_z] projects the third field from [point5]. *)
Definition get_p5_z (p : point5) : nat :=
  match p with MkPoint5 _ _ z _ _ => z end.

(** [get_p5_w] projects the fourth field from [point5]. *)
Definition get_p5_w (p : point5) : nat :=
  match p with MkPoint5 _ _ _ w _ => w end.

(** [get_p5_v] projects the fifth field from [point5]. *)
Definition get_p5_v (p : point5) : nat :=
  match p with MkPoint5 _ _ _ _ v => v end.

(* ------------------------------------------------------------------ *)
(*  Grouped extraction                                                 *)
(*  All definitions land in a single [records.py] module.            *)
(* ------------------------------------------------------------------ *)

(** [records.py]: covers 2-field record projection and swap, and
    5-field record individual field projections (issue #720). *)
Python File Extraction records "proj_first proj_second swap_pair_r set_second_direct pair_r_same get_p5_x get_p5_y get_p5_z get_p5_w get_p5_v".
