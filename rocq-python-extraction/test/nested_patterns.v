(** Nested-pattern acceptance tests.

    These tests exercise [pp_pattern] at more than one level of nesting and
    across a mutual inductive group:

      has_left3       — 3-level deep nested constructor pattern with wildcards
      expr_is_num_pair — cross-type nested pattern over a mutual Expr/Val group

    Coverage (per issue #719):
      Nested [Pcons] at depth 3 within a single inductive (Tree)
      Nested [Pcons] crossing the boundary of a mutual inductive (Expr/Val) *)

Declare ML Module "rocq-python-extraction".

(* [Extract Inductive] and related vernaculars need the extraction plugin.
   [nat] and [bool] are Rocq prelude types; always in scope. *)
Declare ML Module "rocq-runtime.plugins.extraction".

(* Remap primitives so extracted round-trip functions return native Python
   types that are easy to assert on. *)
Extract Inductive bool => "bool" [ "True" "False" ].
Extract Inductive nat  => "int"
  [ "0" "(lambda x: x + 1)" ]
  "(lambda fO, fS, n: fO() if n == 0 else fS(n - 1))".

Unset Universe Polymorphism.

(* ------------------------------------------------------------------ *)
(*  1. Binary tree — 3-level deep nested pattern                        *)
(*                                                                      *)
(*  [has_left3] returns [true] iff the tree has at least three levels   *)
(*  of left children.  The first branch uses a nested constructor       *)
(*  pattern three levels deep:                                          *)
(*    Node _ (Node _ (Node _ _ _) _) _                                  *)
(*  This exercises [pp_pattern] recursing into [Pcons] sub-patterns.   *)
(* ------------------------------------------------------------------ *)

(** [Tree] is a binary tree with a nat payload at every internal node. *)
Inductive Tree :=
  | Leaf : Tree
  | Node : nat -> Tree -> Tree -> Tree.

(** [has_left3]: [true] iff the tree has a 3-level-deep left subtree. *)
Definition has_left3 (t : Tree) : bool :=
  match t with
  | Node _ (Node _ (Node _ _ _) _) _ => true
  | _ => false
  end.

Python Extraction has_left3.

(* ------------------------------------------------------------------ *)
(*  2. Expr/Val mutual inductive — cross-type nested pattern            *)
(*                                                                      *)
(*  [Expr] and [Val] form a mutually defined pair:                      *)
(*    ENum n   : a literal nat expression                               *)
(*    EAdd x y : sum of two expressions                                 *)
(*    ELift v  : lift a [Val] into [Expr]                               *)
(*    VNum n   : a numeric value                                        *)
(*    VPair u v: a pair of values                                       *)
(*                                                                      *)
(*  [expr_is_num_pair] returns [true] iff the expression is a [Val]    *)
(*  wrapped by [ELift] whose payload is a [VPair] of two [VNum]s.       *)
(*  The pattern [ELift (VPair (VNum _) (VNum _))] crosses the           *)
(*  [Expr]/[Val] type boundary, exercising [pp_pattern] on [Pcons]     *)
(*  sub-patterns drawn from a different mutual packet.                  *)
(* ------------------------------------------------------------------ *)

(** [Expr] and [Val] form a mutual expression/value language used to test
    nested patterns that cross mutual-inductive packet boundaries. *)
Inductive Expr :=
  | ENum  : nat -> Expr
  | EAdd  : Expr -> Expr -> Expr
  | ELift : Val -> Expr
with Val :=
  | VNum  : nat -> Val
  | VPair : Val -> Val -> Val.

(** [expr_is_num_pair]: [true] iff [e] is [ELift (VPair (VNum _) (VNum _))]. *)
Definition expr_is_num_pair (e : Expr) : bool :=
  match e with
  | ELift (VPair (VNum _) (VNum _)) => true
  | _ => false
  end.

Python Extraction expr_is_num_pair.
