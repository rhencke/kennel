(** Phase 3 acceptance tests: primitive type remapping round-trips.

    Each section extracts a Rocq function via an [Extract Inductive] pragma
    that maps a Rocq primitive to the matching Python builtin, then verifies
    [extracted(f(x)) == expected(x)] via a Python driver in the [dune] file.

    Coverage (grows as tasks land):
      bool   → Python bool      ([True]/[False] ternary, both directions)
      nat    → Python int       (constructor inline lambdas + custom match)
      option → Python Optional  (Some erased to value; None → Python None)
*)

Declare ML Module "rocq-python-extraction".

From Stdlib Require Import extraction.Extraction.

(* ------------------------------------------------------------------ *)
(*  bool → Python bool                                                 *)
(* ------------------------------------------------------------------ *)

Extract Inductive bool => "bool" [ "True" "False" ].

(** [bool_not]: logical negation defined inline so the extraction includes
    only the ternary pattern; no Stdlib dependency, no extra declarations.
    Extracted form: [False if b else True]. *)
Definition bool_not (b : bool) : bool := if b then false else true.

Python Extraction bool_not.

(* ------------------------------------------------------------------ *)
(*  nat → Python int                                                   *)
(*                                                                     *)
(*  Constructor mapping:                                               *)
(*    O → 0             (zero literal)                                 *)
(*    S → (lambda x: x + 1)  (inline successor function)              *)
(*  Custom match function encodes case analysis via Python lambda:     *)
(*    (lambda fO, fS, n: fO() if n == 0 else fS(n - 1))              *)
(*  Applied as: fn(O_thunk, S_thunk, scrutinee)                       *)
(* ------------------------------------------------------------------ *)

Extract Inductive nat => "int"
  [ "0" "(lambda x: x + 1)" ]
  "(lambda fO, fS, n: fO() if n == 0 else fS(n - 1))".

(** [nat_double]: 2*n via structural recursion — exercises both the
    O branch (returns 0) and the S branch (applies successor twice).
    Extracted form uses the custom match function to dispatch on int
    and inline lambdas for constructor application.
    Expected: nat_double(k) == 2 * k for all k ≥ 0. *)
Fixpoint nat_double (n : nat) : nat :=
  match n with
  | O   => O
  | S m => S (S (nat_double m))
  end.

Python Extraction nat_double.

(* ------------------------------------------------------------------ *)
(*  option → Python Optional                                           *)
(*                                                                     *)
(*  Constructor mapping (Rocq ctor order: Some=0, None=1):            *)
(*    Some → ""     (singleton erasure: emit the wrapped value as-is) *)
(*    None → "None" (Python's None literal)                           *)
(*  Custom match function dispatches on Python None-ness:             *)
(*    (lambda fSome, fNone, x: fNone() if x is None else fSome(x))   *)
(*  Applied as: fn(Some_thunk, None_thunk, scrutinee)                 *)
(* ------------------------------------------------------------------ *)

Extract Inductive option => ""
  [ "" "None" ]
  "(lambda fSome, fNone, x: fNone() if x is None else fSome(x))".

(** [option_inc]: lift successor over option nat — exercises both the
    Some branch (applies S to the wrapped nat) and the None branch
    (propagates None).  With nat→int and option→Optional, the extracted
    function operates on plain Python ints and Python None.
    Expected: option_inc None = None; option_inc (Some n) = Some (S n). *)
Definition option_inc (o : option nat) : option nat :=
  match o with
  | None   => None
  | Some n => Some (S n)
  end.

Python Extraction option_inc.
