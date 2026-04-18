(** Phase 3 acceptance tests: primitive type remapping round-trips.

    Each section extracts a Rocq function via an [Extract Inductive] pragma
    that maps a Rocq primitive to the matching Python builtin, then verifies
    [extracted(f(x)) == expected(x)] via a Python driver in the [dune] file.

    Coverage (grows as tasks land):
      bool → Python bool  ([True]/[False] ternary, both directions)
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
