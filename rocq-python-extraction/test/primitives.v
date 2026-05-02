(** Primitive-remapping acceptance tests.

    Each section extracts a Rocq function via an [Extract Inductive] pragma
    that maps a Rocq primitive to the matching Python builtin, then verifies
    [extracted(f(x)) == expected(x)] via a Python driver in the [dune] file.

    Coverage (grows as tasks land):
      bool   → Python bool      ([True]/[False] ternary, both directions)
      nat    → Python int       (constructor inline lambdas + custom match)
      option → Python Optional  (Some erased to value; None → Python None)
      prod   → Python tuple     (erased constructor; tuple patterns in match)
      list   → Python list      (nil→[], cons→prepend; custom match on emptiness)
*)

Declare ML Module "rocq-python-extraction".

(* [Extract Inductive] and related vernaculars are registered by the
   rocq-runtime.plugins.extraction ML plugin, part of rocq-core.  Load it
   directly — no rocq-stdlib import needed.  [nat], [bool], [option], [prod],
   and [list] are prelude types; always in scope without explicit imports.

   [list_scope] is defined in [Init.Datatypes] (part of the prelude) and
   provides the [[h :: t]] cons notation.  Open it here so the list section
   can use infix notation without importing [Lists.List]. *)
Declare ML Module "rocq-runtime.plugins.extraction".
From Stdlib Require Import Bool.Bool.
Open Scope list_scope.

(* ------------------------------------------------------------------ *)
(*  bool → Python bool                                                 *)
(* ------------------------------------------------------------------ *)

Extract Inductive bool => "bool" [ "True" "False" ].

(** [bool_not]: logical negation defined inline so the extraction includes
    only the ternary pattern; no Stdlib dependency, no extra declarations.
    Extracted form: [False if b else True]. *)
Definition bool_not (b : bool) : bool := if b then false else true.

(** [bool_and]: standard bool conjunction lowers to a native Python [and]
    expression instead of retaining the Rocq helper call. *)
Definition bool_and (b1 b2 : bool) : bool := andb b1 b2.

(** [bool_or]: standard bool disjunction lowers to a native Python [or]
    expression instead of retaining the Rocq helper call. *)
Definition bool_or (b1 b2 : bool) : bool := orb b1 b2.

(** [bool_neg]: standard bool negation lowers to a native Python [not]
    expression instead of retaining the Rocq helper call. *)
Definition bool_neg (b : bool) : bool := negb b.

(** [bool_neg_and]: nested lowered bool operations must preserve Python
    precedence with parentheses around the lowered [and] expression. *)
Definition bool_neg_and (b1 b2 : bool) : bool := negb (andb b1 b2).

(** [bool_neg_or]: nested lowered bool operations must preserve Python
    precedence with parentheses around the lowered [or] expression. *)
Definition bool_neg_or (b1 b2 : bool) : bool := negb (orb b1 b2).

(** [bool_or_and]: conjunction binds more tightly than disjunction, so a
    lowered [and] expression can feed [or] without extra parentheses. *)
Definition bool_or_and (b1 b2 b3 : bool) : bool := orb b1 (andb b2 b3).

(** [bool_and_or]: a lowered disjunction used as a conjunction operand must
    stay parenthesized so Python does not parse the result as [(b1 and b2) or b3]. *)
Definition bool_and_or (b1 b2 b3 : bool) : bool := andb b1 (orb b2 b3).

(** [bool_eq]: standard bool equality lowers to native Python equality instead
    of retaining the Rocq helper call. *)
Definition bool_eq (b1 b2 : bool) : bool := Bool.eqb b1 b2.

(** [bool_eq_and]: equality has higher precedence than conjunction, so the
    lowered equality operand must not be wrapped just to feed [and]. *)
Definition bool_eq_and (b1 b2 b3 : bool) : bool := andb (Bool.eqb b1 b2) b3.

(** [bool_and_eq]: a lowered conjunction used as an equality operand must stay
    parenthesized so Python does not parse the result as [b1 and (b2 == b3)]. *)
Definition bool_and_eq (b1 b2 b3 : bool) : bool := Bool.eqb (andb b1 b2) b3.

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

(** [nat_count_down]: tail-recursive accumulator loop.
    Extracted Python must not rely on Python tail-call optimization. *)
Fixpoint nat_count_down (n acc : nat) : nat :=
  match n with
  | O   => acc
  | S m => nat_count_down m (S acc)
  end.

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

(** [option_nat_neq]: optional primitive values compare directly in Python
    instead of expanding the structural option match. *)
Definition option_nat_neq (left right : option nat) : bool :=
  match left, right with
  | Some l, Some r => negb (Nat.eqb l r)
  | None, None => false
  | _, _ => true
  end.

(* ------------------------------------------------------------------ *)
(*  prod → Python tuple                                                *)
(*                                                                     *)
(*  Constructor mapping:                                               *)
(*    pair → ""   (multi-arg erasure: emit (a, b) directly)           *)
(*  No custom match function needed: Python tuple patterns work natively *)
(*  in [match]/[case], so the general structural match path handles it. *)
(* ------------------------------------------------------------------ *)

Extract Inductive prod => "" [ "" ].

(** [pair_swap]: swap the two components of a [nat * nat] pair.
    Exercises both tuple construction ([MLcons]/[MLtuple] with erased
    constructor → [(a, b)] literal) and tuple pattern matching
    ([Pcons]/[Ptuple] → [(a, b)] pattern).  With [nat → int] and
    [prod → tuple], the extracted function operates on plain Python tuples
    of ints.
    Expected: pair_swap (a, b) = (b, a) for all a, b. *)
Definition pair_swap (p : nat * nat) : nat * nat :=
  match p with
  | (a, b) => (b, a)
  end.

(** [pair_first]: standard [fst] lowers to tuple index access. *)
Definition pair_first (p : nat * bool) : nat :=
  fst p.

(** [pair_second]: standard [snd] lowers to tuple index access. *)
Definition pair_second (p : nat * bool) : bool :=
  snd p.

(* ------------------------------------------------------------------ *)
(*  list → Python list                                                 *)
(*                                                                     *)
(*  Constructor mapping:                                               *)
(*    nil  → "[]"                    (empty list literal)              *)
(*    cons → "(lambda h, t: [h] + t)" (prepend head onto tail)        *)
(*  Custom match function dispatches on Python list emptiness:         *)
(*    (lambda fnil, fcons, l: fnil() if l == [] else fcons(l[0], l[1:])) *)
(*  Applied as: fn(nil_thunk, cons_thunk, scrutinee)                  *)
(* ------------------------------------------------------------------ *)

Extract Inductive list => "list"
  [ "[]" "(lambda h, t: [h] + t)" ]
  "(lambda fnil, fcons, l: fnil() if l == [] else fcons(l[0], l[1:]))".

(** [list_add_one]: increment every element of a [list nat] by one.
    Exercises both the nil branch (returns []) and the cons branch
    (prepends the incremented head onto the recursively-processed tail).
    With [nat → int] and [list → list], the extracted function operates
    on plain Python lists of ints.
    Expected: list_add_one [] = []; list_add_one [h; …] = [h+1; …]. *)
Fixpoint list_add_one (l : list nat) : list nat :=
  match l with
  | nil    => nil
  | h :: t => (S h) :: list_add_one t
  end.

(** [list_cons_append]: list cons and list append both lower to Python [+].
    The generated expression must preserve the left-associative grouping of
    prepending one element before appending the rest. *)
Definition list_cons_append (h : nat) (left right : list nat) : list nat :=
  h :: (left ++ right).

(** [list_append_left_nested]: nested list append lowers to left-associative
    Python [+] without redundant parentheses. *)
Definition list_append_left_nested
    (left middle right : list nat) : list nat :=
  (left ++ middle) ++ right.

(** [list_append_right_nested]: nested list append on the right keeps the
    generated Python expression flat, relying on list-append associativity. *)
Definition list_append_right_nested
    (left middle right : list nat) : list nat :=
  left ++ (middle ++ right).

(** [list_append_let_child]: a lambda-lifted let expression can feed lowered
    list append as its left child. *)
Definition list_append_let_child
    (h : nat) (right : list nat) : list nat :=
  (let prefix := h :: nil in prefix) ++ right.

(** [list_append_match_child]: a lowered boolean match expression must be
    parenthesized when it feeds lowered list append. *)
Definition list_append_match_child
    (flag : bool) (right : list nat) : list nat :=
  (if flag then O :: nil else S O :: nil) ++ right.

(** [lambda_call_head]: application of an inline lambda must parenthesize the
    call head because Python lambda has lower precedence than calls. *)
Definition lambda_call_head (n : nat) : nat :=
  (fun f => f n) (fun x => S x).

Python File Extraction primitives
  "bool_not bool_and bool_or bool_neg bool_neg_and bool_neg_or bool_or_and bool_and_or bool_eq bool_eq_and bool_and_eq nat_double nat_count_down option_inc option_nat_neq pair_swap pair_first pair_second list_add_one list_cons_append list_append_left_nested list_append_right_nested list_append_let_child list_append_match_child lambda_call_head".
