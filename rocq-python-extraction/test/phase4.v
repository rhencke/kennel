(** Phase 4 acceptance tests: inductive datatype emission.

    Non-remapped inductives emit Python [@dataclass(frozen=True)] hierarchies.
    Each section defines a Rocq inductive and extracts a function that
    exercises constructor building and pattern matching on it.  The emitted
    Python is then verified via round-trip assertions in the [dune] rules.

    Coverage (per issue #715):
      MyList A      — parameterised singly-linked list (MNil/MCons)
      BinTree       — non-parameterised binary tree carrying nat values
      RoseTree A    — mutual parameterised rose tree / forest
      MTree/MForest — mutual non-parameterised tree / forest
      MyOpt A       — polymorphic option; option-of-option flatten
      Even/Odd      — mutual non-parameterised parity types + mutual fixpoint *)

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

(* Prevent user-defined inductives from being treated as sort-polymorphic.
   Rocq 9.1.0 had a bug where sort-polymorphic inductives (those whose
   parameters can be Prop or Type) caused an ML arity mismatch during
   extraction when pattern-matching on nullary constructors, giving:
     "constructor expected N argument(s) while applied to 0"
   Rocq 9.1.1 fixed the underlying bug (PR #21479).  Keeping this flag
   here is still good practice: it makes all inductives in this file
   monomorphic, which is the correct semantic for Python extraction. *)
Unset Universe Polymorphism.

(* ------------------------------------------------------------------ *)
(*  1. Polymorphic singly-linked list                                   *)
(* ------------------------------------------------------------------ *)

Inductive MyList (A : Set) :=
  | MNil  : MyList A
  | MCons : A -> MyList A -> MyList A.

(* In Rocq 9.x the inductive parameter [A] is not automatically made implicit
   in constructor arguments; we must declare it explicitly so that pattern
   arms like [| MNil => ...] do not require the type as an extra argument. *)
Arguments MNil {A}.
Arguments MCons {A} _ _.

(** [mylist_is_empty]: [True] iff the list is [MNil].
    Exercises pattern matching on a parameterised non-remapped inductive
    and returning a remapped primitive (bool → Python bool). *)
Definition mylist_is_empty {A : Set} (l : MyList A) : bool :=
  match l with
  | MNil      => true
  | MCons _ _ => false
  end.

Python Extraction mylist_is_empty.

(* ------------------------------------------------------------------ *)
(*  2. Binary tree carrying nat values                                  *)
(* ------------------------------------------------------------------ *)

Inductive BinTree :=
  | BLeaf : BinTree
  | BNode : BinTree -> nat -> BinTree -> BinTree.

(** [bintree_is_leaf]: [True] iff the tree is a [BLeaf].
    Exercises a non-parameterised inductive whose constructor ([BNode]) has
    a field of a remapped type ([nat → int]). *)
Definition bintree_is_leaf (t : BinTree) : bool :=
  match t with
  | BLeaf       => true
  | BNode _ _ _ => false
  end.

Python Extraction bintree_is_leaf.

(* ------------------------------------------------------------------ *)
(*  3. Rose tree / forest (mutual, parameterised)                       *)
(* ------------------------------------------------------------------ *)

Inductive RoseTree (A : Set) :=
  | RNode : A -> RoseForest A -> RoseTree A
with RoseForest (A : Set) :=
  | RFNil  : RoseForest A
  | RFCons : RoseTree A -> RoseForest A -> RoseForest A.

Arguments RNode {A} _ _.
Arguments RFNil {A}.
Arguments RFCons {A} _ _.

(** [roseforest_is_empty]: [True] iff the forest has no trees.
    Exercises the second packet of a mutual parameterised inductive. *)
Definition roseforest_is_empty {A : Set} (f : RoseForest A) : bool :=
  match f with
  | RFNil      => true
  | RFCons _ _ => false
  end.

Python Extraction roseforest_is_empty.

(* ------------------------------------------------------------------ *)
(*  4. Mutual tree / forest (non-parameterised)                         *)
(* ------------------------------------------------------------------ *)

Inductive MTree :=
  | MLeaf : nat -> MTree
  | MNode : nat -> MForest -> MTree
with MForest :=
  | FNil  : MForest
  | FCons : MTree -> MForest -> MForest.

(** [mforest_is_empty]: [True] iff the forest has no trees.
    Exercises the second packet of a mutual non-parameterised inductive. *)
Definition mforest_is_empty (f : MForest) : bool :=
  match f with
  | FNil      => true
  | FCons _ _ => false
  end.

Python Extraction mforest_is_empty.

(* ------------------------------------------------------------------ *)
(*  5. Polymorphic option; option-of-option flatten                     *)
(* ------------------------------------------------------------------ *)

Inductive MyOpt (A : Set) :=
  | MyNone : MyOpt A
  | MySome : A -> MyOpt A.

Arguments MyNone {A}.
Arguments MySome {A} _.

(** [myopt_flatten]: flatten one level of option nesting.
    [MyNone] stays [MyNone]; [MySome x] unwraps to [x].
    Exercises: (a) constructing a zero-arg non-remapped dataclass in the
    return position ([MyNone()] rather than the class object [MyNone]),
    and (b) returning a value bound in the enclosing match arm. *)
Definition myopt_flatten {A : Set} (o : MyOpt (MyOpt A)) : MyOpt A :=
  match o with
  | MyNone   => MyNone
  | MySome x => x
  end.

Python Extraction myopt_flatten.

(* ------------------------------------------------------------------ *)
(*  6. Lowercase-first inductive — capitalization smoke test           *)
(*                                                                      *)
(*  Rocq convention uses lowercase for type names (e.g. [color]);      *)
(*  the extraction framework follows OCaml and also lowercases them.   *)
(*  This section verifies that [python.ml] re-capitalizes the base     *)
(*  class to [Color] in the emitted Python (PEP 8 PascalCase).         *)
(* ------------------------------------------------------------------ *)

Inductive color :=
  | Red   : color
  | Green : color
  | Blue  : color.

(** [color_is_red]: [True] iff the color is [Red].
    The generated file must contain [class Color:] (capitalized) as the
    shared base class; constructors [Red], [Green], [Blue] inherit from
    it.  The dune rule asserts [isinstance(Red(), Color)] to confirm the
    capitalized name is present and correct. *)
Definition color_is_red (c : color) : bool :=
  match c with
  | Red   => true
  | Green => false
  | Blue  => false
  end.

Python Extraction color_is_red.

(* ------------------------------------------------------------------ *)
(*  7. Even/Odd mutual inductive — mutual types + mutual fixpoint       *)
(*                                                                      *)
(*  [Even] and [Odd] are a pair of mutually defined types encoding      *)
(*  parity witnesses:                                                   *)
(*    EvenO      : zero is even                                         *)
(*    EvenS o    : if o : Odd  then the successor is even               *)
(*    OddS  e    : if e : Even then the successor is odd                *)
(*                                                                      *)
(*  This exercises:                                                      *)
(*    (a) a mutual inductive group whose constructors cross-reference   *)
(*        one another (EvenS carries an Odd; OddS carries an Even),     *)
(*    (b) a mutually-recursive fixpoint (Dfix) that pattern-matches on  *)
(*        both packets in alternation.                                   *)
(* ------------------------------------------------------------------ *)

Inductive Even :=
  | EvenO : Even
  | EvenS : Odd -> Even
with Odd :=
  | OddS : Even -> Odd.

(** [even_depth]: count the number of constructor alternations.
    With nat → int remapping:
      even_depth EvenO                          = 0
      even_depth (EvenS (OddS EvenO))           = 2
      even_depth (EvenS (OddS (EvenS (OddS EvenO)))) = 4 *)
Fixpoint even_depth (e : Even) : nat :=
  match e with
  | EvenO   => O
  | EvenS o => S (odd_depth o)
  end
with odd_depth (o : Odd) : nat :=
  match o with
  | OddS e => S (even_depth e)
  end.

Python Extraction even_depth.
