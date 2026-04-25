(** Datatype acceptance tests: inductive datatype emission.

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
      Even/Odd      — mutual non-parameterised parity types + mutual fixpoint
      NTree         — nested inductive (node carries a list of subtrees)
      STree/DTree   — mutual syntax-tree / decl-tree + mutual fixpoint *)

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

(** [MyList] is a parameterised, non-remapped list used to test generated
    dataclass hierarchies for type-parameterised inductives. *)
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

(* ------------------------------------------------------------------ *)
(*  2. Binary tree carrying nat values                                  *)
(* ------------------------------------------------------------------ *)

(** [BinTree] is a non-parameterised tree whose node payload is a remapped
    [nat], mixing custom dataclasses with primitive Python ints. *)
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

(* ------------------------------------------------------------------ *)
(*  3. Rose tree / forest (mutual, parameterised)                       *)
(* ------------------------------------------------------------------ *)

(** [RoseTree] and [RoseForest] are mutually-defined, parameterised
    inductives.  They verify that generated type variables are shared across
    both packets of a mutual block. *)
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

(* ------------------------------------------------------------------ *)
(*  4. Mutual tree / forest (non-parameterised)                         *)
(* ------------------------------------------------------------------ *)

(** [MTree] and [MForest] are mutually-defined, non-parameterised inductives
    used to test cross-packet constructor references without type variables. *)
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

(* ------------------------------------------------------------------ *)
(*  5. Polymorphic option; option-of-option flatten                     *)
(* ------------------------------------------------------------------ *)

(** [MyOpt] is a local option-like inductive that is deliberately not remapped
    to Python [None], so constructor classes are still emitted. *)
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

(* ------------------------------------------------------------------ *)
(*  6. Lowercase-first inductive — capitalization smoke test           *)
(*                                                                      *)
(*  Rocq convention uses lowercase for type names (e.g. [color]);      *)
(*  the extraction framework follows OCaml and also lowercases them.   *)
(*  This section verifies that [python.ml] re-capitalizes the base     *)
(*  class to [Color] in the emitted Python (PEP 8 PascalCase).         *)
(* ------------------------------------------------------------------ *)

(** [color] starts with a lowercase type name to verify that generated Python
    base classes are capitalized without changing constructor names. *)
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

(** [Even] and [Odd] are mutually-defined parity witnesses used by the mutual
    recursive depth/counting fixtures below. *)
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

(** [is_even]: Boolean parity check via mutual fixpoint over nat.
    [is_even] and [is_odd] are mutually recursive: each pattern-matches
    on nat and delegates the successor case to the other.
    With nat → int and bool → Python bool remappings:
      is_even 0 = True    is_odd 0 = False
      is_even 1 = False   is_odd 1 = True
      is_even 2 = True    is_odd 2 = False
    Exercises a Dfix mutual fixpoint entirely over remapped types. *)
Fixpoint is_even (n : nat) : bool :=
  match n with
  | O    => true
  | S n' => is_odd n'
  end
with is_odd (n : nat) : bool :=
  match n with
  | O    => false
  | S n' => is_even n'
  end.

(* ------------------------------------------------------------------ *)
(*  9. Nested inductive — tree carrying a list of subtrees             *)
(*                                                                      *)
(*  [NTree] is a "nested" inductive: the [NNode] constructor stores   *)
(*  its children as [list NTree] — an inductive type nested inside     *)
(*  the standard [list] container.  Unlike a fully mutual definition,  *)
(*  [NTree] only mentions itself through the pre-existing [list]       *)
(*  parameter rather than through another simultaneously-defined type. *)
(*                                                                      *)
(*  Rocq's [list] is remapped to Python's native [list] via an         *)
(*  [Extract Inductive list] directive so that [NNode] carries a plain *)
(*  Python list in the generated code.                                  *)
(*                                                                      *)
(*  This exercises:                                                      *)
(*    (a) a constructor whose field type is a remapped generic          *)
(*        container ([list NTree] → Python [list[NTree]]),              *)
(*    (b) a type annotation involving both a remapped and a non-        *)
(*        remapped type ([list[NTree]]),                                 *)
(*    (c) pattern matching on the nested inductive returning a          *)
(*        remapped primitive (bool → Python bool).                      *)
(* ------------------------------------------------------------------ *)

(* Remap Rocq's standard [list] to Python's built-in [list].
   Constructor remappings:
     nil  → []
     cons → lambda h, t: [h] + t
   Match function: inspect the Python list and dispatch to the
   zero-arg [fnil] thunk or the two-arg [fcons] with head/tail. *)
Extract Inductive list =>
  "list"
  [ "[]" "(lambda h, t: [h] + t)" ]
  "(lambda fnil, fcons, xs: fnil() if not xs else fcons(xs[0], xs[1:]))".

(** [NTree] is a nested inductive: recursive children are stored inside the
    remapped standard [list] container. *)
Inductive NTree :=
  | NLeaf : NTree
  | NNode : list NTree -> NTree.

(** [ntree_is_leaf]: [True] iff the tree is a bare [NLeaf] node.
    Exercises: (a) constructing an [NLeaf()] and an [NNode(children)]
    in Python where [children] is a native Python list of [NTree]
    instances, and (b) pattern-matching on a nested inductive returning
    a remapped bool. *)
Definition ntree_is_leaf (t : NTree) : bool :=
  match t with
  | NLeaf   => true
  | NNode _ => false
  end.

(* ------------------------------------------------------------------ *)
(*  10. STree / DTree (mutual, non-parameterised)                       *)
(*      — syntax-tree / decl-tree mutual group + mutual fixpoint        *)
(*                                                                      *)
(*  [STree] (syntax-tree node) and [DTree] (decl-tree / declaration    *)
(*  list) are a pair of mutually-defined non-parameterised inductives   *)
(*  modelling a tiny expression language:                               *)
(*    SLit n   : a literal nat node                                     *)
(*    SSeq d s : a block — run declarations [d] then evaluate body [s] *)
(*    DEnd     : empty declaration list                                  *)
(*    DDecl s d: prepend syntax node [s] to declaration list [d]        *)
(*                                                                      *)
(*  The two types cross-reference one another:                          *)
(*    SSeq   carries a DTree  (declarations precede the body)           *)
(*    DDecl  carries an STree (each declaration is a syntax node)       *)
(*                                                                      *)
(*  This exercises:                                                      *)
(*    (a) a mutual inductive group different from the rose/M-tree        *)
(*        families — here neither type is a "list of the other";        *)
(*        STree references DTree as a sub-tree, not as a container,    *)
(*    (b) a Dfix mutual fixpoint that accumulates a remapped nat across *)
(*        both packets.                                                  *)
(* ------------------------------------------------------------------ *)

(** [STree] and [DTree] are a mutually-defined syntax/declaration tree pair
    used by [stree_size] and [dtree_size]. *)
Inductive STree :=
  | SLit : nat -> STree
  | SSeq : DTree -> STree -> STree
with DTree :=
  | DEnd  : DTree
  | DDecl : STree -> DTree -> DTree.

(** [stree_size]: count the number of [SLit] leaf nodes in an [STree].
    The mutual partner [dtree_size] accumulates the same count across
    all declarations in a [DTree].  Together they form a Dfix block
    that pattern-matches on both packets.
    With nat → int remapping:
      stree_size (SLit 42)                                = 1
      stree_size (SSeq DEnd (SLit 0))                    = 1
      stree_size (SSeq (DDecl (SLit 1) (DDecl (SLit 2) DEnd)) (SLit 3)) = 3 *)
Fixpoint stree_size (s : STree) : nat :=
  match s with
  | SLit _   => 1
  | SSeq d b => dtree_size d + stree_size b
  end
with dtree_size (d : DTree) : nat :=
  match d with
  | DEnd      => 0
  | DDecl s t => stree_size s + dtree_size t
  end.

Python File Extraction datatypes "mylist_is_empty bintree_is_leaf roseforest_is_empty mforest_is_empty myopt_flatten color_is_red even_depth is_even ntree_is_leaf stree_size".
