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
      MyOpt A       — polymorphic option; option-of-option flatten *)

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

(* ------------------------------------------------------------------ *)
(*  1. Polymorphic singly-linked list                                   *)
(* ------------------------------------------------------------------ *)

Inductive MyList (A : Type) :=
  | MNil  : MyList A
  | MCons : A -> MyList A -> MyList A.

(** [mylist_is_empty]: [True] iff the list is [MNil].
    Exercises pattern matching on a parameterised non-remapped inductive
    and returning a remapped primitive (bool → Python bool). *)
Definition mylist_is_empty {A : Type} (l : MyList A) : bool :=
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

Inductive RoseTree (A : Type) :=
  | RNode : A -> RoseForest A -> RoseTree A
with RoseForest (A : Type) :=
  | RFNil  : RoseForest A
  | RFCons : RoseTree A -> RoseForest A -> RoseForest A.

(** [roseforest_is_empty]: [True] iff the forest has no trees.
    Exercises the second packet of a mutual parameterised inductive. *)
Definition roseforest_is_empty {A : Type} (f : RoseForest A) : bool :=
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

Inductive MyOpt (A : Type) :=
  | MyNone : MyOpt A
  | MySome : A -> MyOpt A.

(** [myopt_flatten]: flatten one level of option nesting.
    [MyNone] stays [MyNone]; [MySome x] unwraps to [x].
    Exercises: (a) constructing a zero-arg non-remapped dataclass in the
    return position ([MyNone()] rather than the class object [MyNone]),
    and (b) returning a value bound in the enclosing match arm. *)
Definition myopt_flatten {A : Type} (o : MyOpt (MyOpt A)) : MyOpt A :=
  match o with
  | MyNone   => MyNone
  | MySome x => x
  end.

Python Extraction myopt_flatten.
