(** Polymorphism acceptance tests: rank-1 polymorphism with typed Python output.

    Acceptance: a polymorphic list-map function extracts with real TypeVars
    that pyright verifies. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

Open Scope list_scope.

Extract Inductive list => "list"
  [ "[]" "(lambda h, t: [h] + t)" ]
  "(lambda fnil, fcons, xs: fnil() if xs == [] else fcons(xs[0], xs[1:]))".

(** [list_map] is a rank-1 polymorphic list map.  The generated Python should
    expose TypeVars for [A] and [B] and pyright-check successfully. *)
Fixpoint list_map {A B : Set} (f : A -> B) (xs : list A) : list B :=
  match xs with
  | nil => nil
  | x :: tl => f x :: list_map f tl
  end.

(* ------------------------------------------------------------------ *)
(*  Grouped extraction                                                 *)
(*  All definitions land in a single [polymorphism.py] module.       *)
(* ------------------------------------------------------------------ *)

(** [polymorphism.py]: covers rank-1 polymorphism with TypeVar output. *)
Python File Extraction polymorphism "list_map".
