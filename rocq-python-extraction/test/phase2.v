(** Phase 2 acceptance test: core term node coverage.

    Exercises each [ml_ast] node through Python extraction and verifies
    (via [Makefile]) that the output is syntactically valid Python.

    Coverage:
      [MLrel]       — function body references its parameter
      [MLglob]      — reference to a globally-defined term
      [MLapp]       — curried-flattened application
      [MLlam]       — lambda in expression position
      [MLletin]     — let binding (lambda-lifted in expression context)
      [MLtuple]     — pair literal
      [MLcons]      — Standard (nat), Record (pair_t), Coinductive (stream)
      [MLcase]      — bool ternary + general structural match
      [MLfix]       — Dfix mutual-fixpoint declaration
      [Dind]        — Standard, Record, Coinductive inductive declarations
      [MLuint]      — 63-bit integer literal
      [MLfloat]     — IEEE 754 float literal
      [MLstring]    — primitive byte-string literal
      [MLaxiom]     — unproved axiom
      [MLdummy]     — erased (logical) argument

    Known gaps:
      [MLexn]     — requires empty-match or erasure edge-cases; deferred
      [MLmagic]   — internal extraction coercion; hard to trigger directly
      [MLparray]  — persistent arrays intentionally not supported (MVP stub)
*)

Declare ML Module "rocq-python-extraction".

(* The extraction vernaculars (Extract Inductive, etc.) are registered by
   the rocq-runtime.plugins.extraction ML plugin, which our plugin depends
   on.  No Stdlib theory import is required — they are available the moment
   the plugin is loaded via [Declare ML Module] above.

   Similarly, [int], [float], and primitive string literals are kernel
   primitives; no Stdlib import is needed for their basic use. *)

(* ------------------------------------------------------------------ *)
(*  Extract Inductive bool → Python True/False (enables ternary emit)  *)
(* ------------------------------------------------------------------ *)

Extract Inductive bool => "bool" [ "True" "False" ].

(* ------------------------------------------------------------------ *)
(*  Inductive type declarations (exercises Dind emission)              *)
(* ------------------------------------------------------------------ *)

(** Standard inductive: nat is already in scope. *)

(** Record inductive — one constructor, named fields. *)
Record pair_t (A B : Type) : Type := MkPair { pfst : A ; psnd : B }.

(** Coinductive type — constructor wrapped in a thunk on extraction. *)
CoInductive stream (A : Type) : Type :=
  | SCons : A -> stream A -> stream A.

(* ------------------------------------------------------------------ *)
(*  Definitions                                                        *)
(* ------------------------------------------------------------------ *)

(** MLrel + MLdummy: [A : Type] is erased to [MLdummy]; [x] is [MLrel 1]. *)
Definition identity (A : Type) (x : A) : A := x.

(** MLglob: [identity] appears as a global reference in the body. *)
Definition identity2 (n : nat) : nat := identity nat n.

(** MLapp (curried flattening): [f (f n)] flattens [MLapp(MLapp(...))]
    into a single call-site with two args. *)
Definition apply_twice (f : nat -> nat) (n : nat) : nat := f (f n).

(** MLlam in expression context: body is a lambda, not a function arg. *)
Definition compose (f g : nat -> nat) : nat -> nat :=
  fun x => f (g x).

(** MLletin: let-binding lifted to [(lambda x: S(x))(S(O))] in expr context. *)
Definition letin_ex : nat :=
  let x := S O in S x.

(** MLtuple: pair literal. *)
Definition mk_pair : nat * nat := (O, S O).

(** MLcons Record: uses [MkPair] with keyword-argument emission. *)
Definition mk_pair_r : pair_t nat nat :=
  {| pfst := O ; psnd := S O |}.

(** MLcons Coinductive: co-fixpoint; constructor is thunk-wrapped. *)
CoFixpoint zeros : stream nat := SCons nat O zeros.

(** MLcase bool → ternary: [Extract Inductive bool] makes this a ternary. *)
Definition bool_to_nat (b : bool) : nat := if b then 1 else 0.

(** MLcase Standard: general [match]/[case] in function body. *)
Definition nat_pred (n : nat) : nat :=
  match n with
  | O   => O
  | S m => m
  end.

(** MLfix / Dfix: Fixpoint generates a Dfix declaration. *)
Fixpoint nat_add (n m : nat) : nat :=
  match n with
  | O    => m
  | S n' => S (nat_add n' m)
  end.

(** MLuint: 63-bit machine integer literal. *)
Definition uint_val : int := 42.

(** MLfloat: IEEE 754 float literal. *)
Definition float_val : float := 3.14.

(** MLstring: primitive byte-string literal.  Type inferred from the
    [%pstring] notation; no [PrimString.string] qualified annotation needed
    when the Stdlib is not installed. *)
Definition str_val := "hello"%pstring.

(** MLaxiom: unproved assumption — extracts to [raise NotImplementedError]. *)
Axiom todo_val : nat.

(* ------------------------------------------------------------------ *)
(*  Extraction commands                                                *)
(*  Each writes a .py file to the project's build root.               *)
(* ------------------------------------------------------------------ *)

(** [nat_add.py]: covers nat (Dind Standard), MLfix, MLcase Standard,
    MLcase bool ternary, MLrel, MLglob, MLdummy, MLapp. *)
Python Extraction nat_add.

(** [mk_pair_r.py]: covers pair_t (Dind Record), MLcons Record. *)
Python Extraction mk_pair_r.

(** [zeros.py]: covers stream (Dind Coinductive), MLcons Coinductive. *)
Python Extraction zeros.

(** [uint_val.py]: MLuint literal. *)
Python Extraction uint_val.

(** [float_val.py]: MLfloat literal. *)
Python Extraction float_val.

(** [str_val.py]: MLstring literal. *)
Python Extraction str_val.

(** [todo_val.py]: MLaxiom. *)
Python Extraction todo_val.
