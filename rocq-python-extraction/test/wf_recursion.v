(** Well-founded recursion / Program Fixpoint acceptance tests
    accessibility erasure (issue #721).

    A function defined through [Program Fixpoint] with a termination proof must
    extract to idiomatic Python with only computational parameters visible.
    In particular, generated Python should look like:

      def wf_countdown(x):
          ...

    not like:

      def wf_countdown(x, _acc, _dummy):
          ...
*)

Declare ML Module "rocq-python-extraction".

(* [Program Fixpoint] and [Extract Inductive] need their plugins. *)
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import Program.Wf.
From Stdlib Require Import Arith.Wf_nat.

(* Remap nat -> int so the extracted checker can call the function directly
   with Python ints. *)
Extract Inductive nat => "int"
  [ "0" "(lambda x: x + 1)" ]
  "(lambda fO, fS, n: fO() if n == 0 else fS(n - 1))".

(** [wf_countdown] is defined with a termination proof.  Extraction must erase
    proof/accessibility arguments and leave only the computational [n]. *)
Program Fixpoint wf_countdown (n : nat) {measure n} : nat :=
  match n with
  | O => O
  | S n' => S (wf_countdown n')
  end.

Python Extraction wf_countdown.
