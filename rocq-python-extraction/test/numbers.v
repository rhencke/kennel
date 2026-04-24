(** Built-in numeric remapping acceptance tests.

    This file intentionally has no local [Extract Inductive] pragmas for
    [nat], [positive], [N], [Z], or [Q].  The Python backend owns those
    Stdlib mappings directly. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import
  Numbers.BinNums
  NArith.BinNat
  ZArith.BinInt
  QArith.QArith_base.

(** [nat_three] checks constant Peano naturals become Python ints. *)
Definition nat_three : nat := S (S (S O)).

(** [nat_pred_or_zero] checks nat pattern matching over the backend-owned
    nat remapping. *)
Definition nat_pred_or_zero (n : nat) : nat :=
  match n with
  | O => O
  | S m => m
  end.

(** [nat_roundtrip] recursively rebuilds its nat argument, covering recursive
    calls under the primitive nat remapping. *)
Fixpoint nat_roundtrip (n : nat) : nat :=
  match n with
  | O => O
  | S m => S (nat_roundtrip m)
  end.

(** [positive_five] checks positive literals become Python positive ints. *)
Definition positive_five : positive := xI (xO xH).

(** [positive_case] distinguishes [xH], even positives, and odd positives. *)
Definition positive_case (p : positive) : nat :=
  match p with
  | xH => O
  | xO _ => S (S O)
  | xI _ => S O
  end.

(** [n_seven] checks [N] literals become Python non-negative ints. *)
Definition n_seven : N := Npos (xI (xI xH)).

(** [n_case] distinguishes zero from positive [N] values. *)
Definition n_case (n : N) : nat :=
  match n with
  | N0 => O
  | Npos _ => S O
  end.

(** [z_neg_three] checks negative [Z] literals become Python negative ints. *)
Definition z_neg_three : Z := Zneg (xI xH).

(** [z_sign_code] reduces a [Z] to zero, +1, or -1. *)
Definition z_sign_code (z : Z) : Z :=
  match z with
  | Z0 => Z0
  | Zpos _ => Zpos xH
  | Zneg _ => Zneg xH
  end.

(** [q_half] checks exact rational constants become [Fraction] values. *)
Definition q_half : Q := Qmake (Zpos xH) (xO xH).

(** [q_num] projects the signed numerator from a rational. *)
Definition q_num (q : Q) : Z :=
  match q with
  | Qmake n _ => n
  end.

(** [q_den] projects the positive denominator from a rational. *)
Definition q_den (q : Q) : positive :=
  match q with
  | Qmake _ d => d
  end.

Python Extraction nat_three.
Python Extraction nat_pred_or_zero.
Python Extraction nat_roundtrip.
Python Extraction positive_five.
Python Extraction positive_case.
Python Extraction n_seven.
Python Extraction n_case.
Python Extraction z_neg_three.
Python Extraction z_sign_code.
Python Extraction q_half.
Python Extraction q_num.
Python Extraction q_den.
