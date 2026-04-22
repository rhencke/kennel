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

Definition nat_three : nat := S (S (S O)).

Definition nat_pred_or_zero (n : nat) : nat :=
  match n with
  | O => O
  | S m => m
  end.

Fixpoint nat_roundtrip (n : nat) : nat :=
  match n with
  | O => O
  | S m => S (nat_roundtrip m)
  end.

Definition positive_five : positive := xI (xO xH).

Definition positive_case (p : positive) : nat :=
  match p with
  | xH => O
  | xO _ => S (S O)
  | xI _ => S O
  end.

Definition n_seven : N := Npos (xI (xI xH)).

Definition n_case (n : N) : nat :=
  match n with
  | N0 => O
  | Npos _ => S O
  end.

Definition z_neg_three : Z := Zneg (xI xH).

Definition z_sign_code (z : Z) : Z :=
  match z with
  | Z0 => Z0
  | Zpos _ => Zpos xH
  | Zneg _ => Zneg xH
  end.

Definition q_half : Q := Qmake (Zpos xH) (xO xH).

Definition q_num (q : Q) : Z :=
  match q with
  | Qmake n _ => n
  end.

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
