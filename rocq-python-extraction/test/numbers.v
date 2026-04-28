(** Built-in numeric remapping acceptance tests.

    This file intentionally has no local [Extract Inductive] pragmas for
    [nat], [positive], [N], [Z], or [Q].  The Python backend owns those
    Stdlib mappings directly. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import
  Bool.Bool
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

(** [positive_eq] lowers [Pos.eqb] to native Python equality without retaining
    a generated [Pos] protocol module. *)
Definition positive_eq (left right : positive) : bool :=
  Pos.eqb left right.

(** [nat_compare_and] keeps primitive comparisons as high-precedence children
    of a lowered boolean conjunction. *)
Definition nat_compare_and (left middle right : nat) : bool :=
  andb (Nat.ltb left middle) (Nat.leb middle right).

(** [nat_compare_or] keeps primitive comparisons as high-precedence children
    of a lowered boolean disjunction. *)
Definition nat_compare_or (left middle right : nat) : bool :=
  orb (Nat.eqb left middle) (Nat.ltb middle right).

(** [nat_compare_neg] parenthesizes a primitive comparison under lowered
    boolean negation. *)
Definition nat_compare_neg (left right : nat) : bool :=
  negb (Nat.leb left right).

(** [nat_compare_bool_eq] parenthesizes a primitive comparison when it becomes
    the left operand of another lowered equality expression. *)
Definition nat_compare_bool_eq (left right : nat) (expected : bool) : bool :=
  Bool.eqb (Nat.ltb left right) expected.

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
Python Extraction positive_eq.
Python Extraction nat_compare_and.
Python Extraction nat_compare_or.
Python Extraction nat_compare_neg.
Python Extraction nat_compare_bool_eq.
Python Extraction n_seven.
Python Extraction n_case.
Python Extraction z_neg_three.
Python Extraction z_sign_code.
Python Extraction q_half.
Python Extraction q_num.
Python Extraction q_den.
