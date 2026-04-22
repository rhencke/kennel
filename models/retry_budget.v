(** Retry and budget numeric model.

    This model exercises backend-owned Python mappings for [nat], [N], [Z],
    and [Q] without local extraction pragmas. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import
  Numbers.BinNums
  NArith.BinNat
  ZArith.BinInt
  QArith.QArith_base.

Definition max_retries : nat := S (S (S O)).

Definition retry_budget : N := Npos (xI xH).

Definition retry_delta (remaining : N) : Z :=
  match remaining with
  | N0 => Zneg xH
  | Npos p => Zpos p
  end.

Definition backoff_ratio : Q := Qmake (Zpos xH) (xO xH).

Python Extraction max_retries.
Python Extraction retry_budget.
Python Extraction retry_delta.
Python Extraction backoff_ratio.
