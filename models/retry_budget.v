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

(** [max_retries] mirrors the current retry ceiling used by the Python event
    handling path.  It is a constant model value, so extraction must emit a
    plain Python integer rather than a Peano tree. *)
Definition max_retries : nat := S (S (S O)).

(** [retry_budget] is a small [N] fixture for non-negative retry counters.
    It exercises the backend-owned [N] remapping used by runtime counters. *)
Definition retry_budget : N := Npos (xI xH).

(** [retry_delta] converts a remaining retry budget into a signed delta:
    exhausted budgets become [-1], and positive budgets preserve their
    positive payload as a [Z]. *)
Definition retry_delta (remaining : N) : Z :=
  match remaining with
  | N0 => Zneg xH
  | Npos p => Zpos p
  end.

(** [backoff_ratio] is a rational fixture for exact fractional values.
    It ensures [Q] extraction uses [fractions.Fraction], not Python float. *)
Definition backoff_ratio : Q := Qmake (Zpos xH) (xO xH).

Python File Extraction retry_budget
  "max_retries retry_budget retry_delta backoff_ratio".
