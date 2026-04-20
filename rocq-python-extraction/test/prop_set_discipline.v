Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

Extract Inductive bool => "bool" [ "True" "False" ].

Definition eq_dec_bool (d : {0 = 0} + {0 <> 0}) : bool :=
  match d with
  | left _ => true
  | right _ => false
  end.

Definition proof_pair_zero (_ : nat) : nat * (0 = 0) :=
  (0, eq_refl).

Python Extraction eq_dec_bool.
Python Extraction proof_pair_zero.
