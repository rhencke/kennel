Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

Extract Inductive bool => "bool" [ "True" "False" ].

(** [eq_dec_bool] consumes a proof-carrying sum and keeps only the
    computational constructor choice. *)
Definition eq_dec_bool (d : {0 = 0} + {0 <> 0}) : bool :=
  match d with
  | left _ => true
  | right _ => false
  end.

(** [proof_pair_zero] pairs a computational nat with a proof in [Prop].
    The legacy Python contract treats that proof component as impossible at
    runtime, so evaluating the extracted pair raises the generated
    [_Impossible] witness instead of fabricating proof data. *)
Definition proof_pair_zero (_ : nat) : nat * (0 = 0) :=
  (0, eq_refl).

Python Extraction eq_dec_bool.
Python Extraction proof_pair_zero.
