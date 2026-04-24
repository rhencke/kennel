Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

(** Directly unwrapping IO inside generated sync code would hide effects.
    The backend must reject this marker with a structured diagnostic. *)
Definition IO (A : Type) : Type := A.

Parameter io_run : forall {A : Type}, IO A -> A.

Definition bad_io_unwrap : nat := io_run 0.

Extract Constant IO "'a" => "__PYMONAD_IO_TYPE__".
Extract Constant io_run => "__PYMONAD_IO_RUN__".

Python Extraction bad_io_unwrap.
