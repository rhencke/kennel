(** Unsupported concurrency scheduling fixture.

    Arbitrary thread interleavings are not executable Python semantics for this
    backend.  They must fail with a structured diagnostic instead. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

Definition IO (A : Type) : Type := A.

Parameter interleave : forall {A : Type}, IO A -> IO A -> IO A.

Definition bad_interleave : nat := interleave 1 2.

Extract Constant IO "'a" => "__PYMONAD_IO_TYPE__".
Extract Constant interleave => "__PYCONC_INTERLEAVE__".

Python Extraction bad_interleave.
