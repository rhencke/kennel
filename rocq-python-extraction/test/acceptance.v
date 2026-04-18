(** Phase 1 acceptance test.
    After [Python Extraction foo.] we expect a [foo.py] file whose
    content starts with the preamble and contains [# UNIMPL Dterm]
    for the constant definition. *)

Declare ML Module "rocq-python-extraction".

(** Simplest possible definition — a constant of type [nat]. *)
Definition foo := 1.

(** Run extraction.  This writes [foo.py] in the working directory. *)
Python Extraction foo.
