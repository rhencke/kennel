(** Negative coverage for classical real extraction.

    Rocq [R] does not have a faithful Python runtime representation, so the
    backend must fail with a structured diagnostic instead of silently choosing
    [float]. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import Reals.Rdefinitions.

(** [real_identity] is intentionally unsupported: extracting it must emit the
    structured PYEX041 diagnostic instead of choosing an unsound float model. *)
Definition real_identity (r : R) : R := r.

Python Extraction real_identity.
