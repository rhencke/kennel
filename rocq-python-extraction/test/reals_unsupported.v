(** Negative coverage for classical real extraction.

    Rocq [R] does not have a faithful Python runtime representation, so the
    backend must fail with a structured diagnostic instead of silently choosing
    [float]. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import Reals.Rdefinitions.

Definition real_identity (r : R) : R := r.

Python Extraction real_identity.
