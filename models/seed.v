(** Seed model: minimal extraction to verify the pipeline end-to-end.

    This file exercises the full [./fido make-rocq] path:
      1. Plugin loaded via [Declare ML Module].
      2. A simple function defined and extracted to Python.
      3. Output deposited in [src/fido/rocq/seed.py].

    This file stays as a smoke-test that the pipeline is wired up before
    larger coordination models are extracted. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

(** Map [bool] to Python [bool] so the output is idiomatic Python. *)
Extract Inductive bool => "bool" [ "True" "False" ].

(** [bool_not]: logical negation.
    Extracted form: [False if b else True] — a one-liner that exercises
    the bool ternary path and confirms the full pipeline produces
    syntactically valid, ruff-formatted Python. *)
Definition bool_not (b : bool) : bool := if b then false else true.

Python Extraction bool_not.
