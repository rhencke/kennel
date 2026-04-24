Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

Extract Inductive nat => "int"
  [ "0" "(lambda x: x + 1)" ]
  "(lambda fO, fS, n: fO() if n == 0 else fS(n - 1))".

Parameter source_map_boom : nat -> nat.
Extract Constant source_map_boom =>
  "(lambda _n: (_ for _ in ()).throw(RuntimeError(""boom"")))".

(** [source_map_runtime_error] calls a realized Python exception function so
    traceback annotation can map the runtime failure back to Rocq source. *)
Definition source_map_runtime_error (n : nat) : nat := source_map_boom n.

Python Extraction source_map_runtime_error.
