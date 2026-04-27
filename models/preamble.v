(** Common extraction preamble shared by all Fido coordination models.

    Every [.v] file in [models/] loads this with:
      [From FidoModels Require Import preamble.]

    Three declarations are centralised here:

    1. [rocq-python-extraction] — the plugin that adds the [Python Extraction]
       vernacular used to emit [.py] files from MiniML terms.
    2. [rocq-runtime.plugins.extraction] — the standard Rocq extraction plugin
       that defines [Extract Inductive] and related pragmas.
    3. [Unset Universe Polymorphism] — disables sort-polymorphism so
       nullary-constructor extraction is clean.  See the note in
       [rocq-python-extraction/test/datatypes.v] for context.
    4. [Extract Inductive option] — remaps [Some x] to [x] and [None] to
       [None] so [option]-returning [transition] functions produce idiomatic
       Python return types. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

(* Prevent sort-polymorphism so nullary-constructor extraction is clean. *)
Unset Universe Polymorphism.

(* Remap [option] so [Some x] erases to [x] and [None] stays [None]. *)
Extract Inductive option => ""
  [ "" "None" ]
  "(lambda fSome, fNone, x: fNone() if x is None else fSome(x))".
