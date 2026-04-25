(** IO boundary extraction acceptance tests.

    Coordination models should remain pure.  This fixture models adapter-facing
    effects only: Rocq describes sequencing through [IO], while Python supplies
    explicit boundary implementations through extraction remappings. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import Strings.String.

Open Scope string_scope.

(** [IO] is intentionally opaque at the Rocq boundary.  The Python backend maps
    it to an async-aware [IO] wrapper rather than pretending effects are pure. *)
Definition IO (A : Type) : Type := A.

(** [io_pure] injects a pure value into the IO wrapper. *)
Definition io_pure {A : Type} (value : A) : IO A := value.

(** [io_bind] sequences IO actions. *)
Definition io_bind {A B : Type} (action : IO A) (next : A -> IO B) : IO B :=
  next action.

(** [read_text] is a file-style adapter boundary supplied by Python. *)
Parameter read_text : String.string -> IO String.string.

(** [http_status] is an HTTP-style adapter boundary supplied by Python. *)
Parameter http_status : String.string -> IO nat.

(** [read_file_echo] sequences a file read and returns its text payload. *)
Definition read_file_echo (path : String.string) : IO String.string :=
  io_bind (read_text path) (fun contents => io_pure contents).

(** [http_status_ok] maps a boundary status code to a pure boolean result. *)
Definition http_status_ok (url : String.string) : IO bool :=
  io_bind (http_status url) (fun code =>
  io_pure
    (match code with
     | S (S O) => true
     | _ => false
     end)).

Extract Constant IO "'a" => "__PYMONAD_IO_TYPE__".
Extract Constant io_pure => "__PYMONAD_IO_PURE__".
Extract Constant io_bind => "__PYMONAD_IO_BIND__".
Extract Constant read_text =>
  "(lambda path: IO.from_sync(lambda: open(path, encoding=""utf-8"").read()))".
Extract Constant http_status =>
  "(lambda url: IO.from_sync(lambda: 2 if url == ""https://ok.example"" else 0))".

(* ------------------------------------------------------------------ *)
(*  Grouped extraction                                                 *)
(*  All definitions land in a single [io_boundaries.py] module.      *)
(* ------------------------------------------------------------------ *)

(** [io_boundaries.py]: covers [IO] sequencing via [io_bind]/[io_pure], file
    and HTTP adapter boundaries, and the async-facade wrapper pattern. *)
Python File Extraction io_boundaries "read_file_echo http_status_ok".
