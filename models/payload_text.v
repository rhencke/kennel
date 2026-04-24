(** Text and byte payload model.

    This model exercises the backend-owned Python mappings for Rocq
    [string], [ascii], [byte], and primitive byte strings without local
    extraction pragmas. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import
  Strings.String
  Strings.Ascii
  Strings.Byte
  Strings.PrimString.

Open Scope string_scope.
Open Scope byte_scope.

(** [event_name] is the canonical GitHub event string exercised by this
    model.  It checks that Rocq [string] constants become Python [str]. *)
Definition event_name : String.string := "pull_request".

(** [event_name_bytes] is the same payload fragment as a primitive byte
    string, covering [%pstring] extraction to Python [bytes]. *)
Definition event_name_bytes := "pull_request"%pstring.

(** [newline_byte] is a concrete [byte] constructor fixture for byte remapping. *)
Definition newline_byte : byte := x0a.

(** [first_char_or_newline] returns the first character of [s], or a literal
    newline ASCII value for the empty string.  It exercises string pattern
    matching and [ascii] construction. *)
Definition first_char_or_newline (s : String.string) : ascii :=
  match s with
  | EmptyString => Ascii false true false true false false false false
  | String c _ => c
  end.

Python File Extraction payload_text
  "event_name event_name_bytes newline_byte first_char_or_newline".
