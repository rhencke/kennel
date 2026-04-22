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

Definition event_name : String.string := "pull_request".

Definition event_name_bytes := "pull_request"%pstring.

Definition newline_byte : byte := x0a.

Definition first_char_or_newline (s : String.string) : ascii :=
  match s with
  | EmptyString => Ascii false true false true false false false false
  | String c _ => c
  end.

Python Extraction event_name.
Python Extraction event_name_bytes.
Python Extraction newline_byte.
Python Extraction first_char_or_newline.
