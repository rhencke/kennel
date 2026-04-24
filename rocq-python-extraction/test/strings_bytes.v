(** Built-in string/byte UTF-8 remapping acceptance tests.

    This file intentionally has no local [Extract Inductive] pragmas for
    [string], [ascii], or [byte].  The Python backend owns those Stdlib
    mappings directly. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import
  Strings.String
  Strings.Ascii
  Strings.Byte
  Strings.PrimString.

Open Scope string_scope.
Open Scope byte_scope.

(** [github_key] is a simple Rocq [string] literal fixture. *)
Definition github_key : String.string := "pull_request".

(** [payload_fragment] checks primitive byte-string literal extraction. *)
Definition payload_fragment := "pull_request"%pstring.

(** [ascii_A] spells the byte bits for ASCII [A]. *)
Definition ascii_A : ascii :=
  Ascii true false false false false false true false.

(** [byte_lf] is the line-feed [byte] constructor. *)
Definition byte_lf : byte := x0a.

(** [first_ascii_or_A] returns the first character of a string, or [ascii_A]
    when the string is empty. *)
Definition first_ascii_or_A (s : String.string) : ascii :=
  match s with
  | EmptyString => ascii_A
  | String c _ => c
  end.

(** [tail_or_empty] drops the first character when present. *)
Definition tail_or_empty (s : String.string) : String.string :=
  match s with
  | EmptyString => EmptyString
  | String _ rest => rest
  end.

(** [ascii_roundtrip] reconstructs an [ascii] value from its eight bits. *)
Definition ascii_roundtrip (a : ascii) : ascii :=
  match a with
  | Ascii b0 b1 b2 b3 b4 b5 b6 b7 =>
      Ascii b0 b1 b2 b3 b4 b5 b6 b7
  end.

(** [byte_label] checks byte constructor matching, including a wildcard arm. *)
Definition byte_label (b : byte) : String.string :=
  match b with
  | x0a => "lf"
  | _ => "other"
  end.

Python Extraction github_key.
Python Extraction payload_fragment.
Python Extraction ascii_A.
Python Extraction byte_lf.
Python Extraction first_ascii_or_A.
Python Extraction tail_or_empty.
Python Extraction ascii_roundtrip.
Python Extraction byte_label.
