(** Umbrella acceptance theory for Rocq -> Python extraction.

    The feature slices live in separate files because they intentionally use
    different extraction remappings for the same Rocq primitives.  This file
    provides one stable build entrypoint for the whole acceptance lane. *)

From PyExtractTest Require Export
  core_terms
  primitives
  datatypes
  nested_patterns
  records
  wf_recursion
  polymorphism
  universe_erasure
  prop_set_discipline
  monads
  coinductives
  modules
  source_maps
  strings_bytes
  numbers.
