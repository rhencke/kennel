Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

Extract Inductive nat => "int"
  [ "0" "(lambda x: x + 1)" ]
  "(lambda fO, fS, n: fO() if n == 0 else fS(n - 1))".

(** [ModuleLookupFixture] groups the module-system acceptance fixture under one module
    so [Python Module Extraction] can emit a namespace-like Python object. *)
Module ModuleLookupFixture.
(** [MapSig] is the small module type consumed by the lookup functors. *)
  Module Type MapSig.
(** [missing] is the default key/value supplied by a map implementation. *)
    Parameter missing : nat.
(** [lookup] maps a natural key to a natural result. *)
    Parameter lookup : nat -> nat.
  End MapSig.

(** [NatMap] implements [MapSig] with identity lookup and missing value [0]. *)
  Module NatMap <: MapSig.
(** [missing] chooses zero for the identity map fixture. *)
    Definition missing : nat := 0.
(** [lookup] returns the input unchanged. *)
    Definition lookup (n : nat) : nat := n.
  End NatMap.

(** [SuccMap] implements [MapSig] with successor lookup and missing value [1]. *)
  Module SuccMap <: MapSig.
(** [missing] chooses one for the successor map fixture. *)
    Definition missing : nat := 1.
(** [lookup] returns the successor of the input. *)
    Definition lookup (n : nat) : nat := S n.
  End SuccMap.

(** [MakeLookup] is an applicative functor that runs [lookup] on [missing]. *)
  Module MakeLookup (X : MapSig).
(** [run] is the extracted value produced by applying [X.lookup] to
    [X.missing]. *)
    Definition run : nat := X.lookup X.missing.
  End MakeLookup.

(** [NatLookup] applies [MakeLookup] to [NatMap]. *)
  Module NatLookup := MakeLookup NatMap.
(** [NatLookupAgain] repeats the same applicative functor application so the
    backend can reuse the module cache. *)
  Module NatLookupAgain := MakeLookup NatMap.
(** [SuccLookup] applies [MakeLookup] to [SuccMap]. *)
  Module SuccLookup := MakeLookup SuccMap.

(** [FreshLookupAFunctor] has the same body as [MakeLookup] but a distinct
    functor identity, so it should get its own cache. *)
  Module FreshLookupAFunctor (X : MapSig).
(** [run] mirrors [MakeLookup.run] for the fresh A functor. *)
    Definition run : nat := X.lookup X.missing.
  End FreshLookupAFunctor.

(** [FreshLookupBFunctor] is another distinct functor with the same body. *)
  Module FreshLookupBFunctor (X : MapSig).
(** [run] mirrors [MakeLookup.run] for the fresh B functor. *)
    Definition run : nat := X.lookup X.missing.
  End FreshLookupBFunctor.

(** [FreshLookupA] applies the A functor to [NatMap]. *)
  Module FreshLookupA := FreshLookupAFunctor NatMap.
(** [FreshLookupB] applies the B functor to [NatMap]. *)
  Module FreshLookupB := FreshLookupBFunctor NatMap.
End ModuleLookupFixture.

Python Module Extraction ModuleLookupFixture.
