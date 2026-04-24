Declare ML Module "rocq-python-extraction".

Set Universe Polymorphism.

(** [poly_id] is universe-polymorphic identity; extraction should erase the
    universe annotation and keep the value-level argument. *)
Polymorphic Definition poly_id@{u} {A : Type@{u}} (x : A) : A := x.

Python Extraction poly_id.
