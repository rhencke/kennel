Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

(** [stream] is an infinite coinductive sequence.  Extraction should represent
    recursive tails lazily enough that constructing a cyclic value terminates. *)
CoInductive stream (A : Type) : Type :=
  | SCons : A -> stream A -> stream A.

(** [zeros] is the canonical self-referential stream of zeroes. *)
CoFixpoint zeros : stream nat := SCons nat O zeros.

(** [nat_id] keeps a global function call inside [zeros_alt] so the fixture
    covers cofixpoints whose payload is not just a literal. *)
Definition nat_id (n : nat) : nat := n.

(** [zeros_alt] is a second cyclic stream used to test products of coinductive
    values. *)
CoFixpoint zeros_alt : stream nat := SCons nat (nat_id O) zeros_alt.

(** [zeros_pair] packages two coinductive values in one extracted tuple. *)
Definition zeros_pair : stream nat * stream nat := (zeros, zeros_alt).

(** [cotree] is an infinite binary tree shape with a value at each node. *)
CoInductive cotree : Type :=
  | CNode : nat -> cotree -> cotree -> cotree.

(** [repeat_tree] is a self-referential binary tree with every root value [O]. *)
CoFixpoint repeat_tree : cotree := CNode O repeat_tree repeat_tree.

(** [tree_root] forces only the root constructor of a coinductive tree. *)
Definition tree_root (t : cotree) : nat :=
  match t with
  | CNode n _ _ => n
  end.

(** [tree_root_of_repeat] proves the extracted tree can be observed without
    recursively forcing the whole infinite structure. *)
Definition tree_root_of_repeat : nat := tree_root repeat_tree.

Python Extraction zeros.
Python Extraction zeros_pair.
Python Extraction repeat_tree.
Python Extraction tree_root_of_repeat.
