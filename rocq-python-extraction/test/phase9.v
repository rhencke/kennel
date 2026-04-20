Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

CoInductive stream (A : Type) : Type :=
  | SCons : A -> stream A -> stream A.

CoFixpoint zeros : stream nat := SCons nat O zeros.

Definition nat_id (n : nat) : nat := n.

CoFixpoint zeros_alt : stream nat := SCons nat (nat_id O) zeros_alt.

Definition zeros_pair : stream nat * stream nat := (zeros, zeros_alt).

CoInductive cotree : Type :=
  | CNode : nat -> cotree -> cotree -> cotree.

CoFixpoint repeat_tree : cotree := CNode O repeat_tree repeat_tree.

Definition tree_root (t : cotree) : nat :=
  match t with
  | CNode n _ _ => n
  end.

Definition tree_root_of_repeat : nat := tree_root repeat_tree.

Python Extraction zeros.
Python Extraction zeros_pair.
Python Extraction repeat_tree.
Python Extraction tree_root_of_repeat.
