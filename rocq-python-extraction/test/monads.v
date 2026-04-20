Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

Open Scope list_scope.

Extract Inductive nat => "int"
  [ "0" "(lambda x: x + 1)" ]
  "(lambda fO, fS, n: fO() if n == 0 else fS(n - 1))".

Extract Inductive unit => "None" [ "None" ].
Extract Inductive option => ""
  [ "" "None" ]
  "(lambda fSome, fNone, x: fNone() if x is None else fSome(x))".

Definition StateT (S A : Type) : Type := S -> (A * S).

Definition state_pure {S A : Type} (x : A) : StateT S A :=
  fun s => (x, s).

Definition state_bind {S A B : Type}
    (m : StateT S A)
    (f : A -> StateT S B) : StateT S B :=
  fun s =>
    let '(x, s') := m s in
    f x s'.

Definition get_state {S : Type} : StateT S S :=
  fun s => (s, s).

Definition put_state {S : Type} (s' : S) : StateT S unit :=
  fun _ => (tt, s').

Definition tick : StateT nat nat :=
  state_bind get_state (fun n =>
  state_bind (put_state (S n)) (fun _ =>
  state_pure n)).

Definition option_bind {A B : Type}
    (o : option A)
    (f : A -> option B) : option B :=
  match o with
  | None => None
  | Some x => f x
  end.

Definition option_chain (o : option nat) : option nat :=
  option_bind o (fun n => Some (S n)).

Extract Constant StateT "'s" "'a" => "StateT".
Extract Constant state_pure => "__PYMONAD_STATE_PURE__".
Extract Constant state_bind => "__PYMONAD_STATE_BIND__".
Extract Constant get_state => "__PYMONAD_STATE_GET__".
Extract Constant put_state => "__PYMONAD_STATE_PUT__".
Extract Constant option_bind => "__PYMONAD_OPTION_BIND__".

Python Extraction tick.
Python Extraction option_chain.
