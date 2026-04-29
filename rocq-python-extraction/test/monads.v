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

(** [StateT] is the state monad representation used by the custom monad
    extraction markers below. *)
Definition StateT (S A : Type) : Type := S -> (A * S).

(** [state_pure] injects a value into the state monad without changing state. *)
Definition state_pure {S A : Type} (x : A) : StateT S A :=
  fun s => (x, s).

(** [state_bind] sequences a state action and feeds its result to the next
    state action. *)
Definition state_bind {S A B : Type}
    (m : StateT S A)
    (f : A -> StateT S B) : StateT S B :=
  fun s =>
    let '(x, s') := m s in
    f x s'.

(** [get_state] returns the current state as the monadic value. *)
Definition get_state {S : Type} : StateT S S :=
  fun s => (s, s).

(** [put_state] overwrites the state and returns unit. *)
Definition put_state {S : Type} (s' : S) : StateT S unit :=
  fun _ => (tt, s').

(** [tick] reads a natural state, increments it, and returns the old value.
    It is the main state-monad extraction acceptance fixture. *)
Definition tick : StateT nat nat :=
  state_bind get_state (fun n =>
  state_bind (put_state (S n)) (fun _ =>
  state_pure n)).

(** [option_bind] is the option monad bind used by the custom option marker. *)
Definition option_bind {A B : Type}
    (o : option A)
    (f : A -> option B) : option B :=
  match o with
  | None => None
  | Some x => f x
  end.

(** [option_chain] increments a present natural and preserves [None]. *)
Definition option_chain (o : option nat) : option nat :=
  option_bind o (fun n => Some (S n)).

(** [option_chain_twice] nests one lowered option-bind expression inside
    another, covering bind-as-child precedence. *)
Definition option_chain_twice (o : option nat) : option nat :=
  option_bind
    (option_bind o (fun n => Some (S n)))
    (fun n => Some (S n)).

Extract Constant StateT "'s" "'a" => "StateT".
Extract Constant state_pure => "__PYMONAD_STATE_PURE__".
Extract Constant state_bind => "__PYMONAD_STATE_BIND__".
Extract Constant get_state => "__PYMONAD_STATE_GET__".
Extract Constant put_state => "__PYMONAD_STATE_PUT__".
Extract Constant option_bind => "__PYMONAD_OPTION_BIND__".

Python Extraction tick.
Python Extraction option_chain.
Python Extraction option_chain_twice.
