(** Concurrency primitive extraction acceptance tests.

    These fixtures expose explicit Python [threading] wrappers at IO
    boundaries.  They intentionally do not model Python scheduler fairness or
    arbitrary thread interleavings in Rocq. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

(** [IO] is the effect boundary shared with the IO acceptance fixture. *)
Definition IO (A : Type) : Type := A.

(** [io_pure] injects a pure value into the IO wrapper. *)
Definition io_pure {A : Type} (value : A) : IO A := value.

(** [io_bind] sequences IO actions. *)
Definition io_bind {A B : Type} (action : IO A) (next : A -> IO B) : IO B :=
  next action.

(** [io_bracket] acquires an owner token, runs an IO action, and releases the
    owner even if the action fails at runtime. *)
Definition io_bracket {Owner A : Type}
    (acquire : IO Owner)
    (release : Owner -> IO unit)
    (use : Owner -> IO A) : IO A :=
  io_bind acquire (fun owner =>
  io_bind (use owner) (fun value =>
  io_bind (release owner) (fun _ =>
  io_pure value))).

(** [Mutex] is an opaque mutual-exclusion handle supplied by Python. *)
Definition Mutex : Type := unit.

(** [Channel] is an opaque FIFO message handle supplied by Python. *)
Definition Channel (A : Type) : Type := list A.

(** [Future] is an opaque single-assignment result handle supplied by Python. *)
Definition Future (A : Type) : Type := option A.

(** [new_mutex] allocates a Python mutex at the IO boundary. *)
Parameter new_mutex : IO Mutex.

(** [new_channel] allocates a Python channel at the IO boundary. *)
Parameter new_channel : forall {A : Type}, IO (Channel A).

(** [new_future] allocates a Python future at the IO boundary. *)
Parameter new_future : forall {A : Type}, IO (Future A).

(** [mutex_acquire] blocks until the mutex is held. *)
Parameter mutex_acquire : Mutex -> IO unit.

(** [mutex_release] releases a held mutex. *)
Parameter mutex_release : Mutex -> IO unit.

(** [channel_send] sends one message to the channel. *)
Parameter channel_send : forall {A : Type}, Channel A -> A -> IO unit.

(** [channel_receive] receives one message from the channel. *)
Parameter channel_receive : forall {A : Type}, Channel A -> IO A.

(** [future_set] completes a future. *)
Parameter future_set : forall {A : Type}, Future A -> A -> IO unit.

(** [future_result] waits for a completed future result. *)
Parameter future_result : forall {A : Type}, Future A -> IO A.

(** [future_done] observes whether a future is complete. *)
Parameter future_done : forall {A : Type}, Future A -> bool.

(** [lock_channel_future_demo] runs a deterministic lock/channel/future
    boundary sequence and returns the value that crossed those boundaries. *)
Definition lock_channel_future_demo : IO nat :=
  io_bind new_mutex (fun mutex =>
  io_bracket (mutex_acquire mutex) (fun _ => mutex_release mutex) (fun _ =>
  io_bind (@new_channel nat) (fun channel =>
  io_bind (channel_send channel 7) (fun _ =>
  io_bind (channel_receive channel) (fun value =>
  io_bind (@new_future nat) (fun future =>
  io_bind (future_set future value) (fun _ =>
  io_bind (future_result future) (fun result =>
  io_pure result)))))))).

(** [future_done_after_set] covers the pure future observation wrapper. *)
Definition future_done_after_set : IO bool :=
  io_bind (@new_future nat) (fun future =>
  io_bind (future_set future 1) (fun _ =>
  io_pure (future_done future))).

Extract Inductive unit => "None" [ "None" ].
Extract Constant IO "'a" => "__PYMONAD_IO_TYPE__".
Extract Constant io_pure => "__PYMONAD_IO_PURE__".
Extract Constant io_bind => "__PYMONAD_IO_BIND__".
Extract Constant io_bracket => "__PYMONAD_IO_BRACKET__".
Extract Constant Mutex => "__PYCONC_MUTEX_TYPE__".
Extract Constant Channel "'a" => "__PYCONC_CHANNEL_TYPE__".
Extract Constant Future "'a" => "__PYCONC_FUTURE_TYPE__".
Extract Constant new_mutex => "__PYCONC_NEW_MUTEX__".
Extract Constant new_channel => "__PYCONC_NEW_CHANNEL__".
Extract Constant new_future => "__PYCONC_NEW_FUTURE__".
Extract Constant mutex_acquire => "__PYCONC_MUTEX_ACQUIRE__".
Extract Constant mutex_release => "__PYCONC_MUTEX_RELEASE__".
Extract Constant channel_send => "__PYCONC_CHANNEL_SEND__".
Extract Constant channel_receive => "__PYCONC_CHANNEL_RECEIVE__".
Extract Constant future_set => "__PYCONC_FUTURE_SET__".
Extract Constant future_result => "__PYCONC_FUTURE_RESULT__".
Extract Constant future_done => "__PYCONC_FUTURE_DONE__".

Python Extraction lock_channel_future_demo.
Python Extraction future_done_after_set.
