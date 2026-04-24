(** Rocq → Python extraction backend.
    MiniML printer for extracted Python modules.

    Target: Python 3.14t (free-threaded build, no GIL).
    Generated code uses [match]/[case] (3.10+), PEP 604 union syntax (3.10+),
    and relies on PEP 649 deferred annotation evaluation so that forward
    references in dataclass fields work without quoting.  No attempt is made
    at compatibility with older Python versions. *)

open Pp
open Names
open Miniml
open Mlutil
open Modutil
open Table
open Common
open CErrors

(*s Keywords that may not be used as Python identifiers. *)

let keywords =
  let kws = [
    "False"; "None"; "True"; "and"; "as"; "assert"; "async"; "await";
    "break"; "class"; "continue"; "def"; "del"; "elif"; "else"; "except";
    "finally"; "for"; "from"; "global"; "if"; "import"; "in"; "is";
    "lambda"; "nonlocal"; "not"; "or"; "pass"; "raise"; "return"; "try";
    "while"; "with"; "yield";
    (* built-ins that clash in practice *)
    "type"; "list"; "dict"; "set"; "int"; "float"; "str"; "bool";
    "bytes"; "object"; "print"; "len"; "range"; "map"; "filter"; "zip";
  ] in
  List.fold_left (fun s w -> Id.Set.add (Id.of_string w) s) Id.Set.empty kws

(*s File naming: module path → base filename (no suffix). *)

let file_naming state mp =
  file_of_modfile (State.get_table state) mp

(*s Preamble emitted at the top of every extracted .py file. *)

let runtime_prelude = {|import asyncio
import queue
import threading
from contextlib import asynccontextmanager
from concurrent.futures import Future as _ConcurrentFuture
from itertools import islice
from dataclasses import dataclass
from fractions import Fraction
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Never,
    Protocol,
    TypeVar,
    assert_never,
    cast,
)

_CoForceT = TypeVar("_CoForceT")
_ModuleArgT = TypeVar("_ModuleArgT")
_ModuleRetT = TypeVar("_ModuleRetT")
_StateTState = TypeVar("_StateTState")
_StateTValue = TypeVar("_StateTValue")
_StateTNext = TypeVar("_StateTNext")
_IOValue = TypeVar("_IOValue")
_IONext = TypeVar("_IONext")
_IOOwner = TypeVar("_IOOwner")
_ChannelValue = TypeVar("_ChannelValue")
_FutureValue = TypeVar("_FutureValue")


class _Impossible(RuntimeError):
    pass


def _impossible() -> Never:
    raise _Impossible()


class _RocqUtf8BoundaryError(UnicodeError):
    pass


class _RocqNumericDomainError(ValueError):
    pass


def _rocq_numeric_domain_error(kind: str, value: object) -> Never:
    raise _RocqNumericDomainError(f"Rocq {kind} value out of domain: {value!r}")


key = int  # finite-map module key alias
elt = int  # finite-set module element alias


def _rocq_positive_key(key: int) -> int:
    if key <= 0:
        raise _RocqNumericDomainError("positive map/set key", key)
    return key


def _rocq_string_key(key: str) -> str:
    if not isinstance(key, str):
        raise TypeError(f"Rocq string map/set key must be str: {key!r}")
    return key


def _rocq_sorted_key(key: object) -> Any:
    return key.encode("utf-8") if isinstance(key, str) else key


def _rocq_map_add(
    key: object, value: object, mapping: dict[object, object]
) -> dict[object, object]:
    result = dict(mapping)
    result[key] = value
    return result


def _rocq_map_remove(
    key: object, mapping: dict[object, object]
) -> dict[object, object]:
    result = dict(mapping)
    result.pop(key, None)
    return result


def _rocq_map_elements(mapping: dict[object, object]) -> list[tuple[object, object]]:
    return sorted(mapping.items(), key=lambda item: _rocq_sorted_key(item[0]))


def _rocq_map_fold(
    function: Callable[[object, object, object], object],
    mapping: dict[object, object],
    initial: object,
) -> object:
    result = initial
    for key, value in _rocq_map_elements(mapping):
        result = function(key, value, result)
    return result


def _rocq_set_add(key: object, values: frozenset[object]) -> frozenset[object]:
    return frozenset((*values, key))


def _rocq_set_remove(key: object, values: frozenset[object]) -> frozenset[object]:
    return frozenset(value for value in values if value != key)


def _rocq_set_elements(values: frozenset[object]) -> list[object]:
    return sorted(values, key=_rocq_sorted_key)


def _rocq_set_fold(
    function: Callable[[object, object], object],
    values: frozenset[object],
    initial: object,
) -> object:
    result = initial
    for value in _rocq_set_elements(values):
        result = function(value, result)
    return result


def _rocq_ascii_to_int(
    b0: bool,
    b1: bool,
    b2: bool,
    b3: bool,
    b4: bool,
    b5: bool,
    b6: bool,
    b7: bool,
) -> int:
    return sum(
        (1 << i) for i, bit in enumerate((b0, b1, b2, b3, b4, b5, b6, b7)) if bit
    )


def _rocq_ascii_bits(
    value: str,
) -> tuple[bool, bool, bool, bool, bool, bool, bool, bool]:
    code = ord(value)
    if code < 0 or code > 255:
        raise ValueError("Rocq byte/ascii value out of range")
    return cast(
        tuple[bool, bool, bool, bool, bool, bool, bool, bool],
        tuple(bool(code & (1 << i)) for i in range(8)),
    )


class StateT(Generic[_StateTState, _StateTValue]):
    def __init__(
        self,
        step: Callable[[_StateTState], tuple[_StateTValue, _StateTState]],
        state: _StateTState | None = None,
    ) -> None:
        self._step = step
        self.state = state

    def run(self, state: _StateTState) -> _StateTValue:
        value, next_state = self._step(state)
        self.state = next_state
        return value

    def run_with_state(self, state: _StateTState) -> tuple[_StateTValue, _StateTState]:
        result = self._step(state)
        self.state = result[1]
        return result

    def bind(
        self,
        f: Callable[[_StateTValue], StateT[_StateTState, _StateTNext]],
    ) -> StateT[_StateTState, _StateTNext]:
        def run_bound(state: _StateTState) -> tuple[_StateTNext, _StateTState]:
            value, next_state = self._step(state)
            return f(value).run_with_state(next_state)

        return StateT(run_bound, self.state)

    @classmethod
    def pure(cls, value: _StateTValue) -> StateT[_StateTState, _StateTValue]:
        return StateT(lambda state: (value, state))

    @classmethod
    def get_state(cls) -> StateT[_StateTState, _StateTState]:
        return StateT(lambda state: (state, state))

    @classmethod
    def put_state(cls, new_state: _StateTState) -> StateT[_StateTState, None]:
        return StateT(lambda _state: (None, new_state))


class IO(Generic[_IOValue]):
    def __init__(self, thunk: Callable[[], Awaitable[_IOValue]]) -> None:
        self._thunk = thunk

    async def run(self) -> _IOValue:
        return await self._thunk()

    def bind(self, f: Callable[[_IOValue], IO[_IONext]]) -> IO[_IONext]:
        async def run_bound() -> _IONext:
            value = await self.run()
            return await f(value).run()

        return IO(run_bound)

    @classmethod
    def pure(cls, value: _IOValue) -> IO[_IOValue]:
        async def run_pure() -> _IOValue:
            return value

        return IO(run_pure)

    @classmethod
    def from_sync(cls, thunk: Callable[[], _IOValue]) -> IO[_IOValue]:
        async def run_sync() -> _IOValue:
            return thunk()

        return IO(run_sync)

    @classmethod
    def from_blocking(cls, thunk: Callable[[], _IOValue]) -> IO[_IOValue]:
        async def run_blocking() -> _IOValue:
            return await asyncio.to_thread(thunk)

        return IO(run_blocking)

    @classmethod
    @asynccontextmanager
    async def ownership(
        cls,
        acquire: IO[_IOOwner],
        release: Callable[[_IOOwner], IO[None]],
    ) -> AsyncIterator[_IOOwner]:
        owner = await acquire.run()
        try:
            yield owner
        finally:
            await release(owner).run()

    @classmethod
    def bracket(
        cls,
        acquire: IO[_IOOwner],
        release: Callable[[_IOOwner], IO[None]],
        use: Callable[[_IOOwner], IO[_IONext]],
    ) -> IO[_IONext]:
        async def run_bracket() -> _IONext:
            async with cls.ownership(acquire, release) as owner:
                return await use(owner).run()

        return IO(run_bracket)


class Mutex:
    """Explicit threading mutex wrapper for extracted coordination models.

    This wrapper exposes Python's lock behavior; it does not prove fairness,
    starvation freedom, or a Rocq-level scheduler semantics.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @classmethod
    def new(cls) -> IO[Mutex]:
        return IO.from_sync(cls)

    def acquire(self) -> IO[None]:
        def lock() -> None:
            self._lock.acquire()

        return IO.from_blocking(lock)

    def release(self) -> IO[None]:
        def unlock() -> None:
            self._lock.release()

        return IO.from_sync(unlock)


class Channel(Generic[_ChannelValue]):
    """Explicit FIFO channel wrapper backed by queue.SimpleQueue.

    FIFO order is for completed sends observed by this queue.  This wrapper
    does not model scheduler fairness, producer/consumer races, or cancellation.
    """

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[_ChannelValue] = queue.SimpleQueue()

    @classmethod
    def new(cls) -> IO[Channel[_ChannelValue]]:
        return IO.from_sync(cls)

    def send(self, value: _ChannelValue) -> IO[None]:
        def put() -> None:
            self._queue.put(value)

        return IO.from_sync(put)

    def receive(self) -> IO[_ChannelValue]:
        return IO.from_blocking(self._queue.get)


class Future(Generic[_FutureValue]):
    """Explicit future wrapper backed by concurrent.futures.Future.

    Completion follows Python Future semantics: result blocks until completed,
    double completion raises, and no Rocq-level thread interleaving is implied.
    """

    def __init__(self) -> None:
        self._future: _ConcurrentFuture[_FutureValue] = _ConcurrentFuture()

    @classmethod
    def new(cls) -> IO[Future[_FutureValue]]:
        return IO.from_sync(cls)

    def set_result(self, value: _FutureValue) -> IO[None]:
        def set_value() -> None:
            self._future.set_result(value)

        return IO.from_sync(set_value)

    def result(self) -> IO[_FutureValue]:
        return IO.from_blocking(self._future.result)

    def done(self) -> bool:
        return self._future.done()


def coforce(value: Callable[[], _CoForceT]) -> _CoForceT:
    return value()


def coprefix_eq(n: int, left: Iterable[object], right: Iterable[object]) -> bool:
    return tuple(islice(iter(left), n)) == tuple(islice(iter(right), n))


def coprefix_hash(n: int, value: Iterable[object]) -> int:
    return hash(tuple(islice(iter(value), n)))


class __ModuleNamespace:
    pass


def __apply_applicative(
    cache: dict[int, _ModuleRetT],
    build: Callable[[_ModuleArgT], _ModuleRetT],
    arg: _ModuleArgT,
) -> _ModuleRetT:
    key = id(arg)
    cached = cache.get(key)
    if cached is None:
        cached = build(arg)
        cache[key] = cached
    return cached


__ = None  # erased logical argument
|}

let preamble _state _name comment _used_modules _safe =
  (* Target: Python 3.14t (free-threaded build, no GIL).
     PEP 649 deferred annotation evaluation is the default on 3.14, so forward
     references in dataclass field types (e.g. [class EvenS: arg0: Odd] before
     [Odd] is defined) work without any import.  We do not support older Python
     versions. *)
  let comment =
    match comment with
    | None -> mt ()
    | Some c -> str "# " ++ c ++ fnl ()
  in
  str (Printf.sprintf "# Generated by rocq-python-extraction (target: %s)\n"
         "Python 3.14t") ++
  comment ++ str runtime_prelude

(*s Helpers for Python literal emission. *)

(** Escape a string for use in Python double-quoted string literals. *)
let py_escape_str s =
  let buf = Buffer.create (String.length s + 2) in
  String.iter (function
    | '"'  -> Buffer.add_string buf "\\\""
    | '\\' -> Buffer.add_string buf "\\\\"
    | '\n' -> Buffer.add_string buf "\\n"
    | '\r' -> Buffer.add_string buf "\\r"
    | '\t' -> Buffer.add_string buf "\\t"
    | c    -> Buffer.add_char buf c) s;
  Buffer.contents buf

(** Emit a Python double-quoted single-character string literal for an ASCII
    code point (0–255).  Printable ASCII characters are emitted directly;
    common escape sequences ([\\n], [\\r], [\\t]) use their symbolic form;
    everything else uses [\\xNN]. *)
let pp_ascii_char_lit code =
  let escaped =
    match Char.chr code with
    | '"'  -> "\\\""
    | '\\' -> "\\\\"
    | '\n' -> "\\n"
    | '\r' -> "\\r"
    | '\t' -> "\\t"
    | c when Char.code c >= 32 && Char.code c < 127 -> String.make 1 c
    | c    -> Printf.sprintf "\\x%02x" (Char.code c)
  in
  str ("\"" ^ escaped ^ "\"")

(** Escape bytes for Python [b"..."] literals.  Non-printable and non-ASCII
    bytes are emitted as [\\xHH] hex escapes. *)
let py_escape_bytes s =
  let buf = Buffer.create (String.length s + 2) in
  String.iter (fun c ->
    let code = Char.code c in
    if code >= 32 && code < 127 && c <> '"' && c <> '\\' then
      Buffer.add_char buf c
    else
      Buffer.add_string buf (Printf.sprintf "\\x%02x" code)) s;
  Buffer.contents buf

(** Emit a Python float literal.  [Float64.to_string] uses ["nan"],
    ["infinity"], and ["neg_infinity"]; map these to Python's forms. *)
let pp_float_lit f =
  match Float64.to_string f with
  | "nan"          -> str "float('nan')"
  | "infinity"     -> str "float('inf')"
  | "neg_infinity" -> str "float('-inf')"
  | s              -> str s

(*s Python-safe identifier printer.
    Rocq/OCaml allow primed identifiers ([n''], [f'']) that are illegal in
    Python.  Convert each apostrophe to an underscore so names like [n'] become
    [n_] and remain valid Python identifiers. *)

let pp_pyid id =
  str (String.map (function '\'' -> '_' | c -> c) (Id.to_string id))

let pp_pyname s =
  str (String.map (function '\'' -> '_' | c -> c) s)

let indent_string indent =
  String.make indent ' '

let strip_trailing_newline s =
  if String.length s > 0 && s.[String.length s - 1] = '\n' then
    String.sub s 0 (String.length s - 1)
  else
    s

let indent_pp indent pp =
  let prefix = indent_string indent in
  let body = strip_trailing_newline (Pp.string_of_ppcmds pp) in
  if String.equal body "" then
    mt ()
  else
    let lines = String.split_on_char '\n' body in
    str (String.concat "\n"
           (List.map
              (fun line -> if String.equal line "" then line else prefix ^ line)
              lines)) ++
    fnl ()

let is_dummy_id id =
  Id.equal id dummy_name

let visible_params ids =
  List.filter (fun id -> not (is_dummy_id id)) ids

let pp_param id =
  pp_pyid id

let pp_param_list ids =
  prlist_with_sep (fun () -> str ", ") pp_param (visible_params ids)

let pp_lambda ids body =
  let params = visible_params ids in
  if List.is_empty params then str "lambda: " ++ body
  else
    str "lambda " ++
    prlist_with_sep (fun () -> str ", ") pp_param params ++
    str ": " ++ body

let is_erased_arg = function
  | MLdummy _ -> true
  | _         -> false

let rec type_has_typevars = function
  | Tarr (t1, t2) ->
      type_has_typevars t1 || type_has_typevars t2
  | Tglob (_, args) ->
      List.exists type_has_typevars args
  | Tvar _ | Tvar' _ ->
      true
  | Tdummy _ | Tunknown | Taxiom | Tmeta _ ->
      false

type prop_context =
  | PropIrrelevant
  | PropValue
  | PropControl

let is_prop_type = function
  | Tdummy Kprop -> true
  | _ -> false

let pp_impossible_expr () =
  str "_impossible()"

let pp_impossible_stmt () =
  str "raise _Impossible()"

let set_once slot value =
  match !slot with
  | None -> slot := Some value
  | Some _ -> ()

let active_method_targets : (string * (string * string)) list ref = ref []

let lookup_active_method_target source_name =
  List.assoc_opt source_name !active_method_targets

let pp_collection_key kind pp_key =
  match kind with
  | `Positive -> str "_rocq_positive_key(" ++ pp_key ++ str ")"
  | `String -> str "_rocq_string_key(" ++ pp_key ++ str ")"

(*s Built-in Stdlib remappings.

    These are deliberately backend-owned rather than expressed with local
    [Extract Inductive] pragmas.  They give ordinary Rocq text/byte values a
    stable Python representation in every model:

      String.string -> str
      Ascii.ascii   -> int in 0..255
      Byte.byte     -> int in 0..255

    Rocq [String.string] is structurally a byte list.  The Python value is
    decoded UTF-8 text, so destructing a string is only valid at UTF-8
    character boundaries. *)

let has_suffix s suffix =
  let s_len = String.length s in
  let suffix_len = String.length suffix in
  s_len >= suffix_len &&
  String.equal suffix (String.sub s (s_len - suffix_len) suffix_len)

let string_contains s needle =
  let s_len = String.length s in
  let needle_len = String.length needle in
  if needle_len = 0 then true
  else if needle_len > s_len then false
  else
    let rec loop i =
      i + needle_len <= s_len &&
      (String.equal needle (String.sub s i needle_len) || loop (i + 1))
    in
    loop 0

let global_path r =
  try
    DirPath.to_string (Nametab.dirpath_of_global r.glob) ^ "." ^
    Id.to_string (Nametab.basename_of_global r.glob)
  with Not_found ->
    ""

let global_basename r =
  try Id.to_string (Nametab.basename_of_global r.glob)
  with Not_found -> ""

let global_path_has_suffix r suffix =
  has_suffix (global_path r) suffix

let is_std_string_type_ref r =
  global_path_has_suffix r ".Strings.String.string"

let is_std_string_empty_ref r =
  global_path_has_suffix r ".Strings.String.EmptyString"

let is_std_string_cons_ref r =
  global_path_has_suffix r ".Strings.String.String"

let is_std_ascii_type_ref r =
  global_path_has_suffix r ".Strings.Ascii.ascii"

let is_std_ascii_cons_ref r =
  global_path_has_suffix r ".Strings.Ascii.Ascii"

let is_std_byte_type_ref r =
  global_path_has_suffix r ".Init.Byte.byte"

let is_prim_string_type_ref r =
  global_path_has_suffix r ".Strings.PrimString.string"

let is_std_nat_type_ref r =
  global_path_has_suffix r ".Init.Datatypes.nat"

let is_std_nat_zero_ref r =
  global_path_has_suffix r ".Init.Datatypes.O"

let is_std_nat_succ_ref r =
  global_path_has_suffix r ".Init.Datatypes.S"

let is_std_positive_type_ref r =
  global_path_has_suffix r ".Numbers.BinNums.positive"

let is_std_positive_xh_ref r =
  global_path_has_suffix r ".Numbers.BinNums.xH"

let is_std_positive_xo_ref r =
  global_path_has_suffix r ".Numbers.BinNums.xO"

let is_std_positive_xi_ref r =
  global_path_has_suffix r ".Numbers.BinNums.xI"

let is_std_N_type_ref r =
  global_path_has_suffix r ".Numbers.BinNums.N"

let is_std_N_zero_ref r =
  global_path_has_suffix r ".Numbers.BinNums.N0"

let is_std_N_pos_ref r =
  global_path_has_suffix r ".Numbers.BinNums.Npos"

let is_std_Z_type_ref r =
  global_path_has_suffix r ".Numbers.BinNums.Z"

let is_std_Z_zero_ref r =
  global_path_has_suffix r ".Numbers.BinNums.Z0"

let is_std_Z_pos_ref r =
  global_path_has_suffix r ".Numbers.BinNums.Zpos"

let is_std_Z_neg_ref r =
  global_path_has_suffix r ".Numbers.BinNums.Zneg"

let is_std_Q_type_ref r =
  global_path_has_suffix r ".QArith.QArith_base.Q"

let is_std_Q_make_ref r =
  global_path_has_suffix r ".QArith.QArith_base.Qmake"

let is_std_real_type_ref r =
  global_path_has_suffix r ".Reals.Rdefinitions.R" ||
  global_path_has_suffix r ".Reals.Rdefinitions.RbaseSymbolsImpl.R"

let is_std_option_type_ref r =
  global_path_has_suffix r ".Init.Datatypes.option"

let is_std_option_none_ref r =
  global_path_has_suffix r ".Init.Datatypes.None"

let is_std_option_some_ref r =
  global_path_has_suffix r ".Init.Datatypes.Some"

let is_std_list_type_ref r =
  global_path_has_suffix r ".Init.Datatypes.list"

let is_std_list_nil_ref r =
  global_path_has_suffix r ".Init.Datatypes.nil"

let is_std_list_cons_ref r =
  global_path_has_suffix r ".Init.Datatypes.cons"

let is_std_list_app_ref r =
  global_path_has_suffix r ".Lists.List.app" ||
  global_path_has_suffix r ".Lists.ListDef.app" ||
  global_path_has_suffix r ".Init.Datatypes.app"

let is_std_prod_type_ref r =
  global_path_has_suffix r ".Init.Datatypes.prod"

let is_std_prod_pair_ref r =
  global_path_has_suffix r ".Init.Datatypes.pair"

let is_std_bool_type_ref r =
  global_path_has_suffix r ".Init.Datatypes.bool"

let is_std_nat_ref r name =
  global_path_has_suffix r (".Init.Nat." ^ name)

let is_std_ascii_ref r name =
  global_path_has_suffix r (".Strings.Ascii." ^ name)

let is_std_string_ref r name =
  global_path_has_suffix r (".Strings.String." ^ name)

let is_std_positive_ref r name =
  global_path_has_suffix r (".PArith.BinPos.Pos." ^ name) ||
  global_path_has_suffix r (".PArith.BinPosDef.Pos." ^ name)

let is_std_primitive_compare_ref r =
  is_std_nat_ref r "eqb" || is_std_nat_ref r "leb" || is_std_nat_ref r "ltb" ||
  is_std_positive_ref r "eqb" || is_std_positive_ref r "leb" || is_std_positive_ref r "ltb" ||
  is_std_ascii_ref r "eqb" || is_std_ascii_ref r "leb" || is_std_ascii_ref r "ltb" ||
  is_std_string_ref r "eqb" || is_std_string_ref r "leb" || is_std_string_ref r "ltb"

let is_positive_map_type_ref r =
  global_path_has_suffix r ".FSets.FMapPositive.PositiveMap.t"

let is_positive_set_type_ref r =
  global_path_has_suffix r ".MSets.MSetPositive.PositiveSet.t"

let is_string_map_type_ref r =
  let p = global_path r in
  has_suffix p ".t" && string_contains p ".StringMap."

let is_string_set_type_ref r =
  let p = global_path r in
  has_suffix p ".t" && string_contains p ".StringSet."

let is_positive_map_ref r name =
  global_path_has_suffix r (".FSets.FMapPositive.PositiveMap." ^ name)

let is_positive_set_ref r name =
  global_path_has_suffix r (".MSets.MSetPositive.PositiveSet." ^ name)

let is_string_map_ref r name =
  let p = global_path r in
  has_suffix p ("." ^ name) && string_contains p ".StringMap."

let is_string_set_ref r name =
  let p = global_path r in
  has_suffix p ("." ^ name) && string_contains p ".StringSet."

let is_std_collection_module_name name =
  name = "PositiveMap" || name = "PositiveSet" ||
  name = "StringMap" || name = "StringSet" ||
  has_suffix name ".PositiveMap" || has_suffix name ".PositiveSet" ||
  has_suffix name ".StringMap" || has_suffix name ".StringSet"

let is_std_collection_term_ref r =
  let names = ["empty"; "add"; "remove"; "find"; "mem"; "cardinal"; "elements"; "fold";
               "union"; "inter"; "diff"] in
  List.exists
    (fun name ->
       is_positive_map_ref r name || is_string_map_ref r name ||
       is_positive_set_ref r name || is_string_set_ref r name)
    names || is_std_list_app_ref r

let std_byte_constructor_value r =
  let name = global_basename r in
  if String.length name = 3 && name.[0] = 'x' then
    int_of_string_opt ("0x" ^ String.sub name 1 2)
  else
    None

let is_std_byte_cons_ref r =
  match std_byte_constructor_value r with
  | Some n -> n >= 0 && n <= 255
  | None -> false

let is_std_remapped_type_ref r =
  is_std_string_type_ref r || is_std_ascii_type_ref r || is_std_byte_type_ref r ||
  is_prim_string_type_ref r || is_std_nat_type_ref r ||
  is_std_positive_type_ref r || is_std_N_type_ref r || is_std_Z_type_ref r ||
  is_std_Q_type_ref r || is_std_option_type_ref r || is_std_list_type_ref r ||
  is_std_prod_type_ref r || is_positive_map_type_ref r ||
  is_positive_set_type_ref r || is_string_map_type_ref r ||
  is_string_set_type_ref r || is_std_bool_type_ref r

let is_std_string_type = function
  | Tglob (r, _) -> is_std_string_type_ref r
  | _ -> false

let is_std_ascii_type = function
  | Tglob (r, _) -> is_std_ascii_type_ref r
  | _ -> false

let is_std_byte_type = function
  | Tglob (r, _) -> is_std_byte_type_ref r
  | _ -> false

let is_std_nat_type = function
  | Tglob (r, _) -> is_std_nat_type_ref r
  | _ -> false

let is_std_positive_type = function
  | Tglob (r, _) -> is_std_positive_type_ref r
  | _ -> false

let is_std_N_type = function
  | Tglob (r, _) -> is_std_N_type_ref r
  | _ -> false

let is_std_Z_type = function
  | Tglob (r, _) -> is_std_Z_type_ref r
  | _ -> false

let is_std_Q_type = function
  | Tglob (r, _) -> is_std_Q_type_ref r
  | _ -> false

let is_std_option_type = function
  | Tglob (r, _) -> is_std_option_type_ref r
  | _ -> false

let is_std_list_type = function
  | Tglob (r, _) -> is_std_list_type_ref r
  | _ -> false

let is_std_prod_type = function
  | Tglob (r, _) -> is_std_prod_type_ref r
  | _ -> false

let is_std_bool_true_ref r =
  global_path_has_suffix r ".Init.Datatypes.true"

let is_std_bool_false_ref r =
  global_path_has_suffix r ".Init.Datatypes.false"

let is_std_bool_type = function
  | Tglob (r, _) -> is_std_bool_type_ref r
  | _ -> false

let std_bool_expr = function
  | MLcons (_, r, []) when is_std_bool_true_ref r -> Some true
  | MLcons (_, r, []) when is_std_bool_false_ref r -> Some false
  | _ -> None

let std_ascii_expr_value = function
  | MLcons (_, r, args) when is_std_ascii_cons_ref r && List.length args = 8 ->
      let bits = List.map std_bool_expr args in
      if List.exists (function None -> true | Some _ -> false) bits then None
      else
        Some
          (List.mapi
             (fun i bit -> if Option.get bit then 1 lsl i else 0)
             bits
           |> List.fold_left ( + ) 0)
  | _ -> None

let std_string_expr_value expr =
  let buf = Buffer.create 16 in
  let rec loop = function
    | MLcons (_, r, []) when is_std_string_empty_ref r ->
        Some (Buffer.contents buf)
    | MLcons (_, r, [head; tail]) when is_std_string_cons_ref r ->
        (match std_ascii_expr_value head with
         | Some value when value >= 0 && value <= 255 ->
             Buffer.add_char buf (Char.chr value);
             loop tail
         | _ -> None)
    | _ -> None
  in
  loop expr

let needs_numeric_constructor_parens = function
  | MLcons (_, r, [_]) when is_std_nat_succ_ref r ->
      true
  | MLcons (_, r, [_])
    when is_std_positive_xo_ref r || is_std_positive_xi_ref r ->
      true
  | _ ->
      false

type diagnostic = {
  code : string;
  title : string;
  category : string;
  remediation : string;
  docs : string;
}

let diagnostic_catalogue = [
  { code = "PYEX001"; title = "Persistent arrays are unsupported"; category = "miniml-expression"; remediation = "Avoid Rocq PArray in extracted terms or add an explicit Python remapping."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex001-persistent-arrays-are-unsupported" };
  { code = "PYEX002"; title = "Prop term used for computation"; category = "prop-erasure"; remediation = "Move the proof to Prop-only positions or return data in Set/Type before extraction."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex002-prop-term-used-for-computation" };
  { code = "PYEX003"; title = "Type alias declaration is not emitted"; category = "miniml-declaration"; remediation = "Use a concrete inductive/record or add an extraction remapping for the type."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex003-type-alias-declaration-is-not-emitted" };
  { code = "PYEX004"; title = "Axiom has no computational realization"; category = "runtime-stub"; remediation = "Provide an Extract Constant remapping for the axiom before Python extraction."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex004-axiom-has-no-computational-realization" };
  { code = "PYEX005"; title = "Exception term cannot be emitted as an expression"; category = "miniml-expression"; remediation = "Keep extracted exceptions at statement position or rewrite the Rocq term to return data."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex005-exception-term-cannot-be-emitted-as-an-expression" };
  { code = "PYEX006"; title = "Erased logical value reached runtime"; category = "prop-erasure"; remediation = "Keep the logical argument proof-irrelevant or pass a computational witness in Set/Type."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex006-erased-logical-value-reached-runtime" };
  { code = "PYEX007"; title = "Unsupported custom match encoding"; category = "custom-remap"; remediation = "Give Extract Inductive a Python match function with one thunk per constructor plus the scrutinee."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex007-unsupported-custom-match-encoding" };
  { code = "PYEX008"; title = "Custom constructor arity mismatch"; category = "custom-remap"; remediation = "Make every custom constructor expression accept the same arguments as the Rocq constructor."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex008-custom-constructor-arity-mismatch" };
  { code = "PYEX009"; title = "Unknown monad marker"; category = "custom-remap"; remediation = "Use one of the supported __PYMONAD_* markers or extract the operation normally."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex009-unknown-monad-marker" };
  { code = "PYEX010"; title = "Monad marker arity mismatch"; category = "custom-remap"; remediation = "Use the marker with the expected number of computational arguments."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex010-monad-marker-arity-mismatch" };
  { code = "PYEX011"; title = "Record projection pattern is too complex"; category = "pattern"; remediation = "Split the nested match into separate matches or bind the record before matching."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex011-record-projection-pattern-is-too-complex" };
  { code = "PYEX012"; title = "Nested wildcard binder escaped erasure"; category = "pattern"; remediation = "Name the computational field explicitly or keep the wildcard proof-only."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex012-nested-wildcard-binder-escaped-erasure" };
  { code = "PYEX013"; title = "Coinductive packet is not stream-shaped"; category = "coinductive"; remediation = "Expose a one-step destructor or avoid relying on Python iterator synthesis."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex013-coinductive-packet-is-not-stream-shaped" };
  { code = "PYEX014"; title = "Coinductive constructor arity mismatch"; category = "coinductive"; remediation = "Use native coinductive constructors without custom erasure for this extraction."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex014-coinductive-constructor-arity-mismatch" };
  { code = "PYEX015"; title = "Mutual cofixpoint shape is unsupported"; category = "coinductive"; remediation = "Extract a wrapper function around one cofixpoint or split the mutual block."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex015-mutual-cofixpoint-shape-is-unsupported" };
  { code = "PYEX016"; title = "Higher-order module signature is unsupported"; category = "module"; remediation = "Extract the applied module result or simplify the functor signature."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex016-higher-order-module-signature-is-unsupported" };
  { code = "PYEX017"; title = "Module alias could not be resolved"; category = "module"; remediation = "Extract the canonical module path or make the alias transparent before extraction."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex017-module-alias-could-not-be-resolved" };
  { code = "PYEX018"; title = "Module type with constraints is unsupported"; category = "module"; remediation = "Extract the constrained module after elaboration or remove the with-constraint."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex018-module-type-with-constraints-is-unsupported" };
  { code = "PYEX019"; title = "Applicative functor cache key is unsupported"; category = "module"; remediation = "Pass a first-class module value or extract a non-functorized wrapper."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex019-applicative-functor-cache-key-is-unsupported" };
  { code = "PYEX020"; title = "Expected an inductive or constructor reference"; category = "backend-invariant"; remediation = "Report this backend invariant with the extracted declaration and source map."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex020-expected-an-inductive-or-constructor-reference" };
  { code = "PYEX021"; title = "Expected an inductive reference"; category = "backend-invariant"; remediation = "Report this backend invariant with the extracted declaration and source map."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex021-expected-an-inductive-reference" };
  { code = "PYEX022"; title = "Pattern shorthand was not expanded"; category = "backend-invariant"; remediation = "Report this backend invariant; the printer should expand Pusual before rendering."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex022-pattern-shorthand-was-not-expanded" };
  { code = "PYEX023"; title = "Unexpected coinductive constructor payload"; category = "backend-invariant"; remediation = "Report this backend invariant with the constructor and generated source map."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex023-unexpected-coinductive-constructor-payload" };
  { code = "PYEX024"; title = "Unsupported primitive integer type alias"; category = "primitive"; remediation = "Keep the integer literal in a computational term or remap the type explicitly."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex024-unsupported-primitive-integer-type-alias" };
  { code = "PYEX025"; title = "Unsupported primitive float type alias"; category = "primitive"; remediation = "Keep the float literal in a computational term or remap the type explicitly."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex025-unsupported-primitive-float-type-alias" };
  { code = "PYEX026"; title = "Unsupported primitive string type alias"; category = "primitive"; remediation = "Keep the string literal in a computational term or remap the type explicitly."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex026-unsupported-primitive-string-type-alias" };
  { code = "PYEX027"; title = "Unknown type annotation shape"; category = "type-annotation"; remediation = "Use an explicit extracted type remapping or simplify the polymorphic type."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex027-unknown-type-annotation-shape" };
  { code = "PYEX028"; title = "Function protocol annotation is too complex"; category = "type-annotation"; remediation = "Name the higher-order argument type or specialize the extracted function."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex028-function-protocol-annotation-is-too-complex" };
  { code = "PYEX029"; title = "Generic constructor could not be typed"; category = "type-annotation"; remediation = "Specialize the inductive parameters or add a primitive remapping."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex029-generic-constructor-could-not-be-typed" };
  { code = "PYEX030"; title = "Logical inductive used computationally"; category = "prop-erasure"; remediation = "Move the inductive to Set/Type or erase the use before extraction."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex030-logical-inductive-used-computationally" };
  { code = "PYEX031"; title = "Logical record used computationally"; category = "prop-erasure"; remediation = "Separate computational fields into a Set/Type record before extraction."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex031-logical-record-used-computationally" };
  { code = "PYEX032"; title = "Proof-carrying pair leaked into Python"; category = "prop-erasure"; remediation = "Project the computational component before extraction."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex032-proof-carrying-pair-leaked-into-python" };
  { code = "PYEX033"; title = "Unsupported well-founded recursion shape"; category = "recursion"; remediation = "Expose the structurally recursive helper or simplify the Program Fixpoint obligation shape."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex033-unsupported-well-founded-recursion-shape" };
  { code = "PYEX034"; title = "Local fixpoint escaped statement context"; category = "recursion"; remediation = "Eta-expand the definition so the local fixpoint appears inside a function body."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex034-local-fixpoint-escaped-statement-context" };
  { code = "PYEX035"; title = "Mutual recursion has erased selected function"; category = "recursion"; remediation = "Extract a computational member of the mutual block or remove the proof-only member."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex035-mutual-recursion-has-erased-selected-function" };
  { code = "PYEX036"; title = "Unsupported bytes literal encoding"; category = "primitive"; remediation = "Restrict extracted strings to byte strings or provide a Python remapping."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex036-unsupported-bytes-literal-encoding" };
  { code = "PYEX037"; title = "Unsupported float literal"; category = "primitive"; remediation = "Avoid NaN/infinity payloads that cannot round-trip or remap the constant explicitly."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex037-unsupported-float-literal" };
  { code = "PYEX038"; title = "Generated Python identifier is invalid"; category = "naming"; remediation = "Rename the Rocq identifier or add an extraction rename before Python extraction."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex038-generated-python-identifier-is-invalid" };
  { code = "PYEX039"; title = "Generated Python name collision"; category = "naming"; remediation = "Rename one Rocq declaration or extract through a module namespace."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex039-generated-python-name-collision" };
  { code = "PYEX040"; title = "Unclassified extraction failure"; category = "internal"; remediation = "Check the detail field, reduce the Rocq input, and add a catalogue entry for this failure."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex040-unclassified-extraction-failure" };
  { code = "PYEX041"; title = "Unsupported real number extraction"; category = "numeric"; remediation = "Use nat, positive, N, Z, or Q for extracted computation; Rocq R has no faithful Python runtime mapping."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex041-unsupported-real-number-extraction" };
  { code = "PYEX042"; title = "Unsupported IO effect extraction"; category = "io"; remediation = "Keep IO values at an async boundary or provide an explicit IO adapter remapping."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex042-unsupported-io-effect-extraction" };
  { code = "PYEX043"; title = "Unsupported concurrency scheduling extraction"; category = "concurrency"; remediation = "Model deterministic wrapper boundaries only; do not extract scheduler interleavings as executable Python."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex043-unsupported-concurrency-scheduling-extraction" };
  { code = "PYEX044"; title = "Concurrency marker arity mismatch"; category = "concurrency"; remediation = "Use supported __PYCONC_* markers with the documented number of computational arguments."; docs = "rocq-python-extraction/DIAGNOSTICS.md#pyex044-concurrency-marker-arity-mismatch" };
]

let diagnostic_prefix = "PYTHON_EXTRACTION_DIAGNOSTIC_JSON: "

let json_escape s =
  let buf = Buffer.create (String.length s + 8) in
  String.iter
    (function
      | '"' -> Buffer.add_string buf "\\\""
      | '\\' -> Buffer.add_string buf "\\\\"
      | '\b' -> Buffer.add_string buf "\\b"
      | '\012' -> Buffer.add_string buf "\\f"
      | '\n' -> Buffer.add_string buf "\\n"
      | '\r' -> Buffer.add_string buf "\\r"
      | '\t' -> Buffer.add_string buf "\\t"
      | c ->
          let code = Char.code c in
          if code < 0x20 then Buffer.add_string buf (Printf.sprintf "\\u%04x" code)
          else Buffer.add_char buf c)
    s;
  Buffer.contents buf

let json_string s = "\"" ^ json_escape s ^ "\""

let diagnostic_by_code code =
  match List.find_opt (fun d -> String.equal d.code code) diagnostic_catalogue with
  | Some d -> d
  | None -> List.find (fun d -> String.equal d.code "PYEX040") diagnostic_catalogue

let diagnostic_json ?symbol ?detail d =
  let field name value = "\"" ^ name ^ "\": " ^ json_string value in
  let optional =
    (match symbol with None -> [] | Some s -> [field "symbol" s]) @
    (match detail with None -> [] | Some s -> [field "detail" s])
  in
  "{" ^
  String.concat ", "
    ([
      "\"version\": 1";
      field "code" d.code;
      field "title" d.title;
      field "category" d.category;
      field "message" d.title;
      field "remediation" d.remediation;
      field "docs" d.docs;
    ] @ optional) ^
  "}"

let diagnostic_pp ?symbol ?detail code =
  let d = diagnostic_by_code code in
  str "Python ExtractionError [" ++ str d.code ++ str "]: " ++ str d.title ++ fnl () ++
  str "Remediation: " ++ str d.remediation ++ fnl () ++
  str "Docs: " ++ str d.docs ++ fnl () ++
  (match detail with
   | None -> mt ()
   | Some detail -> str "Detail: " ++ str detail ++ fnl ()) ++
  str diagnostic_prefix ++ str (diagnostic_json ?symbol ?detail d)

let extraction_diagnostic_error ?symbol ?detail code =
  user_err (diagnostic_pp ?symbol ?detail code)

let diagnostic_comment ?detail code =
  let d = diagnostic_by_code code in
  let detail =
    match detail with
    | None -> ""
    | Some detail -> " Detail: " ^ detail
  in
  str "# Python ExtractionDiagnostic [" ++ str d.code ++ str "]: " ++
  str d.title ++ str " Remediation: " ++ str d.remediation ++
  str detail ++ fnl ()

let prop_extraction_error detail =
  extraction_diagnostic_error ~detail "PYEX002"

let marker_state_type = "__PYMONAD_STATE_TYPE__"
let marker_state_pure = "__PYMONAD_STATE_PURE__"
let marker_state_bind = "__PYMONAD_STATE_BIND__"
let marker_state_get = "__PYMONAD_STATE_GET__"
let marker_state_put = "__PYMONAD_STATE_PUT__"
let marker_option_bind = "__PYMONAD_OPTION_BIND__"
let marker_reader_pure = "__PYMONAD_READER_PURE__"
let marker_reader_bind = "__PYMONAD_READER_BIND__"
let marker_reader_ask = "__PYMONAD_READER_ASK__"
let marker_io_type = "__PYMONAD_IO_TYPE__"
let marker_io_pure = "__PYMONAD_IO_PURE__"
let marker_io_bind = "__PYMONAD_IO_BIND__"
let marker_io_bracket = "__PYMONAD_IO_BRACKET__"
let marker_io_run = "__PYMONAD_IO_RUN__"
let marker_mutex_type = "__PYCONC_MUTEX_TYPE__"
let marker_channel_type = "__PYCONC_CHANNEL_TYPE__"
let marker_future_type = "__PYCONC_FUTURE_TYPE__"
let marker_new_mutex = "__PYCONC_NEW_MUTEX__"
let marker_new_channel = "__PYCONC_NEW_CHANNEL__"
let marker_new_future = "__PYCONC_NEW_FUTURE__"
let marker_mutex_acquire = "__PYCONC_MUTEX_ACQUIRE__"
let marker_mutex_release = "__PYCONC_MUTEX_RELEASE__"
let marker_channel_send = "__PYCONC_CHANNEL_SEND__"
let marker_channel_receive = "__PYCONC_CHANNEL_RECEIVE__"
let marker_future_set = "__PYCONC_FUTURE_SET__"
let marker_future_result = "__PYCONC_FUTURE_RESULT__"
let marker_future_done = "__PYCONC_FUTURE_DONE__"
let marker_interleave = "__PYCONC_INTERLEAVE__"

let is_monad_marker_string s =
  let prefix = "__PYMONAD_" in
  let prefix_len = String.length prefix in
  String.length s >= prefix_len &&
  String.equal prefix (String.sub s 0 prefix_len)

let is_concurrency_marker_string s =
  let prefix = "__PYCONC_" in
  let prefix_len = String.length prefix in
  String.length s >= prefix_len &&
  String.equal prefix (String.sub s 0 prefix_len)

let is_monad_marker_ref r =
  is_custom r && is_monad_marker_string (find_custom r)

let is_runtime_marker_ref r =
  is_custom r &&
  let marker = find_custom r in
  is_monad_marker_string marker || is_concurrency_marker_string marker

let marker_of_ast = function
  | MLglob r when is_custom r ->
      Some (find_custom r)
  | _ ->
      None

let rec validate_prop_discipline_expr context = function
  | MLdummy Kprop ->
      (match context with
       | PropControl ->
           prop_extraction_error
             "Prop-typed term used in computational control position"
       | PropIrrelevant | PropValue -> ())
  | MLrel _ | MLglob _ | MLdummy _ | MLexn _ | MLaxiom _
  | MLuint _ | MLfloat _ | MLstring _ | MLparray _ ->
      ()
  | MLmagic a ->
      validate_prop_discipline_expr context a
  | MLapp (f, args) ->
      validate_prop_discipline_expr PropControl f;
      List.iter (validate_prop_discipline_expr PropValue) args
  | MLlam (_, body) ->
      validate_prop_discipline_expr PropValue body
  | MLletin (_, a1, a2) ->
      validate_prop_discipline_expr PropValue a1;
      validate_prop_discipline_expr PropValue a2
  | MLcons (_, _, args) ->
      List.iter (validate_prop_discipline_expr PropValue) args
  | MLtuple args ->
      List.iter (validate_prop_discipline_expr PropValue) args
  | MLcase (_, scrutinee, branches) ->
      validate_prop_discipline_expr PropControl scrutinee;
      Array.iter
        (fun (_, _, body) -> validate_prop_discipline_expr PropValue body)
        branches
  | MLfix (_, _, defs) ->
      Array.iter (validate_prop_discipline_expr PropValue) defs

let validate_prop_discipline_decl = function
  | Dterm (_, body, typ) ->
      if not (is_prop_type typ) then
        validate_prop_discipline_expr PropValue body
  | Dfix (_, defs, typs) ->
      Array.iteri
        (fun i body ->
           if not (is_prop_type typs.(i)) then
             validate_prop_discipline_expr PropValue body)
        defs
  | Dind _ | Dtype _ ->
      ()

let pp_unreachable_fallback ty indent =
  let pfx = String.make indent ' ' in
  (* Static exhaustiveness and runtime exhaustiveness are different here.
     For monomorphic matches, pyright can usually narrow the fallback binder
     to [Never], so [assert_never] is the right terminator.  For generic
     matches (e.g. [MyOpt[_T]], [MyList[_T]]), the fallback is still runtime-
     unreachable but pyright often cannot prove that, so use
     [assert False, "unreachable"] instead. *)
  if type_has_typevars ty then
    str pfx ++ str "case _:" ++ fnl () ++
    str pfx ++ str "    assert False, \"unreachable\""
  else
    str pfx ++ str "case __impossible:" ++ fnl () ++
    str pfx ++ str "    assert_never(__impossible)"

(** Capitalize the first character of [s] for Python class names.
    Rocq type names are lowercased by the extraction framework (OCaml
    convention); Python expects PascalCase for class names (PEP 8). *)
let capitalize_first s =
  if String.length s = 0 then s
  else
    let b = Bytes.of_string s in
    Bytes.set b 0 (Char.uppercase_ascii (Bytes.get b 0));
    Bytes.to_string b

let type_is_coinductive state = function
  | Tglob (r, _) ->
      is_coinductive (State.get_table state) r
  | _ ->
      false

let pp_unreachable_fallback_for state ty indent =
  if type_is_coinductive state ty then
    let pfx = String.make indent ' ' in
    str pfx ++ str "case _:" ++ fnl () ++
    str pfx ++ str "    assert False, \"unreachable\""
  else
    pp_unreachable_fallback ty indent

(*s Global reference printer.  Honours [Extract Constant ... => "..."]
    inline-custom declarations. *)

let pp_glob state r =
  if is_inline_custom r then str (find_custom r)
  else str (pp_global state Term r)

(*s Helpers for [MLcons] emission. *)

(** Return the parent inductive reference for a constructor or inductive ref. *)
let get_ind r =
  let open GlobRef in
  match r.glob with
  | IndRef _              -> r
  | ConstructRef (ind, _) -> { glob = IndRef ind; inst = r.inst }
  | _                     ->
      extraction_diagnostic_error "PYEX020"

let coinductive_name state r =
  capitalize_first (pp_global state Term (get_ind r))

let packet_name state packet =
  capitalize_first (pp_global state Term packet.ip_typename_ref)

let packet_is_self_recursive state packet = function
  | Tglob (r, _) ->
      String.equal (packet_name state packet)
        (capitalize_first (pp_global state Term r))
  | _ ->
      false

let packet_stream_payload_type state packet =
  if Array.length packet.ip_types <> 1 then None
  else
    match packet.ip_types.(0) with
    | [payload_ty; tail_ty] when packet_is_self_recursive state packet tail_ty ->
        Some payload_ty
    | _ ->
        None

(** KerName of an inductive reference (for record field key lookup). *)
let kn_of_ind r =
  let open GlobRef in
  match r.glob with
  | IndRef (kn, _) -> MutInd.user kn
  | _              ->
      extraction_diagnostic_error "PYEX021"

(** Extract constructor parameter names from the Rocq kernel.
    Returns [string option list] of length [nargs]:
    [Some name] for a named binder, [None] for an anonymous one.
    Skips the [mind_nparams] leading products (type parameters) before
    collecting the [nargs] argument binders from [mind_user_lc.(j)]. *)
let cons_arg_names_from_kernel packet j nargs =
  let open GlobRef in
  match packet.ip_typename_ref.glob with
  | IndRef (mut_kn, i_ind) ->
      let env = Global.env () in
      let mib = Environ.lookup_mind mut_kn env in
      let oib = mib.mind_packets.(i_ind) in
      let constr_ty = oib.mind_user_lc.(j) in
      let rec skip n ty =
        if n <= 0 then ty
        else
          match Constr.kind ty with
          | Constr.Prod (_, _, body) -> skip (n - 1) body
          | _                        -> ty
      in
      let arg_ty = skip mib.mind_nparams constr_ty in
      let rec collect n ty acc =
        if n <= 0 then List.rev acc
        else
          match Constr.kind ty with
          | Constr.Prod (bnd, _, body) ->
              let name_opt =
                match Context.binder_name bnd with
                | Name.Name id ->
                    let s = Id.to_string id in
                    let s = if Id.Set.mem id keywords then s ^ "_" else s in
                    Some s
                | Name.Anonymous -> None
              in
              collect (n - 1) body (name_opt :: acc)
          | _ -> List.rev acc
      in
      collect nargs arg_ty []
  | _ -> List.init nargs (fun _ -> None)

(** Python keyword-argument name for record field at position [i].
    Uses [pp_global_with_key] for named fields, ["arg<i>"] for anonymous ones. *)
let pp_field_name state r fds i =
  match List.nth fds i with
  | Some r' -> str (pp_global_with_key state Term (kn_of_ind (get_ind r)) r')
  | None    -> str (Printf.sprintf "arg%d" i)

(** Raw constructor name for [r], accounting for inline-custom declarations. *)
let str_cons state r =
  if is_inline_custom r then find_custom r
  else pp_global state Cons r

(*s Pattern printer for Python [case] syntax. *)

(** True when the last branch in [branches] is a bare wildcard pattern.
    When it is, we skip the synthetic catch-all arm — the explicit wildcard
    already handles every remaining case and a second [case _:] would be
    unreachable. *)
let has_wildcard_last branches =
  let n = Array.length branches in
  n > 0 &&
  (let (_, pat, _) = branches.(n - 1) in
   match pat with
   | Pwild -> true
   | _     -> false)

(** Check whether pattern [p] has a custom extraction equal to string [s].
    Used to detect bool patterns mapped to ["True"] / ["False"]. *)
let is_bool_patt p s =
  try
    let r = match p with
      | Pusual r | Pcons (r, []) -> r
      | _ -> raise Not_found
    in
    String.equal (find_custom r) s
  with Not_found -> false

(** Expand a top-level [Pusual r] into [Pcons(r, [Prel n; …; Prel 1])]
    before the pattern reaches [pp_pattern].  [n] is the number of branch
    binders (i.e. [List.length ids] in the enclosing [pp_branch]).

    [Pusual r] is a shorthand emitted by Rocq's extraction only at the
    outermost level of a branch pattern — never nested inside another
    [Pcons] or [Ptuple].  Expanding it here into ordinary [Pcons]/[Prel]
    nodes means [pp_pattern] can work purely from the de Bruijn environment
    without a separate [ids] list, which in turn means nested sub-patterns
    (e.g. [Pcons(MCons, [Prel 2, Pcons(MCons, [Prel 1, Pwild])])]) render
    correctly: each [Prel n] resolves to the right binder name via [env']. *)
let expand_pusual n = function
  | Pusual r -> Pcons (r, List.init n (fun i -> Prel (n - i)))
  | pat      -> pat

(** Pretty-print an [ml_pattern] in Python [case] syntax.
    [env'] is the de Bruijn environment with all branch binders pushed.
    Call [expand_pusual] on the top-level pattern before entering here so
    that [Pusual] nodes never reach this function. *)
let rec pp_pattern state env' = function
  | Pcons (r, []) when is_std_byte_cons_ref r ->
      str (string_of_int (Option.get (std_byte_constructor_value r)))
  | Pusual r when is_std_byte_cons_ref r ->
      str (string_of_int (Option.get (std_byte_constructor_value r)))
  | Pcons (r, []) ->
      let cons = str_cons state r in
      (* Erased / empty-name constructor: emit wildcard so the arm still matches *)
      if String.equal "" cons then str "_"
      else str cons ++ str "()"
  | Pcons (r, pats) ->
      str (str_cons state r) ++ str "(" ++
      prlist_with_sep (fun () -> str ", ") (pp_pattern state env') pats ++
      str ")"
  | Pusual _ ->
      (* Should have been expanded by [expand_pusual] before this call. *)
      extraction_diagnostic_error "PYEX022"
  | Ptuple pats ->
      str "(" ++
      prlist_with_sep (fun () -> str ", ") (pp_pattern state env') pats ++
      str ")"
  | Prel n ->
      pp_pyid (get_db_name n env')
  | Pwild ->
      str "_"

(*s Record-projection detection helper.

    Returns [Some (r, fds, sub_pats)] when [branches] describes a single-branch
    match that the optimiser can convert into lambda-lifted attribute accesses:

      (lambda param_0, param_1, …: body)(scrutinee.field_0, scrutinee.field_1, …)

    Three conditions must hold:
    1. Exactly one branch.
    2. The branch pattern is [Pusual r] or [Pcons(r,_)] for a constructor [r]
       whose parent inductive is a record ([get_record_fields] is non-empty).
    3. Every expanded sub-pattern is [Prel k] (bound) or [Pwild] (discarded),
       and the sub-pattern count matches the field count.

    Condition 3 rules out nested constructor patterns inside a field position,
    which require genuine case analysis rather than a plain attribute access.

    Used by both [pp_expr] (expression context) and [pp_return_body] (inside
    a [def] body), so the optimisation fires in both settings. *)
let record_proj_info state branches =
  if Array.length branches <> 1 then None
  else
    let (ids, pat, _) = branches.(0) in
    match pat with
    | Pusual r | Pcons (r, _) ->
        let fds = get_record_fields (State.get_table state) r in
        if List.is_empty fds then None
        else
          (* Expand [Pusual r] into [Pcons(r, pats)] for uniform treatment;
             [Pcons] is returned unchanged by [expand_pusual]. *)
          ( match expand_pusual (List.length ids) pat with
            | Pcons (_, sub_pats) ->
                if List.length sub_pats = List.length fds &&
                   List.for_all
                     (function Prel _ | Pwild -> true | _ -> false)
                     sub_pats
                then Some (r, fds, sub_pats)
                else None
            | _ -> None )
    | _ -> None

(*s Core expression printer.
    [env] carries de Bruijn binder names (innermost first). *)

let rec pp_expr state env expr =
  match std_string_expr_value expr with
  | Some value ->
      str "\"" ++ str (py_escape_str value) ++ str "\""
  | None ->
  match expr with
  | MLglob r when is_custom r && String.equal (find_custom r) marker_state_get ->
      str "StateT.get_state()"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_state_pure ->
      str "StateT.pure"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_state_bind ->
      str "StateT.bind"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_state_put ->
      str "StateT.put_state"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_reader_ask ->
      str "lambda __reader_env: __reader_env"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_reader_pure ->
      str "(lambda value: (lambda __reader_env: value))"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_reader_bind ->
      str "(lambda reader_expr, fn_expr: lambda __reader_env: fn_expr(reader_expr(__reader_env))(__reader_env))"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_io_pure ->
      str "IO.pure"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_io_bind ->
      str "IO.bind"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_io_bracket ->
      str "IO.bracket"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_new_mutex ->
      str "Mutex.new()"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_new_channel ->
      str "Channel.new()"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_new_future ->
      str "Future.new()"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_mutex_acquire ->
      str "Mutex.acquire"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_mutex_release ->
      str "Mutex.release"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_channel_send ->
      str "Channel.send"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_channel_receive ->
      str "Channel.receive"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_future_set ->
      str "Future.set_result"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_future_result ->
      str "Future.result"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_future_done ->
      str "Future.done"
  | MLglob r when is_custom r && String.equal (find_custom r) marker_interleave ->
      extraction_diagnostic_error ~detail:(find_custom r) "PYEX043"
  | MLglob r when is_custom r && is_concurrency_marker_string (find_custom r) ->
      extraction_diagnostic_error ~detail:(find_custom r) "PYEX044"
  | MLglob r when is_positive_map_ref r "empty" || is_string_map_ref r "empty" ->
      str "{}"
  | MLglob r when is_positive_set_ref r "empty" || is_string_set_ref r "empty" ->
      str "frozenset()"
  | MLrel n ->
      (* De Bruijn variable: look up the binder name.  The dummy name [_]
         signals an erased binder; emit [__] (the module-level sentinel). *)
      let id = get_db_name n env in
      if Id.equal id dummy_name then str "__" else pp_pyid id
  | MLglob r ->
      pp_glob state r
  | MLdummy Kprop ->
      pp_impossible_expr ()
  | MLdummy _ ->
      str "__"
  | MLexn s ->
      str "raise RuntimeError(\"" ++ str (py_escape_str s) ++ str "\")"
  | MLaxiom s ->
      str "raise NotImplementedError(\"AXIOM TO BE REALIZED: " ++
      str (py_escape_str s) ++ str "\")"
  | MLmagic a ->
      (* Identity coercion — Python is dynamically typed; emit inner expr. *)
      pp_expr state env a
  | MLuint i ->
      str (Uint63.to_string i)
  | MLfloat f ->
      pp_float_lit f
  | MLstring s ->
      str "b\"" ++ str (py_escape_bytes (Pstring.to_string s)) ++ str "\""
  | MLparray _ ->
      extraction_diagnostic_error "PYEX001"
  | MLapp (f, args) ->
      (* Flatten left-associative curried application:
         MLapp(MLapp(f,[a]),[b]) → f(a, b) *)
      let rec collect acc = function
        | MLapp (g, more) -> collect (more @ acc) g
        | head            -> (head, acc)
      in
      let (head, all_args) = collect args f in
      let all_args = List.filter (fun a -> not (is_erased_arg a)) all_args in
      let pp_method_call_expr =
        match head, all_args with
        | MLglob r, self_arg :: method_args -> (
            match lookup_active_method_target (pp_global state Term r) with
            | Some (_class_name, method_name) ->
                Some
                  (pp_expr state env self_arg ++ str "." ++ str method_name ++ str "(" ++
                   prlist_with_sep (fun () -> str ", ") (pp_expr state env) method_args ++
                   str ")")
            | None -> None)
        | _ -> None
      in
      let collection_key kind key =
        pp_collection_key kind (pp_expr state env key)
      in
      let pp_map_app kind r =
        match all_args with
        | [] when is_positive_map_ref r "empty" || is_string_map_ref r "empty" ->
            Some (str "{}")
        | [key; value; mapping] when is_positive_map_ref r "add" || is_string_map_ref r "add" ->
            Some (str "_rocq_map_add(" ++ collection_key kind key ++ str ", " ++
                  pp_expr state env value ++ str ", " ++ pp_expr state env mapping ++ str ")")
        | [key; mapping] when is_positive_map_ref r "remove" || is_string_map_ref r "remove" ->
            Some (str "_rocq_map_remove(" ++ collection_key kind key ++ str ", " ++
                  pp_expr state env mapping ++ str ")")
        | [key; mapping] when is_positive_map_ref r "find" || is_string_map_ref r "find" ->
            Some (pp_expr state env mapping ++ str ".get(" ++ collection_key kind key ++ str ")")
        | [key; mapping] when is_positive_map_ref r "mem" || is_string_map_ref r "mem" ->
            Some (collection_key kind key ++ str " in " ++ pp_expr state env mapping)
        | [mapping] when is_positive_map_ref r "cardinal" || is_string_map_ref r "cardinal" ->
            Some (str "len(" ++ pp_expr state env mapping ++ str ")")
        | [mapping] when is_positive_map_ref r "elements" || is_string_map_ref r "elements" ->
            Some (str "_rocq_map_elements(" ++ pp_expr state env mapping ++ str ")")
        | [fn; mapping; initial] when is_positive_map_ref r "fold" || is_string_map_ref r "fold" ->
            Some (str "_rocq_map_fold(" ++ pp_expr state env fn ++ str ", " ++
                  pp_expr state env mapping ++ str ", " ++ pp_expr state env initial ++ str ")")
        | _ -> None
      in
      let pp_set_app kind r =
        match all_args with
        | [] when is_positive_set_ref r "empty" || is_string_set_ref r "empty" ->
            Some (str "frozenset()")
        | [key; values] when is_positive_set_ref r "add" || is_string_set_ref r "add" ->
            Some (str "_rocq_set_add(" ++ collection_key kind key ++ str ", " ++
                  pp_expr state env values ++ str ")")
        | [key; values] when is_positive_set_ref r "remove" || is_string_set_ref r "remove" ->
            Some (str "_rocq_set_remove(" ++ collection_key kind key ++ str ", " ++
                  pp_expr state env values ++ str ")")
        | [key; values] when is_positive_set_ref r "mem" || is_string_set_ref r "mem" ->
            Some (collection_key kind key ++ str " in " ++ pp_expr state env values)
        | [left; right] when is_positive_set_ref r "union" || is_string_set_ref r "union" ->
            Some (str "(" ++ pp_expr state env left ++ str " | " ++ pp_expr state env right ++ str ")")
        | [left; right] when is_positive_set_ref r "inter" || is_string_set_ref r "inter" ->
            Some (str "(" ++ pp_expr state env left ++ str " & " ++ pp_expr state env right ++ str ")")
        | [left; right] when is_positive_set_ref r "diff" || is_string_set_ref r "diff" ->
            Some (str "(" ++ pp_expr state env left ++ str " - " ++ pp_expr state env right ++ str ")")
        | [values] when is_positive_set_ref r "cardinal" || is_string_set_ref r "cardinal" ->
            Some (str "len(" ++ pp_expr state env values ++ str ")")
        | [values] when is_positive_set_ref r "elements" || is_string_set_ref r "elements" ->
            Some (str "_rocq_set_elements(" ++ pp_expr state env values ++ str ")")
        | [fn; values; initial] when is_positive_set_ref r "fold" || is_string_set_ref r "fold" ->
            Some (str "_rocq_set_fold(" ++ pp_expr state env fn ++ str ", " ++
                  pp_expr state env values ++ str ", " ++ pp_expr state env initial ++ str ")")
        | _ -> None
      in
      let pp_list_app r =
        match all_args with
        | [left; right] when is_std_list_app_ref r ->
            Some (pp_expr state env left ++ str " + " ++ pp_expr state env right)
        | _ -> None
      in
      let pp_std_primitive_compare r =
        match all_args with
        | [left; right]
          when is_std_nat_ref r "eqb" || is_std_positive_ref r "eqb" ||
               is_std_ascii_ref r "eqb" || is_std_string_ref r "eqb" ->
            Some (pp_expr state env left ++ str " == " ++ pp_expr state env right)
        | [left; right]
          when is_std_nat_ref r "leb" || is_std_positive_ref r "leb" ||
               is_std_ascii_ref r "leb" || is_std_string_ref r "leb" ->
            Some (pp_expr state env left ++ str " <= " ++ pp_expr state env right)
        | [left; right]
          when is_std_nat_ref r "ltb" || is_std_positive_ref r "ltb" ||
               is_std_ascii_ref r "ltb" || is_std_string_ref r "ltb" ->
            Some (pp_expr state env left ++ str " < " ++ pp_expr state env right)
        | _ ->
            None
      in
      let pp_collection_expr =
        match head with
        | MLglob r when is_std_primitive_compare_ref r ->
            pp_std_primitive_compare r
        | MLglob r when is_std_list_app_ref r ->
            pp_list_app r
        | MLglob r when is_positive_map_ref r "empty" || is_positive_map_ref r "add" ||
                        is_positive_map_ref r "remove" || is_positive_map_ref r "find" ||
                        is_positive_map_ref r "mem" || is_positive_map_ref r "cardinal" ||
                        is_positive_map_ref r "elements" || is_positive_map_ref r "fold" ->
            pp_map_app `Positive r
        | MLglob r when is_string_map_ref r "empty" || is_string_map_ref r "add" ||
                        is_string_map_ref r "remove" || is_string_map_ref r "find" ||
                        is_string_map_ref r "mem" || is_string_map_ref r "cardinal" ||
                        is_string_map_ref r "elements" || is_string_map_ref r "fold" ->
            pp_map_app `String r
        | MLglob r when is_positive_set_ref r "empty" || is_positive_set_ref r "add" ||
                        is_positive_set_ref r "remove" || is_positive_set_ref r "mem" ||
                        is_positive_set_ref r "union" || is_positive_set_ref r "inter" ||
                        is_positive_set_ref r "diff" || is_positive_set_ref r "cardinal" ||
                        is_positive_set_ref r "elements" || is_positive_set_ref r "fold" ->
            pp_set_app `Positive r
        | MLglob r when is_string_set_ref r "empty" || is_string_set_ref r "add" ||
                        is_string_set_ref r "remove" || is_string_set_ref r "mem" ||
                        is_string_set_ref r "union" || is_string_set_ref r "inter" ||
                        is_string_set_ref r "diff" || is_string_set_ref r "cardinal" ||
                        is_string_set_ref r "elements" || is_string_set_ref r "fold" ->
            pp_set_app `String r
        | _ -> None
      in
      let pp_option_bind_expr opt_expr fn_expr =
        str "(lambda __option_value: None if __option_value is None else (" ++
        pp_expr state env fn_expr ++ str ")(__option_value))(" ++
        pp_expr state env opt_expr ++
        str ")"
      in
      let pp_reader_pure_expr value =
        str "lambda __reader_env: " ++ pp_expr state env value
      in
      let pp_reader_bind_expr reader_expr fn_expr =
        str "lambda __reader_env: (" ++
        pp_expr state env fn_expr ++
        str ")(" ++
        pp_expr state env reader_expr ++
        str "(__reader_env))(__reader_env)"
      in
      let pp_concurrency_expr =
        match marker_of_ast head, all_args with
        | Some marker, [] when String.equal marker marker_new_mutex ->
            Some (str "Mutex.new()")
        | Some marker, [] when String.equal marker marker_new_channel ->
            Some (str "Channel.new()")
        | Some marker, [] when String.equal marker marker_new_future ->
            Some (str "Future.new()")
        | Some marker, [mutex] when String.equal marker marker_mutex_acquire ->
            Some (pp_expr state env mutex ++ str ".acquire()")
        | Some marker, [mutex] when String.equal marker marker_mutex_release ->
            Some (pp_expr state env mutex ++ str ".release()")
        | Some marker, [channel; value] when String.equal marker marker_channel_send ->
            Some (pp_expr state env channel ++ str ".send(" ++ pp_expr state env value ++ str ")")
        | Some marker, [channel] when String.equal marker marker_channel_receive ->
            Some (pp_expr state env channel ++ str ".receive()")
        | Some marker, [future; value] when String.equal marker marker_future_set ->
            Some (pp_expr state env future ++ str ".set_result(" ++ pp_expr state env value ++ str ")")
        | Some marker, [future] when String.equal marker marker_future_result ->
            Some (pp_expr state env future ++ str ".result()")
        | Some marker, [future] when String.equal marker marker_future_done ->
            Some (pp_expr state env future ++ str ".done()")
        | Some marker, _ when String.equal marker marker_interleave ->
            extraction_diagnostic_error ~detail:marker "PYEX043"
        | Some marker, _ when is_concurrency_marker_string marker ->
            extraction_diagnostic_error ~detail:marker "PYEX044"
        | _ ->
            None
      in
      let pp_monad_expr =
        match marker_of_ast head, all_args with
        | Some marker, [value] when String.equal marker marker_state_pure ->
            Some (str "StateT.pure(" ++ pp_expr state env value ++ str ")")
        | Some marker, [m; f] when String.equal marker marker_state_bind ->
            Some (pp_expr state env m ++ str ".bind(" ++ pp_expr state env f ++ str ")")
        | Some marker, [] when String.equal marker marker_state_get ->
            Some (str "StateT.get_state()")
        | Some marker, [new_state] when String.equal marker marker_state_put ->
            Some (str "StateT.put_state(" ++ pp_expr state env new_state ++ str ")")
        | Some marker, [value] when String.equal marker marker_reader_pure ->
            Some (pp_reader_pure_expr value)
        | Some marker, [reader_expr; fn_expr] when String.equal marker marker_reader_bind ->
            Some (pp_reader_bind_expr reader_expr fn_expr)
        | Some marker, [] when String.equal marker marker_reader_ask ->
            Some (str "lambda __reader_env: __reader_env")
        | Some marker, [value] when String.equal marker marker_io_pure ->
            Some (str "IO.pure(" ++ pp_expr state env value ++ str ")")
        | Some marker, [m; f] when String.equal marker marker_io_bind ->
            Some (pp_expr state env m ++ str ".bind(" ++ pp_expr state env f ++ str ")")
        | Some marker, [acquire; release; use] when String.equal marker marker_io_bracket ->
            Some (str "IO.bracket(" ++ pp_expr state env acquire ++ str ", " ++
                  pp_expr state env release ++ str ", " ++ pp_expr state env use ++ str ")")
        | Some marker, _ when String.equal marker marker_io_run ->
            extraction_diagnostic_error ~detail:marker "PYEX042"
        | Some marker, [opt_expr; fn_expr] when String.equal marker marker_option_bind ->
            Some (pp_option_bind_expr opt_expr fn_expr)
        | Some marker, _
          when String.length marker >= String.length "__PYMONAD_" &&
               String.equal "__PYMONAD_"
                 (String.sub marker 0 (String.length "__PYMONAD_")) ->
            extraction_diagnostic_error ~detail:marker "PYEX010"
        | _ ->
            None
      in
      let pp_default_app =
        (* Parenthesise the function if it is itself a lambda expression,
           since [lambda x: e] has very low precedence in Python. *)
        let pp_head =
          match head with
          | MLlam _ -> str "(" ++ pp_expr state env head ++ str ")"
          | _ -> pp_expr state env head
        in
        if List.is_empty all_args then pp_head
        else
          pp_head ++ str "(" ++
          prlist_with_sep (fun () -> str ", ") (pp_expr state env) all_args ++
          str ")"
      in
      let pp_special =
        match pp_method_call_expr with
        | Some _ as pp -> pp
        | None ->
            (match pp_collection_expr with
             | Some _ as pp -> pp
             | None ->
                 (match pp_concurrency_expr with
                  | Some _ as pp -> pp
                  | None -> pp_monad_expr))
      in
      (match pp_special with
       | Some pp -> pp
       | None -> pp_default_app)
  | MLlam _ as a ->
      (* Collect consecutive lambdas: MLlam(x, MLlam(y, body)) → lambda x, y: body.
         [collect_lams] returns ids innermost-first; reverse for Python source order. *)
      let ids, body = collect_lams a in
      let params = List.map id_of_mlid ids in
      let params, env' = push_vars params env in
      let params = List.rev params in
      if List.is_empty (visible_params params) then pp_expr state env' body
      else pp_lambda params (pp_expr state env' body)
  | MLletin (id, a1, a2) ->
      (* Let binding in expression context: lambda-lift to [(lambda x: a2)(a1)].
         This is safe in pure functional code and avoids statement–expression
         impedance.  Statement-level [Dterm] will get a proper [def]/assignment
         form in the "Wire Dterm and Dfix" task. *)
      let params, env' = push_vars [id_of_mlid id] env in
      let bname = List.hd params in
      let pp_binder =
        if Id.equal bname dummy_name then str "_" else pp_pyid bname
      in
      str "(lambda " ++ pp_binder ++ str ": " ++
      pp_expr state env' a2 ++ str ")(" ++
      pp_expr state env a1 ++ str ")"
  | MLtuple l ->
      (* Tuple literal: (a, b, c) *)
      str "(" ++
      prlist_with_sep (fun () -> str ", ") (pp_expr state env) l ++
      str ")"
  | MLcons (_, r, []) when is_std_option_none_ref r ->
      str "None"
  | MLcons (_, r, [value]) when is_std_option_some_ref r ->
      pp_expr state env value
  | MLcons (_, r, []) when is_std_list_nil_ref r ->
      str "[]"
  | MLcons (_, r, [head; tail]) when is_std_list_cons_ref r ->
      str "[" ++ pp_expr state env head ++ str "] + " ++
      pp_expr state env tail
  | MLcons (_, r, [left; right]) when is_std_prod_pair_ref r ->
      str "(" ++ pp_expr state env left ++ str ", " ++
      pp_expr state env right ++ str ")"
  | MLcons (_, r, []) when is_std_bool_true_ref r ->
      str "True"
  | MLcons (_, r, []) when is_std_bool_false_ref r ->
      str "False"
  | MLcons (_, r, []) when is_std_nat_zero_ref r ->
      str "0"
  | MLcons (_, r, [n]) when is_std_nat_succ_ref r ->
      let pp_n = pp_expr state env n in
      if needs_numeric_constructor_parens n then
        str "(" ++ pp_n ++ str ") + 1"
      else
        pp_n ++ str " + 1"
  | MLcons (_, r, []) when is_std_positive_xh_ref r ->
      str "1"
  | MLcons (_, r, [p]) when is_std_positive_xo_ref r ->
      let pp_p = pp_expr state env p in
      if needs_numeric_constructor_parens p then
        str "(" ++ pp_p ++ str ") * 2"
      else
        pp_p ++ str " * 2"
  | MLcons (_, r, [p]) when is_std_positive_xi_ref r ->
      let pp_p = pp_expr state env p in
      if needs_numeric_constructor_parens p then
        str "(" ++ pp_p ++ str ") * 2 + 1"
      else
        pp_p ++ str " * 2 + 1"
  | MLcons (_, r, []) when is_std_N_zero_ref r ->
      str "0"
  | MLcons (_, r, [p]) when is_std_N_pos_ref r ->
      pp_expr state env p
  | MLcons (_, r, []) when is_std_Z_zero_ref r ->
      str "0"
  | MLcons (_, r, [p]) when is_std_Z_pos_ref r ->
      pp_expr state env p
  | MLcons (_, r, [p]) when is_std_Z_neg_ref r ->
      let pp_p = pp_expr state env p in
      if needs_numeric_constructor_parens p then
        str "-(" ++ pp_p ++ str ")"
      else
        str "-" ++ pp_p
  | MLcons (_, r, [num; den]) when is_std_Q_make_ref r ->
      str "Fraction(" ++ pp_expr state env num ++ str ", " ++
      pp_expr state env den ++ str ")"
  | MLcons (_, r, []) when is_std_string_empty_ref r ->
      str "\"\""
  | MLcons (_, r, [head; tail]) when is_std_string_cons_ref r ->
      pp_expr state env head ++ str " + " ++ pp_expr state env tail
  | MLcons (t_, r, args) when is_std_ascii_cons_ref r ->
      if List.length args <> 8 then extraction_diagnostic_error "PYEX008"
      else
        (match std_ascii_expr_value (MLcons (t_, r, args)) with
         | Some code -> pp_ascii_char_lit code
         | None ->
             (* Non-static bit arguments — wrap the int helper in chr() to
                preserve the str representation contract for ascii values. *)
             let pp_bool_bit a =
               match std_bool_expr a with
               | Some true -> str "True"
               | Some false -> str "False"
               | None -> pp_expr state env a
             in
             str "chr(_rocq_ascii_to_int(" ++
             prlist_with_sep (fun () -> str ", ") pp_bool_bit args ++
             str "))")
  | MLcons (_, r, []) when is_std_byte_cons_ref r ->
      str (string_of_int (Option.get (std_byte_constructor_value r)))
  | MLcons (_, r, args) ->
      let cons_name = str_cons state r in
      if is_coinductive (State.get_table state) r then
        (* Coinductive values stay lazy via a packet wrapper.  The wrapper is
           callable for one-step forcing, and stream-shaped packets also expose
           [__iter__] so finite consumers can use [itertools.islice]. *)
        let step_expr =
          if String.equal "" cons_name then
            (* erased coinductive constructor — unusual but valid *)
            (match args with
             | [a] -> pp_expr state env a
             | _ -> extraction_diagnostic_error "PYEX023")
          else if List.is_empty args then
            str cons_name ++ str "()"
          else
            str cons_name ++ str "(" ++
            prlist_with_sep (fun () -> str ", ") (pp_expr state env) args ++
            str ")"
        in
        str (coinductive_name state r) ++ str "(lambda: " ++ step_expr ++ str ")"
      else if String.equal "" cons_name then
        (* Singleton / newtype erasure — emit content directly *)
        ( match args with
          | []  -> str "None"
          | [a] -> pp_expr state env a
          | _   ->
              str "(" ++
              prlist_with_sep (fun () -> str ", ") (pp_expr state env) args ++
              str ")" )
      else
        let fds = get_record_fields (State.get_table state) r in
        if not (List.is_empty fds) then
          (* Record type: keyword arguments [T(field=a, field2=b)] *)
          str cons_name ++ str "(" ++
          prlist_with_sep (fun () -> str ", ")
            (fun (i, a) ->
               pp_field_name state r fds i ++ str "=" ++ pp_expr state env a)
            (List.mapi (fun i a -> (i, a)) args) ++
          str ")"
        else if List.is_empty args then
          (* Zero-argument constructor.  Inline-custom constructors are plain
             literals (["True"], ["None"], ["0"], etc.) that are emitted
             as-is.  Native dataclass constructors need ["()"] appended so
             they produce an instance rather than a class reference. *)
          if is_inline_custom r then str cons_name
          else str cons_name ++ str "()"
        else
          (* Standard type: positional arguments [ConstrName(a, b)] *)
          str cons_name ++ str "(" ++
          prlist_with_sep (fun () -> str ", ") (pp_expr state env) args ++
          str ")"
  | MLcase (ty, scrutinee, branches) ->
      (* Boolean match: [if cond then e1 else e2] → Python ternary [e1 if cond else e2].
         Detect the bool pattern by checking if the two arms are mapped to
         ["True"] and ["False"] via [Extract Inductive bool]. *)
      let is_bool =
        Array.length branches = 2 &&
        ( let (_, p0, _) = branches.(0) in
          let (_, p1, _) = branches.(1) in
          is_bool_patt p0 "True" && is_bool_patt p1 "False" )
      in
      if is_std_string_type ty then
        pp_std_string_match_expr state env scrutinee branches
      else if is_std_ascii_type ty then
        pp_std_ascii_match_expr state env scrutinee branches
      else if is_std_nat_type ty then
        pp_std_nat_match_expr state env scrutinee branches
      else if is_std_positive_type ty then
        pp_std_positive_match_expr state env scrutinee branches
      else if is_std_N_type ty then
        pp_std_N_match_expr state env scrutinee branches
      else if is_std_Z_type ty then
        pp_std_Z_match_expr state env scrutinee branches
      else if is_std_Q_type ty then
        pp_std_Q_match_expr state env scrutinee branches
      else if is_std_option_type ty then
        pp_std_option_match_expr state env scrutinee branches
      else if is_std_list_type ty then
        pp_std_list_match_expr state env scrutinee branches
      else if is_std_prod_type ty then
        pp_std_prod_match_expr state env scrutinee branches
      else if is_std_bool_type ty then
        pp_std_bool_match_expr state env scrutinee branches
      else if is_bool then
        let (_, _, body_true)  = branches.(0) in
        let (_, _, body_false) = branches.(1) in
        pp_ternary_operand_expr state env body_true  ++ str " if " ++
        pp_ternary_operand_expr state env scrutinee  ++ str " else " ++
        pp_ternary_operand_expr state env body_false
      else if is_custom_match branches then
        (* Custom match function: [Extract Inductive T => "t" [...] "fn"].
           Emit as [fn(branch_thunk_0, branch_thunk_1, …, scrutinee)] where
           each branch becomes a Python lambda.  This handles types like
           [nat → int] whose constructors do not exist as Python patterns. *)
        pp_custom_match_expr state env (find_custom_match branches) scrutinee branches
      else
        (* Record projection: single-branch match over a record inductive.
           Instead of [match scrutinee: case Ctor(f0,f1): body], emit the
           lambda-lifted attribute form:
             (lambda f0, f1, …: body)(scrutinee.f0, scrutinee.f1, …)
           This mirrors [MLletin]'s lambda-lift strategy and stays at
           expression level without a [match] statement.
           [record_proj_info] handles the full detection and edge-case guarding.

           Key subtlety: branch [ids] only contains BOUND binders.  Wildcard
           sub-patterns in Rocq source compile to real variable binders (with
           fresh generated names) rather than [Pwild] in MiniML — so in
           practice every sub-pattern is [Prel k] and every field has a bound
           name.  The [Pwild] arm is kept as a defence against future changes. *)
        (match record_proj_info state branches with
        | Some (r, fds, sub_pats) ->
            let (ids, _, body) = branches.(0) in
            let n_fds = List.length fds in
            (* Push all binders (innermost-first for push_vars). *)
            let _, env' = push_vars (List.rev_map id_of_mlid ids) env in
            (* Derive the lambda parameter name for field position [i]:
               [Prel k] → look up the name at depth [k] in [env']
               [Pwild]  → synthetic [_e<i>] (can't repeat [_] in a lambda) *)
            let pp_param_for_pos i =
              match List.nth sub_pats i with
              | Prel k -> pp_pyid (get_db_name k env')
              | _      -> str ("_e" ^ string_of_int i)
            in
            (* The common record-projection case extracts one bound field.
               Emit direct attribute access so generated accessors are plain
               Python instead of lambda applications. *)
            let direct_projection =
              match body with
              | MLrel k ->
                  List.find_mapi
                    (fun i -> function
                       | Prel k' when Int.equal k k' -> Some i
                       | _ -> None)
                    sub_pats
              | _ -> None
            in
            let pp_scr = pp_expr state env scrutinee in
            (match direct_projection with
             | Some i ->
                 pp_scr ++ str "." ++ pp_field_name state r fds i
             | None ->
                 let pp_arg i = pp_scr ++ str "." ++ pp_field_name state r fds i in
                 str "(lambda " ++
                 prlist_with_sep (fun () -> str ", ") pp_param_for_pos
                   (List.init n_fds (fun i -> i)) ++
                 str ": " ++
                 pp_expr state env' body ++
                 str ")(" ++
                 prlist_with_sep (fun () -> str ", ") pp_arg
                   (List.init n_fds (fun i -> i)) ++
                 str ")")
        | None ->
        (* General match: emit Python [match]/[case] statement.
           This is a statement in Python, so it only works correctly when the
           surrounding [Dterm] or [MLletin] emits it inside a function body.
           The "Wire Dterm and Dfix" task handles that lifting.  For now we
           emit the raw statement form so the structure is visible. *)
        let pp_branch (ids, pat, body) =
          (* [push_vars] wants ids innermost-first (de Bruijn order).
             [collect_lams]/branch ids come outermost-first, so reverse. *)
          let _ids', env' = push_vars (List.rev_map id_of_mlid ids) env in
          (* Expand any top-level [Pusual r] into [Pcons(r,[Prel n;…;Prel 1])]
             so that [pp_pattern] can resolve every binder via [env'] alone. *)
          let pp_pat  = pp_pattern state env' (expand_pusual (List.length ids) pat) in
          let pp_body = pp_expr state env' body in
          str "    case " ++ pp_pat ++ str ":" ++ fnl () ++
          str "        " ++ pp_body
        in
        (* Append a catch-all arm unless the last explicit branch is already
           a wildcard.  Python [match] silently falls through with no result
           when no arm matches, so terminate explicitly.  The exact terminator
           is chosen by [pp_unreachable_fallback]: [assert_never] when the
           fallback is statically provable, [assert False] when it is only
           runtime-unreachable. *)
        let catch_all =
          if has_wildcard_last branches then mt ()
          else fnl () ++ pp_unreachable_fallback_for state ty 4
        in
        let pp_scrutinee =
          if type_is_coinductive state ty then
            str "coforce(" ++ pp_expr state env scrutinee ++ str ")"
          else
            pp_expr state env scrutinee
        in
        str "match " ++ pp_scrutinee ++ str ":" ++ fnl () ++
        prlist_with_sep fnl pp_branch (Array.to_list branches) ++
        catch_all)
  | MLfix (i, ids, defs) ->
      (* Mutual fixpoint: push all n names into the env (reversed, per the
         extraction convention so [ids.(0)] becomes the outermost binder),
         emit one [def] per function, then reference [ids.(i)] as the value.
         Python [def] is a statement, not an expression, so this is
         well-formed only at statement level; the "Wire Dterm and Dfix" task
         handles hoisting it into a function body when needed. *)
      let n = Array.length ids in
      (* [ids] is already [Id.t array]; no [id_of_mlid] needed. *)
      let id_list = List.rev (Array.to_list ids) in
      let names_rev, env' = push_vars id_list env in
      (* [List.rev names_rev] restores original array order:
         [name_arr.(j)] is the (possibly renamed) identifier for [ids.(j)]. *)
      let name_arr = Array.of_list (List.rev names_rev) in
      let pp_one j =
        (* Unwrap the body's lambdas to emit:
             def fname(p1, p2, ...):
                 return body
           If the body has no lambdas, emit [def fname(): return body]. *)
        let lam_ids, body = collect_lams defs.(j) in
        let params = List.map id_of_mlid lam_ids in
        let params', env'' = push_vars params env' in
        str "def " ++ pp_pyid name_arr.(j) ++ str "(" ++
        pp_param_list (List.rev params') ++
        str "):" ++ fnl () ++
        str "    return " ++ pp_expr state env'' body
      in
      prlist_with_sep fnl pp_one (List.init n (fun j -> j)) ++ fnl () ++
      pp_pyid name_arr.(i)

and pp_branch_lambda state env ids body =
  let ids', env' = push_vars (List.rev_map id_of_mlid ids) env in
  let params = List.rev ids' in
  match visible_params params with
  | [] -> str "lambda: " ++ pp_expr state env' body
  | params ->
      str "lambda " ++
      prlist_with_sep (fun () -> str ", ") pp_param params ++
      str ": " ++ pp_expr state env' body

and pp_branch_thunk_expr state env ids body =
  str "(" ++ pp_branch_lambda state env ids body ++ str ")()"

and pp_branch_call_expr state env ids body args =
  str "(" ++ pp_branch_lambda state env ids body ++ str ")(" ++
  prlist_with_sep (fun () -> str ", ") (fun arg -> arg) args ++ str ")"

and pp_branch_or_impossible_expr state env = function
  | Some (ids, body) -> pp_branch_thunk_expr state env ids body
  | None -> pp_impossible_expr ()

and pp_branch_or_fallback_expr state env fallback = function
  | Some (ids, body) -> pp_branch_thunk_expr state env ids body
  | None -> fallback ()

and pp_ternary_text pp =
  str "(" ++ fnl () ++ indent_pp 4 pp ++ str ")"

and pp_ternary_operand_expr state env expr =
  pp_ternary_text (pp_expr state env expr)

and pp_branch_pair_expr state env ids body pair_expr =
  str "(lambda __pair: (" ++ pp_branch_lambda state env ids body ++
  str ")(__pair[0], __pair[1]))(" ++ pair_expr ++ str ")"

and pp_std_string_match_expr state env scrutinee branches =
  let empty_arm = ref None in
  let cons_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_string_empty_ref r ->
        set_once empty_arm (ids, body)
    | Pcons (r, _) when is_std_string_cons_ref r ->
        set_once cons_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error
          ~detail:"unsupported nested String.string pattern shape"
          "PYEX040"
  in
  Array.iter classify branches;
  let fallback () =
    pp_branch_or_impossible_expr state env !wildcard_arm
  in
  let pp_empty =
    pp_branch_or_fallback_expr state env fallback !empty_arm
  in
  let pp_cons =
    match !cons_arm with
    | Some (ids, body) ->
        pp_branch_pair_expr state env ids body
          (str "(__s[0], __s[1:])")
    | None -> fallback ()
  in
  str "(lambda __s: " ++ pp_ternary_text pp_empty ++ str " if __s == \"\" else " ++
  pp_ternary_text pp_cons ++ str ")(" ++ pp_expr state env scrutinee ++ str ")"

and pp_std_ascii_match_expr state env scrutinee branches =
  let ascii_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, _) when is_std_ascii_cons_ref r ->
        set_once ascii_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error
          ~detail:"unsupported Ascii.ascii pattern shape"
          "PYEX040"
  in
  Array.iter classify branches;
  match !ascii_arm with
  | Some (ids, body) ->
      str "(lambda __bits: (" ++
      pp_branch_lambda state env ids body ++
      str ")(__bits[0], __bits[1], __bits[2], __bits[3], __bits[4], __bits[5], __bits[6], __bits[7]))(_rocq_ascii_bits(" ++
      pp_expr state env scrutinee ++ str "))"
  | None ->
      pp_branch_or_impossible_expr state env !wildcard_arm

and pp_std_nat_match_expr state env scrutinee branches =
  let zero_arm = ref None in
  let succ_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_nat_zero_ref r ->
        set_once zero_arm (ids, body)
    | Pcons (r, _) when is_std_nat_succ_ref r ->
        set_once succ_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported nat pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let fallback () =
    pp_branch_or_impossible_expr state env !wildcard_arm
  in
  let pp_zero =
    pp_branch_or_fallback_expr state env fallback !zero_arm
  in
  let pp_succ =
    match !succ_arm with
    | Some (ids, body) ->
        pp_branch_call_expr state env ids body [str "__n - 1"]
    | None -> fallback ()
  in
  str "(lambda __n: " ++ pp_ternary_text pp_zero ++ str " if __n == 0 else " ++
  pp_ternary_text pp_succ ++ str " if __n > 0 else _rocq_numeric_domain_error(\"nat\", __n))(" ++
  pp_expr state env scrutinee ++ str ")"

and pp_std_positive_match_expr state env scrutinee branches =
  let xh_arm = ref None in
  let xo_arm = ref None in
  let xi_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_positive_xh_ref r ->
        set_once xh_arm (ids, body)
    | Pcons (r, _) when is_std_positive_xo_ref r ->
        set_once xo_arm (ids, body)
    | Pcons (r, _) when is_std_positive_xi_ref r ->
        set_once xi_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error
          ~detail:"unsupported positive pattern shape"
          "PYEX040"
  in
  Array.iter classify branches;
  let fallback () =
    pp_branch_or_impossible_expr state env !wildcard_arm
  in
  let pp_xh =
    pp_branch_or_fallback_expr state env fallback !xh_arm
  in
  let pp_xo =
    match !xo_arm with
    | Some (ids, body) ->
        pp_branch_call_expr state env ids body [str "__p // 2"]
    | None -> fallback ()
  in
  let pp_xi =
    match !xi_arm with
    | Some (ids, body) ->
        pp_branch_call_expr state env ids body [str "(__p - 1) // 2"]
    | None -> fallback ()
  in
  str "(lambda __p: _rocq_numeric_domain_error(\"positive\", __p) if __p <= 0 else " ++
  pp_ternary_text pp_xh ++ str " if __p == 1 else " ++
  pp_ternary_text pp_xo ++ str " if __p % 2 == 0 else " ++ pp_ternary_text pp_xi ++ str ")(" ++
  pp_expr state env scrutinee ++ str ")"

and pp_std_N_match_expr state env scrutinee branches =
  let zero_arm = ref None in
  let pos_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_N_zero_ref r ->
        set_once zero_arm (ids, body)
    | Pcons (r, _) when is_std_N_pos_ref r ->
        set_once pos_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported N pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let fallback () =
    pp_branch_or_impossible_expr state env !wildcard_arm
  in
  let pp_zero =
    pp_branch_or_fallback_expr state env fallback !zero_arm
  in
  let pp_pos =
    match !pos_arm with
    | Some (ids, body) -> pp_branch_call_expr state env ids body [str "__n"]
    | None -> fallback ()
  in
  str "(lambda __n: " ++ pp_ternary_text pp_zero ++ str " if __n == 0 else " ++
  pp_ternary_text pp_pos ++ str " if __n > 0 else _rocq_numeric_domain_error(\"N\", __n))(" ++
  pp_expr state env scrutinee ++ str ")"

and pp_std_Z_match_expr state env scrutinee branches =
  let zero_arm = ref None in
  let pos_arm = ref None in
  let neg_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_Z_zero_ref r ->
        set_once zero_arm (ids, body)
    | Pcons (r, _) when is_std_Z_pos_ref r ->
        set_once pos_arm (ids, body)
    | Pcons (r, _) when is_std_Z_neg_ref r ->
        set_once neg_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported Z pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let fallback () =
    pp_branch_or_impossible_expr state env !wildcard_arm
  in
  let pp_zero =
    pp_branch_or_fallback_expr state env fallback !zero_arm
  in
  let pp_pos =
    match !pos_arm with
    | Some (ids, body) -> pp_branch_call_expr state env ids body [str "__z"]
    | None -> fallback ()
  in
  let pp_neg =
    match !neg_arm with
    | Some (ids, body) -> pp_branch_call_expr state env ids body [str "-__z"]
    | None -> fallback ()
  in
  str "(lambda __z: " ++ pp_ternary_text pp_zero ++ str " if __z == 0 else " ++
  pp_ternary_text pp_pos ++ str " if __z > 0 else " ++ pp_ternary_text pp_neg ++ str ")(" ++
  pp_expr state env scrutinee ++ str ")"

and pp_std_Q_match_expr state env scrutinee branches =
  let q_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, _) when is_std_Q_make_ref r ->
        set_once q_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported Q pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  match !q_arm with
  | Some (ids, body) ->
      str "(lambda __q: (" ++
      pp_branch_lambda state env ids body ++
      str ")(__q.numerator, __q.denominator))(" ++
      pp_expr state env scrutinee ++ str ")"
  | None ->
      pp_branch_or_impossible_expr state env !wildcard_arm

and pp_std_bool_match_expr state env scrutinee branches =
  let true_arm = ref None in
  let false_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_bool_true_ref r ->
        set_once true_arm (ids, body)
    | Pcons (r, []) when is_std_bool_false_ref r ->
        set_once false_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported bool pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let fallback () =
    pp_branch_or_impossible_expr state env !wildcard_arm
  in
  let pp_true =
    pp_branch_or_fallback_expr state env fallback !true_arm
  in
  let pp_false =
    pp_branch_or_fallback_expr state env fallback !false_arm
  in
  pp_ternary_text pp_true ++ str " if " ++ pp_ternary_operand_expr state env scrutinee ++ str " else " ++ pp_ternary_text pp_false

and pp_std_option_match_expr state env scrutinee branches =
  let none_arm = ref None in
  let some_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_option_none_ref r ->
        set_once none_arm (ids, body)
    | Pcons (r, _) when is_std_option_some_ref r ->
        set_once some_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported option pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let fallback () =
    pp_branch_or_impossible_expr state env !wildcard_arm
  in
  let pp_none =
    pp_branch_or_fallback_expr state env fallback !none_arm
  in
  let pp_some =
    match !some_arm with
    | Some (ids, body) ->
        pp_branch_call_expr state env ids body [str "__option"]
    | None -> fallback ()
  in
  str "(" ++ fnl () ++
  str "    lambda __option: (" ++ fnl () ++
  indent_pp 8 pp_none ++
  str "        if __option is None else" ++ fnl () ++
  indent_pp 8 pp_some ++
  str "    )" ++ fnl () ++
  str ")(" ++ pp_expr state env scrutinee ++ str ")"

and pp_std_list_match_expr state env scrutinee branches =
  let nil_arm = ref None in
  let cons_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_list_nil_ref r ->
        set_once nil_arm (ids, body)
    | Pcons (r, _) when is_std_list_cons_ref r ->
        set_once cons_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported list pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let fallback () =
    pp_branch_or_impossible_expr state env !wildcard_arm
  in
  let pp_nil =
    pp_branch_or_fallback_expr state env fallback !nil_arm
  in
  let pp_cons =
    match !cons_arm with
    | Some (ids, body) ->
        pp_branch_call_expr state env ids body [str "__list[0]"; str "__list[1:]"]
    | None -> fallback ()
  in
  str "(lambda __list: " ++ pp_ternary_text pp_nil ++ str " if __list == [] else " ++
  pp_ternary_text pp_cons ++ str ")(" ++ pp_expr state env scrutinee ++ str ")"

and pp_std_prod_match_expr state env scrutinee branches =
  let pair_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, _) when is_std_prod_pair_ref r ->
        set_once pair_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported prod pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  match !pair_arm with
  | Some (ids, body) ->
      pp_branch_pair_expr state env ids body (pp_expr state env scrutinee)
  | None ->
      pp_branch_or_impossible_expr state env !wildcard_arm

(*s Custom-match expression emitter.
    When [Extract Inductive T => "t" [conA conB] "fn"] supplies a match
    function, case analysis on [T] cannot use Python [match]/[case] (the
    constructors don't exist as Python patterns).  Instead we emit a
    functional application:

      (fn)(lambda: arm0_body, lambda x: arm1_body, …, scrutinee)

    Each branch becomes a [lambda] thunk.  The convention — branch thunks
    first, scrutinee last — must match what the user wrote in ["fn"].
    For example, for [nat → int]:

      Extract Inductive nat => "int" ["0" "(lambda x: x+1)"]
        "(lambda fO, fS, n: fO() if n == 0 else fS(n-1))".

    yields  [(lambda fO, fS, n: fO() if n == 0 else fS(n-1))(lambda: 0,
              lambda m_: ..., n)].

    [pp_custom_match_expr] is mutually recursive with [pp_expr] because it
    calls [pp_expr] to emit each branch body. *)

and pp_custom_match_expr state env s scrutinee branches =
  let pp_arm (ids, _pat, body) =
    let ids', env' = push_vars (List.rev_map id_of_mlid ids) env in
    let params = List.rev ids' in
    match params with
    | [] ->
        (* No binders: unit thunk. *)
        str "lambda: " ++ pp_expr state env' body
    | _  ->
        pp_lambda params (pp_expr state env' body)
  in
  str "(" ++ str s ++ str ")(" ++
  prlist_with_sep (fun () -> str ", ") pp_arm (Array.to_list branches) ++
  str ", " ++
  pp_expr state env scrutinee ++
  str ")"

let rec unwrap_fix = function
  | MLfix (i, ids, defs) -> Some (i, ids, defs)
  | MLmagic a            -> unwrap_fix a
  | _                    -> None

let collect_app f args =
  let rec collect acc = function
    | MLapp (g, more) -> collect (more @ acc) g
    | head            -> (head, acc)
  in
  collect args f

let pp_multiline_enclosed indent open_pp close_pp items =
  let arg_pfx = indent_string (indent + 4) in
  let close_pfx = indent_string indent in
  open_pp ++ fnl () ++
  prlist_with_sep
    (fun () -> str "," ++ fnl ())
    (fun item -> str arg_pfx ++ item)
    items ++
  str "," ++ fnl () ++ str close_pfx ++ close_pp

let pp_multiline_items indent head items =
  pp_multiline_enclosed indent (head ++ str "(") (str ")") items

let pp_multiline_tuple indent items =
  pp_multiline_enclosed indent (str "(") (str ")") items

let rec pp_statement_expr state env indent = function
  | MLmagic a ->
      pp_statement_expr state env indent a
  | MLcons (_, r, [value]) when is_std_option_some_ref r ->
      pp_statement_expr state env indent value
  | MLcons (_, r, [left; right]) when is_std_prod_pair_ref r ->
      pp_multiline_tuple indent
        [ pp_statement_expr state env (indent + 4) left;
          pp_statement_expr state env (indent + 4) right ]
  | MLcons (_, r, args)
    when not (type_is_coinductive state (Tglob (get_ind r, []))) &&
         not (String.equal "" (str_cons state r)) &&
         not (List.is_empty (get_record_fields (State.get_table state) r)) ->
      let cons_name = str_cons state r in
      let fds = get_record_fields (State.get_table state) r in
      pp_multiline_items indent (str cons_name)
        (List.mapi
           (fun i a ->
              pp_field_name state r fds i ++ str "=" ++
              pp_statement_expr state env (indent + 4) a)
           args)
  | MLcons (_, r, args)
    when not (type_is_coinductive state (Tglob (get_ind r, []))) &&
         not (String.equal "" (str_cons state r)) &&
         List.is_empty (get_record_fields (State.get_table state) r) &&
         not (is_std_list_cons_ref r) &&
         not (is_std_ascii_cons_ref r) &&
         List.length args >= 2 ->
      (* Non-record, non-coinductive sum constructor with two or more arguments.
         Using pp_multiline_items produces stable output (one arg per line with
         trailing comma) that ruff will not reformat regardless of indentation
         depth.  Single-argument constructors fall through to pp_expr since
         they always fit on one line.  Cons and Ascii have their own lowerings
         in pp_expr; the guards above prevent this case from shadowing them. *)
      let cons_name = str_cons state r in
      pp_multiline_items indent (str cons_name)
        (List.map (pp_statement_expr state env (indent + 4)) args)
  | MLapp (f, args) -> (
      let head, all_args = collect_app f args in
      match head with
      | MLmagic a ->
          pp_statement_expr state env indent (MLapp (a, all_args))
      | MLletin (id, a1, a2) ->
          pp_statement_expr state env indent (MLletin (id, a1, MLapp (a2, all_args)))
      | _ ->
          let all_args = List.filter (fun a -> not (is_erased_arg a)) all_args in
          let collection_key kind key =
            pp_collection_key kind (pp_expr state env key)
          in
          let call callee args =
            pp_multiline_items indent callee
              (List.map (pp_statement_expr state env (indent + 4)) args)
          in
          match head, all_args with
          | MLglob r, [left; right] when is_std_list_app_ref r ->
              pp_expr state env left ++ str " + " ++ pp_expr state env right
          | MLglob r, [key; value; mapping]
            when is_positive_map_ref r "add" || is_string_map_ref r "add" ->
              let kind =
                if is_positive_map_ref r "add" then `Positive else `String
              in
              pp_multiline_items indent (str "_rocq_map_add")
                [ collection_key kind key;
                  pp_statement_expr state env (indent + 4) value;
                  pp_statement_expr state env (indent + 4) mapping ]
          | MLglob r, [key; mapping]
            when is_positive_map_ref r "remove" || is_string_map_ref r "remove" ->
              let kind =
                if is_positive_map_ref r "remove" then `Positive else `String
              in
              pp_multiline_items indent (str "_rocq_map_remove")
                [ collection_key kind key;
                  pp_statement_expr state env (indent + 4) mapping ]
          | MLglob r, [key; value; values]
            when is_positive_set_ref r "add" || is_string_set_ref r "add" ->
              let kind =
                if is_positive_set_ref r "add" then `Positive else `String
              in
              pp_multiline_items indent (str "_rocq_set_add")
                [ collection_key kind key;
                  pp_statement_expr state env (indent + 4) value;
                  pp_statement_expr state env (indent + 4) values ]
          | MLglob r, [key; values]
            when is_positive_set_ref r "remove" || is_string_set_ref r "remove" ->
              let kind =
                if is_positive_set_ref r "remove" then `Positive else `String
              in
              pp_multiline_items indent (str "_rocq_set_remove")
                [ collection_key kind key;
                  pp_statement_expr state env (indent + 4) values ]
          | _, _ when List.length all_args >= 3 ->
              call (pp_expr state env head) all_args
          | _ ->
              pp_expr state env (MLapp (f, args)))
  | expr ->
      (match expr with
       | MLtuple items when List.length items >= 2 ->
           pp_multiline_tuple indent
             (List.map (pp_statement_expr state env (indent + 4)) items)
       | _ ->
      pp_expr state env expr
      )

(*s Statement-level body printer for Python [def] bodies.
    Inside a [def], [return <match-stmt>] is invalid Python because [match] is
    a statement, not an expression.  This printer recurses into [MLcase]
    branches so that each leaf arm gets its own [return], producing valid
    Python.  The boolean ternary (two-arm bool match) is still treated as an
    expression and wrapped in a single [return].  Custom-match expressions
    are pure expressions and likewise wrapped in a single [return].

    [indent] is the indentation level (in spaces) at which the [match] keyword
    itself will appear.  [case] arms are emitted at [indent + 4], and their
    bodies at [indent + 8].  The first line of the emitted text does NOT include
    leading whitespace — the caller is responsible for that. *)

let rec pp_return_body state env indent = function
  | MLdummy Kprop ->
      pp_impossible_stmt ()
  | MLmagic a ->
      pp_return_body state env indent a
  | MLcase (ty, scrutinee, branches) as expr ->
      let is_bool =
        Array.length branches = 2 &&
        ( let (_, p0, _) = branches.(0) in
          let (_, p1, _) = branches.(1) in
          is_bool_patt p0 "True" && is_bool_patt p1 "False" )
      in
      if is_std_string_type ty then
        pp_std_string_return_body state env indent scrutinee branches
      else if is_std_N_type ty then
        pp_std_N_return_body state env indent scrutinee branches
      else if is_std_list_type ty then
        pp_std_list_return_body state env indent scrutinee branches
      else if is_std_option_type ty then
        pp_std_option_return_body state env indent scrutinee branches
      else if is_std_prod_type ty then
        pp_std_prod_return_body state env indent scrutinee branches
      else if is_std_ascii_type ty ||
              is_std_nat_type ty || is_std_positive_type ty ||
              is_std_Z_type ty || is_std_Q_type ty then
        str "return " ++ pp_expr state env expr
      else if is_std_bool_type ty then
        pp_std_bool_return_body state env indent scrutinee branches
      else if is_bool then
        (* Ternary — valid expression; a single [return] suffices. *)
        str "return " ++ pp_expr state env expr
      else if is_custom_match branches then
        (* Custom-match — also a pure expression. *)
        str "return " ++
        pp_custom_match_expr state env (find_custom_match branches) scrutinee branches
      else
        (* Record projection — [record_proj_info] detects the single-branch
           record match; [pp_expr] emits the lambda-lift form, which is a
           valid expression so a single [return] suffices. *)
        ( match record_proj_info state branches with
          | Some _ ->
              str "return " ++ pp_expr state env expr
          | None ->
              (* General match: put [return] inside each arm so the branch is
                 a valid statement.  [case] must be indented inside the [match]
                 block; [body] must be indented inside the [case] block. *)
              let case_pfx = String.make (indent + 4) ' ' in
              let body_pfx = String.make (indent + 8) ' ' in
              let pp_branch (ids, pat, body) =
                let _ids', env' = push_vars (List.rev_map id_of_mlid ids) env in
                let pp_pat  = pp_pattern state env' (expand_pusual (List.length ids) pat) in
                str case_pfx ++ str "case " ++ pp_pat ++ str ":" ++ fnl () ++
                str body_pfx ++ pp_return_body state env' (indent + 8) body
              in
              (* Same catch-all logic as the expression-context path: append
                 an explicit unreachable terminator unless the last arm is
                 already a wildcard.  [pp_unreachable_fallback] picks the
                 statically-provable vs runtime-only form. *)
              let catch_all =
                if has_wildcard_last branches then mt ()
                else fnl () ++ pp_unreachable_fallback_for state ty (indent + 4)
              in
              let pp_scrutinee =
                if type_is_coinductive state ty then
                  str "coforce(" ++ pp_expr state env scrutinee ++ str ")"
                else
                  pp_expr state env scrutinee
              in
              str "match " ++ pp_scrutinee ++ str ":" ++ fnl () ++
              prlist_with_sep fnl pp_branch (Array.to_list branches) ++
              catch_all )
  | MLaxiom _ | MLexn _ | MLparray _ as expr ->
      (* These emit [raise …], which is a valid Python statement but NOT a
         valid expression.  Emit bare — no [return] prefix. *)
      pp_expr state env expr
  | MLletin (id, a1, a2) -> (
      (* Let-bound local fixpoint inside a function body:
           let f = fix ... in body
         becomes nested [def] statements followed by the returned body.  This
         is the shape produced by Program Fixpoint's [Fix_sub] helper. *)
      match unwrap_fix a1 with
      | None ->
          let params, env' = push_vars [id_of_mlid id] env in
          let bname = List.hd params in
          let pfx = String.make indent ' ' in
          let pp_binder =
            if Id.equal bname dummy_name then str "_" else pp_pyid bname
          in
          pp_binder ++ str " = " ++ pp_statement_expr state env indent a1 ++
          fnl () ++ str pfx ++ pp_return_body state env' indent a2
      | Some (i, ids, defs) ->
      let params, env' = push_vars [id_of_mlid id] env in
      let bname = List.hd params in
      let def_pfx = String.make indent ' ' in
      let pp_defs, selected = pp_fix_statement state env indent i ids defs in
      let pp_alias =
        if Id.equal bname selected then mt ()
        else
          fnl () ++ str def_pfx ++ pp_pyid bname ++
          str " = " ++ pp_pyid selected
      in
      pp_defs ++ pp_alias ++ fnl () ++ fnl () ++ str def_pfx ++
      pp_return_body state env' indent a2 )
  | MLapp (f, args) -> (
      match collect_app f args with
      | MLmagic a, all_args ->
          pp_return_body state env indent (MLapp (a, all_args))
      | MLletin (id, a1, a2), all_args ->
          pp_return_body state env indent (MLletin (id, a1, MLapp (a2, all_args)))
      | head, all_args -> (
          let visible_args =
            List.filter (fun a -> not (is_erased_arg a)) all_args
          in
          let pp_io_bind_statement action next =
            let ids, body = collect_lams next in
            let params, env' = push_vars (List.rev_map id_of_mlid ids) env in
            let params = List.rev (visible_params params) in
            let bind_name = "__io_bind_next" in
            let pfx = String.make indent ' ' in
            let body_pfx = String.make (indent + 4) ' ' in
            let pp_params =
              if List.is_empty params then str "__ignored"
              else pp_param_list params
            in
            str "def " ++ str bind_name ++ str "(" ++ pp_params ++ str "):" ++
            fnl () ++ str body_pfx ++ pp_return_body state env' (indent + 4) body ++
            fnl () ++ fnl () ++ str pfx ++ str "return " ++ pp_expr state env action ++
            str ".bind(" ++ str bind_name ++ str ")"
          in
          let pp_io_lambda_statement name lam =
            let ids, body = collect_lams lam in
            let params, env' = push_vars (List.rev_map id_of_mlid ids) env in
            let params = List.rev (visible_params params) in
            let body_pfx = String.make (indent + 4) ' ' in
            let pp_params =
              if List.is_empty params then str "__ignored"
              else pp_param_list params
            in
            str "def " ++ str name ++ str "(" ++ pp_params ++ str "):" ++
            fnl () ++ str body_pfx ++ pp_return_body state env' (indent + 4) body
          in
          let pp_io_bracket_statement acquire release use =
            let release_name = "__io_bracket_release" in
            let use_name = "__io_bracket_use" in
            let pfx = String.make indent ' ' in
            pp_io_lambda_statement release_name release ++
            fnl () ++ fnl () ++ str pfx ++
            pp_io_lambda_statement use_name use ++
            fnl () ++ fnl () ++ str pfx ++
            str "return IO.bracket(" ++ pp_expr state env acquire ++ str ", " ++
            str release_name ++ str ", " ++ str use_name ++ str ")"
          in
          match marker_of_ast head, visible_args with
          | Some marker, [action; next] when String.equal marker marker_io_bind ->
              pp_io_bind_statement action next
          | Some marker, [acquire; release; use]
            when String.equal marker marker_io_bracket ->
              pp_io_bracket_statement acquire release use
          | _ ->
          match unwrap_fix head with
          | None ->
              (match head with
               | MLlam _ as lam ->
                   let ids, body = collect_lams lam in
                   let visible_args =
                     List.filter (fun a -> not (is_erased_arg a)) all_args
                   in
                   pp_return_arm state env indent ids
                     (List.map (pp_expr state env) visible_args)
                     body
               | _ ->
                   str "return " ++
                   pp_statement_expr state env indent (MLapp (f, args)))
          | Some (i, ids, defs) ->
              let def_pfx = String.make indent ' ' in
              let pp_defs, selected = pp_fix_statement state env indent i ids defs in
              pp_defs ++ fnl () ++ fnl () ++ str def_pfx ++ str "return " ++
              pp_pyid selected ++ str "(" ++
              prlist_with_sep (fun () -> str ", ") (pp_expr state env) visible_args ++
              str ")" ) )
  | MLfix (i, ids, defs) ->
      (* Local fixpoint inside a function body.  Python [def] is a statement,
         so emit nested defs followed by [return selected_name] instead of
         trying to put [def] after [return]. *)
      let def_pfx = String.make indent ' ' in
      let pp_defs, selected = pp_fix_statement state env indent i ids defs in
      pp_defs ++ fnl () ++ fnl () ++ str def_pfx ++ str "return " ++ pp_pyid selected
  | expr ->
      str "return " ++ pp_statement_expr state env indent expr

and pp_return_arm state env indent ids values body =
  let ids', env' = push_vars (List.rev_map id_of_mlid ids) env in
  let params = List.rev ids' in
  let pfx = String.make indent ' ' in
  let rec bindings params values =
    match params, values with
    | [], _ | _, [] -> []
    | param :: params, value :: values ->
        let rest = bindings params values in
        if Id.equal param dummy_name then rest
        else (pp_pyid param ++ str " = " ++ value) :: rest
  in
  match bindings params values with
  | [] ->
      pp_return_body state env' indent body
  | lines ->
      prlist_with_sep (fun () -> fnl () ++ str pfx) (fun line -> line) lines ++
      fnl () ++ str pfx ++ pp_return_body state env' indent body

and pp_return_or_impossible state env indent = function
  | Some (ids, body) -> pp_return_arm state env indent ids [] body
  | None -> pp_impossible_stmt ()

and pp_std_list_return_body state env indent scrutinee branches =
  let nil_arm = ref None in
  let cons_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_list_nil_ref r ->
        set_once nil_arm (ids, body)
    | Pcons (r, _) when is_std_list_cons_ref r ->
        set_once cons_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported list pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let pfx = String.make indent ' ' in
  let body_pfx = String.make (indent + 4) ' ' in
  let fallback () =
    pp_return_or_impossible state env (indent + 4) !wildcard_arm
  in
  let pp_nil =
    match !nil_arm with
    | Some (ids, body) -> pp_return_arm state env (indent + 4) ids [] body
    | None -> fallback ()
  in
  let pp_cons =
    match !cons_arm with
    | Some (ids, body) ->
        pp_return_arm state env indent ids
          [str "__list[0]"; str "__list[1:]"]
          body
    | None ->
        pp_return_or_impossible state env indent !wildcard_arm
  in
  str "__list = " ++ pp_expr state env scrutinee ++ fnl () ++
  str pfx ++ str "if __list == []:" ++ fnl () ++
  str body_pfx ++ pp_nil ++ fnl () ++
  str pfx ++ pp_cons

and pp_std_option_return_body state env indent scrutinee branches =
  let none_arm = ref None in
  let some_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_option_none_ref r ->
        set_once none_arm (ids, body)
    | Pcons (r, _) when is_std_option_some_ref r ->
        set_once some_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported option pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let pfx = String.make indent ' ' in
  let body_pfx = String.make (indent + 4) ' ' in
  let fallback body_indent =
    pp_return_or_impossible state env body_indent !wildcard_arm
  in
  let pp_none =
    match !none_arm with
    | Some (ids, body) -> pp_return_arm state env (indent + 4) ids [] body
    | None -> fallback (indent + 4)
  in
  let pp_some =
    match !some_arm with
    | Some (ids, body) ->
        pp_return_arm state env indent ids [str "__option"] body
    | None ->
        pp_return_or_impossible state env indent !wildcard_arm
  in
  str "__option = " ++ pp_expr state env scrutinee ++ fnl () ++
  str pfx ++ str "if __option is None:" ++ fnl () ++
  str body_pfx ++ pp_none ++ fnl () ++
  str pfx ++ pp_some

and pp_std_bool_return_body state env indent scrutinee branches =
  let true_arm = ref None in
  let false_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_bool_true_ref r ->
        set_once true_arm (ids, body)
    | Pcons (r, []) when is_std_bool_false_ref r ->
        set_once false_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported bool pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let pfx = String.make indent ' ' in
  let body_pfx = String.make (indent + 4) ' ' in
  let fallback body_indent =
    pp_return_or_impossible state env body_indent !wildcard_arm
  in
  let pp_true =
    match !true_arm with
    | Some (ids, body) -> pp_return_arm state env (indent + 4) ids [] body
    | None -> fallback (indent + 4)
  in
  let pp_false =
    match !false_arm with
    | Some (ids, body) -> pp_return_arm state env indent ids [] body
    | None ->
        pp_return_or_impossible state env indent !wildcard_arm
  in
  str "if " ++ pp_expr state env scrutinee ++ str ":" ++ fnl () ++
  str body_pfx ++ pp_true ++ fnl () ++
  str pfx ++ pp_false

and pp_std_string_return_body state env indent scrutinee branches =
  let empty_arm = ref None in
  let cons_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_string_empty_ref r ->
        set_once empty_arm (ids, body)
    | Pcons (r, _) when is_std_string_cons_ref r ->
        set_once cons_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported string pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let pfx = String.make indent ' ' in
  let body_pfx = String.make (indent + 4) ' ' in
  let fallback () =
    pp_return_or_impossible state env (indent + 4) !wildcard_arm
  in
  let pp_empty =
    match !empty_arm with
    | Some (ids, body) -> pp_return_arm state env (indent + 4) ids [] body
    | None -> fallback ()
  in
  let pp_cons =
    match !cons_arm with
    | Some (ids, body) ->
        pp_return_arm state env indent ids
          [str "__s[0]"; str "__s[1:]"]
          body
    | None ->
        pp_return_or_impossible state env indent !wildcard_arm
  in
  str "__s = " ++ pp_expr state env scrutinee ++ fnl () ++
  str pfx ++ str "if __s == \"\":" ++ fnl () ++
  str body_pfx ++ pp_empty ++ fnl () ++
  str pfx ++ pp_cons

and pp_std_N_return_body state env indent scrutinee branches =
  let zero_arm = ref None in
  let pos_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, []) when is_std_N_zero_ref r ->
        set_once zero_arm (ids, body)
    | Pcons (r, _) when is_std_N_pos_ref r ->
        set_once pos_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported N pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  let pfx = String.make indent ' ' in
  let body_pfx = String.make (indent + 4) ' ' in
  let fallback branch_indent =
    pp_return_or_impossible state env branch_indent !wildcard_arm
  in
  let pp_zero =
    match !zero_arm with
    | Some (ids, body) -> pp_return_arm state env (indent + 4) ids [] body
    | None -> fallback (indent + 4)
  in
  let pp_pos =
    match !pos_arm with
    | Some (ids, body) -> pp_return_arm state env (indent + 4) ids [str "__n"] body
    | None -> fallback (indent + 4)
  in
  str "__n = " ++ pp_expr state env scrutinee ++ fnl () ++
  str pfx ++ str "if __n == 0:" ++ fnl () ++
  str body_pfx ++ pp_zero ++ fnl () ++
  str pfx ++ str "if __n > 0:" ++ fnl () ++
  str body_pfx ++ pp_pos ++ fnl () ++
  str pfx ++ str "return _rocq_numeric_domain_error(\"N\", __n)"

and pp_std_prod_return_body state env indent scrutinee branches =
  let pair_arm = ref None in
  let wildcard_arm = ref None in
  let classify (ids, pat, body) =
    match expand_pusual (List.length ids) pat with
    | Pcons (r, _) when is_std_prod_pair_ref r ->
        set_once pair_arm (ids, body)
    | Pwild ->
        set_once wildcard_arm (ids, body)
    | _ ->
        extraction_diagnostic_error ~detail:"unsupported prod pattern shape" "PYEX040"
  in
  Array.iter classify branches;
  match !pair_arm with
  | Some (ids, body) ->
      str "__pair = " ++ pp_expr state env scrutinee ++ fnl () ++
      str (String.make indent ' ') ++
      pp_return_arm state env indent ids [str "__pair[0]"; str "__pair[1]"] body
  | None ->
      pp_return_or_impossible state env indent !wildcard_arm

and pp_fix_statement state env indent i ids defs =
  let n = Array.length ids in
  let id_list = List.rev (Array.to_list ids) in
  let names_rev, env' = push_vars id_list env in
  let name_arr = Array.of_list (List.rev names_rev) in
  let def_pfx = String.make indent ' ' in
  let body_pfx = String.make (indent + 4) ' ' in
  let pp_one j =
    let lam_ids, body = collect_lams defs.(j) in
    let params = List.map id_of_mlid lam_ids in
    let params', env'' = push_vars params env' in
    str "def " ++ pp_pyid name_arr.(j) ++ str "(" ++
    pp_param_list (List.rev params') ++
    str "):" ++ fnl () ++
    str body_pfx ++ pp_return_body state env'' (indent + 4) body
  in
  prlist_with_sep (fun () -> fnl () ++ str def_pfx) pp_one
    (List.init n (fun j -> j)),
  name_arr.(i)

(*s TypeVar name helper. *)

(** Python [TypeVar] name for the [i]-th type parameter (0-based).
    Indices 0–25 map to [_A]–[_Z]; beyond that we use [_T<i>] so we
    never run out of names for exotic mutual inductives. *)
let typevar_name i =
  if i < 26 then Printf.sprintf "_%c" (Char.chr (65 + i))
  else Printf.sprintf "_T%d" i

let ml_typevar_index i =
  if i <= 0 then 0 else i - 1

let ml_typevar_name i =
  typevar_name (ml_typevar_index i)

let add_once x xs =
  if List.mem x xs then xs else x :: xs

let rec collect_typevars acc = function
  | Tarr (t1, t2) ->
      collect_typevars (collect_typevars acc t1) t2
  | Tglob (_, args) ->
      List.fold_left collect_typevars acc args
  | Tvar i | Tvar' i ->
      add_once (ml_typevar_index i) acc
  | Tdummy _ | Tunknown | Taxiom | Tmeta _ ->
      acc

let typevars_of_type typ =
  List.sort_uniq compare (collect_typevars [] typ)

let pp_typevar_decls ids =
  if List.is_empty ids then mt ()
  else
    prlist_with_sep mt
      (fun i ->
         let tv = typevar_name i in
         str tv ++ str " = TypeVar(\"" ++ str tv ++ str "\")" ++ fnl ())
      ids ++ fnl () ++ fnl ()

type protocol_spec = {
  protocol_name : string;
  protocol_arg_types : ml_type list;
  protocol_ret_type : ml_type;
}

let protocol_variance_name base i =
  Printf.sprintf "%s_P%d_contra" base i

let protocol_return_name base =
  base ^ "_R_co"

(*s Type annotation emitter.
    Converts an [ml_type] to a Python annotation fragment.  The annotations
    are used only for documentation and optional static analysis —
    correctness of generated code never depends on them (Python is
    dynamically typed).

    PEP 649 deferred annotation evaluation (Python 3.14t default) means
    forward and recursive references work without explicit quoting. *)

let rec pp_type_with state pp_tvar = function
  | Tarr (t1, t2) ->
      (* Arrow type → [Callable[[arg], ret]].  Curried chains stay nested:
         [Tarr(a, Tarr(b, c))] → [Callable[[a], Callable[[b], c]]]. *)
      str "Callable[[" ++ pp_type_with state pp_tvar t1 ++ str "], " ++
      pp_type_with state pp_tvar t2 ++ str "]"
  | Tglob (r, args) ->
      (* Look up the Python name: prefer the custom string if present
         (e.g. [nat → "int"]), otherwise use the mangled global name.
         An empty custom string signals singleton/transparent erasure
         (e.g. [option → ""]); fall back to [object] in that case.
         Inductive type names are capitalized to match the class names
         emitted by [pp_ind_decl] (PEP 8 PascalCase convention). *)
      let name =
        if is_std_real_type_ref r then extraction_diagnostic_error "PYEX041"
        else if is_std_bool_type_ref r then "bool"
        else if is_std_string_type_ref r then "str"
        else if is_prim_string_type_ref r then "bytes"
        else if is_std_ascii_type_ref r then "str"
        else if is_std_byte_type_ref r then "int"
        else if is_std_nat_type_ref r || is_std_positive_type_ref r ||
                is_std_N_type_ref r || is_std_Z_type_ref r then "int"
        else if is_std_Q_type_ref r then "Fraction"
        else if is_std_option_type_ref r then
          (match args with
           | [arg] ->
               Pp.string_of_ppcmds (pp_type_with state pp_tvar arg) ^ " | None"
           | _ -> "object | None")
        else if is_std_list_type_ref r then "list"
        else if is_std_prod_type_ref r then "tuple"
        else if is_positive_map_type_ref r then "dict[int, object]"
        else if is_positive_set_type_ref r then "frozenset[int]"
        else if is_string_map_type_ref r then "dict[str, object]"
        else if is_string_set_type_ref r then "frozenset[str]"
        else if is_custom r && String.equal (find_custom r) marker_io_type then "IO"
        else if is_custom r && String.equal (find_custom r) marker_mutex_type then "Mutex"
        else if is_custom r && String.equal (find_custom r) marker_channel_type then "Channel"
        else if is_custom r && String.equal (find_custom r) marker_future_type then "Future"
        else if is_custom r then find_custom r
        else
          let n = pp_global state Term r in
          let open GlobRef in
          match r.glob with
          | IndRef _ -> capitalize_first n
          | _        -> n
      in
      if String.contains name '|' then str name
      else if String.equal "" name then str "object"
      else if is_std_list_type_ref r then
        (match args with
         | [arg] -> str "list[" ++ pp_type_with state pp_tvar arg ++ str "]"
         | _ -> str "list[object]")
      else if is_std_prod_type_ref r then
        (match args with
         | [left; right] ->
             str "tuple[" ++ pp_type_with state pp_tvar left ++ str ", " ++
             pp_type_with state pp_tvar right ++ str "]"
         | _ -> str "tuple[object, object]")
      else if is_positive_map_type_ref r then
        (match args with
         | [arg] -> str "dict[int, " ++ pp_type_with state pp_tvar arg ++ str "]"
         | _ -> str "dict[int, object]")
      else if is_string_map_type_ref r then
        (match args with
         | [arg] -> str "dict[str, " ++ pp_type_with state pp_tvar arg ++ str "]"
         | _ -> str "dict[str, object]")
      else if is_custom r && String.equal (find_custom r) marker_io_type then
        (match args with
         | [arg] -> str "IO[" ++ pp_type_with state pp_tvar arg ++ str "]"
         | _ -> str "IO[object]")
      else if is_custom r && String.equal (find_custom r) marker_mutex_type then
        str "Mutex"
      else if is_custom r && String.equal (find_custom r) marker_channel_type then
        (match args with
         | [arg] -> str "Channel[" ++ pp_type_with state pp_tvar arg ++ str "]"
         | _ -> str "Channel[object]")
      else if is_custom r && String.equal (find_custom r) marker_future_type then
        (match args with
         | [arg] -> str "Future[" ++ pp_type_with state pp_tvar arg ++ str "]"
         | _ -> str "Future[object]")
      else if is_positive_set_type_ref r || is_string_set_type_ref r then str name
      else if List.is_empty args then str name
      else
        str name ++ str "[" ++
        prlist_with_sep (fun () -> str ", ") (pp_type_with state pp_tvar) args ++
        str "]"
  | Tvar i | Tvar' i ->
      (* Emit the TypeVar name corresponding to the [i]-th type parameter.
         The TypeVar declaration itself is emitted by [pp_ind_decl]. *)
      str (pp_tvar i)
  | Tdummy _ | Tunknown | Taxiom | Tmeta _ ->
      str "object"

let pp_type state typ =
  pp_type_with state ml_typevar_name typ

let rec type_ends_in_io = function
  | Tarr (_, ret) -> type_ends_in_io ret
  | Tglob (r, [value]) when is_custom r && String.equal (find_custom r) marker_io_type ->
      Some value
  | _ ->
      None

let rec arrow_type args ret =
  match args with
  | [] -> ret
  | arg :: rest -> Tarr (arg, arrow_type rest ret)

let pp_protocol_decl state spec =
  let n = List.length spec.protocol_arg_types in
  let pvars =
    List.init n (fun i -> protocol_variance_name spec.protocol_name i)
  in
  let rvar = protocol_return_name spec.protocol_name in
  let pp_variance =
    prlist_with_sep mt
      (fun tv ->
         str tv ++ str " = TypeVar(\"" ++ str tv ++
         str "\", contravariant=True)" ++ fnl ())
      pvars ++
    str rvar ++ str " = TypeVar(\"" ++ str rvar ++
    str "\", covariant=True)" ++ fnl ()
  in
  let pp_generic_args =
    pvars @ [rvar]
  in
  let pp_call_arg i tv =
    str "arg" ++ int i ++ str ": " ++ str tv
  in
  pp_variance ++
  str "class " ++ str spec.protocol_name ++ str "(Protocol[" ++
  prlist_with_sep (fun () -> str ", ") str pp_generic_args ++
  str "]):" ++ fnl () ++
  str "    def __call__(self, " ++
  prlist_with_sep (fun () -> str ", ") (fun i ->
    pp_call_arg i (List.nth pvars i)) (List.init n Fun.id) ++
  (if Int.equal n 0 then mt () else str ", /") ++
  str ") -> " ++ str rvar ++ str ": ..." ++ fnl () ++ fnl ()

let protocol_annotation state spec =
  str spec.protocol_name ++ str "[" ++
  prlist_with_sep (fun () -> str ", ") (pp_type state)
    (spec.protocol_arg_types @ [spec.protocol_ret_type]) ++
  str "]"

let make_protocol_spec name i typ =
  let args, ret = type_decomp typ in
  {
    protocol_name =
      Printf.sprintf "_%sArg%dFn" (capitalize_first name) i;
    protocol_arg_types = args;
    protocol_ret_type = ret;
  }

let signature_data state name typ =
  let tvars = typevars_of_type typ in
  let local_tvar_name i =
    let ordinal = ml_typevar_index i in
    Printf.sprintf "_%sT%d" (capitalize_first name) ordinal
  in
  let pp_term_type typ =
    pp_type_with state local_tvar_name typ
  in
  let pp_term_typevar_decls =
    if List.is_empty tvars then mt ()
    else
      prlist_with_sep mt
        (fun i ->
           let tv = local_tvar_name (i + 1) in
           str tv ++ str " = TypeVar(\"" ++ str tv ++ str "\")" ++ fnl ())
        tvars ++ fnl () ++ fnl ()
  in
  let args, ret = type_decomp typ in
  let protocols = ref [] in
  let annotate_arg i arg_ty =
    match arg_ty with
    | Tarr _ ->
        let spec = make_protocol_spec name i arg_ty in
        protocols := !protocols @ [spec];
        str spec.protocol_name ++ str "[" ++
        prlist_with_sep (fun () -> str ", ") pp_term_type
          (spec.protocol_arg_types @ [spec.protocol_ret_type]) ++
        str "]"
    | _ ->
        pp_term_type arg_ty
  in
  let arg_annots =
    List.mapi annotate_arg args
  in
  let protocol_pp =
    prlist_with_sep mt (pp_protocol_decl state) !protocols
  in
  (pp_term_typevar_decls ++ protocol_pp, arg_annots, pp_term_type ret)

let pp_def_signature ?(is_async=false) name params ret_annot =
  let def_prefix = if is_async then str "async def " else str "def " in
  if List.length params >= 2 then
    def_prefix ++ str name ++ str "(" ++ fnl () ++
    prlist_with_sep
      (fun () -> str "," ++ fnl ())
      (fun (param, annot) -> str "    " ++ param ++ str ": " ++ annot)
      params ++
    str "," ++ fnl () ++
    str ") -> " ++ ret_annot ++ str ":"
  else
    def_prefix ++ str name ++ str "(" ++
    prlist_with_sep
      (fun () -> str ", ")
      (fun (param, annot) -> param ++ str ": " ++ annot)
      params ++
    str ") -> " ++ ret_annot ++ str ":"

let pp_unannotated_def_signature ?(is_async=false) name params ret_annot =
  let def_prefix = if is_async then str "async def " else str "def " in
  def_prefix ++ str name ++ str "(" ++ pp_param_list params ++ str ") -> " ++
  ret_annot ++ str ":"

let annotated_params_opt params annots =
  if Int.equal (List.length params) (List.length annots) then
    Some (List.mapi (fun i param -> (pp_param param, List.nth annots i)) params)
  else
    None

(*s Python term declaration emitter.
    Detects a lambda-headed RHS and promotes it to a [def]; non-lambda
    RHS stays as a simple assignment. *)

(** Emit either [def name(p1,…): return body] or [name = expr], choosing
    based on whether [a] has leading lambdas. *)
let pp_function_wrapper state env name a typ =
  let args, _ret = type_decomp typ in
  let n = List.length args in
  if Int.equal n 0 then None
  else
    let pp_prefix, arg_annots, ret_annot = signature_data state name typ in
    let arg_names =
      if Int.equal n 1 then ["x"]
      else List.init n (fun i -> "arg" ^ string_of_int i)
    in
    let pp_call =
      match a with
      | MLapp (f, app_args) ->
          let head, all_args = collect_app f app_args in
          let visible_args =
            List.filter (fun a -> not (is_erased_arg a)) all_args
          in
          let pp_head =
            match head with
            | MLlam _ -> str "(" ++ pp_expr state env head ++ str ")"
            | _       -> pp_expr state env head
          in
          pp_head ++ str "(" ++
          prlist_with_sep (fun () -> str ", ") (pp_expr state env) visible_args ++
          (if List.is_empty visible_args then mt () else str ", ") ++
          prlist_with_sep (fun () -> str ", ")
            (fun i -> pp_pyname (List.nth arg_names i))
            (List.init n Fun.id) ++
          str ")"
      | _ ->
          str "(" ++ pp_expr state env a ++ str ")(" ++
          prlist_with_sep (fun () -> str ", ")
            (fun i -> pp_pyname (List.nth arg_names i))
            (List.init n Fun.id) ++
          str ")"
    in
    let pp_signature =
      pp_def_signature name
        (List.mapi (fun i arg -> (pp_pyname arg, List.nth arg_annots i)) arg_names)
        ret_annot
    in
    Some (
      pp_prefix ++
      pp_signature ++ fnl () ++
      str "    return " ++ pp_call ++ fnl ()
    )

let pp_io_term_decl state env name a typ ret_typ =
  let builder_name = "_io_" ^ name in
  let args, _io_ret = type_decomp typ in
  let facade_typ = arrow_type args ret_typ in
  let facade_prefix, facade_arg_annots, facade_ret_annot =
    signature_data state name facade_typ
  in
  let builder_prefix, builder_arg_annots, builder_ret_annot =
    signature_data state builder_name typ
  in
  let facade_call pp_arg args =
    str "return await " ++ str builder_name ++ str "(" ++
    prlist_with_sep (fun () -> str ", ")
      pp_arg
      args ++
    str ").run()"
  in
  let lam_ids, body = collect_lams a in
  match lam_ids with
  | [] ->
      let builder =
        match pp_function_wrapper state env builder_name a typ with
        | Some pp -> pp
        | None ->
            builder_prefix ++
            str builder_name ++ str ": " ++ builder_ret_annot ++
            str " = " ++ pp_expr state env a ++ fnl ()
      in
      if List.is_empty args then
        builder ++ fnl () ++ fnl () ++ facade_prefix ++
        pp_def_signature ~is_async:true name [] facade_ret_annot ++ fnl () ++
        str "    return await " ++ str builder_name ++ str ".run()" ++ fnl ()
      else
        let arg_names =
          if Int.equal (List.length args) 1 then ["x"]
          else List.init (List.length args) (fun i -> "arg" ^ string_of_int i)
        in
        builder ++ fnl () ++ fnl () ++ facade_prefix ++
        pp_def_signature ~is_async:true name
          (List.mapi
             (fun i arg -> (pp_pyname arg, List.nth facade_arg_annots i))
             arg_names)
          facade_ret_annot ++ fnl () ++
        str "    " ++ facade_call pp_pyname arg_names ++ fnl ()
  | _ ->
      let params = List.map id_of_mlid lam_ids in
      let params', env' = push_vars params env in
      let visible_params_rev = List.rev (visible_params params') in
      let builder_params =
        annotated_params_opt visible_params_rev builder_arg_annots
      in
      let facade_params =
        annotated_params_opt visible_params_rev facade_arg_annots
      in
      let builder =
        builder_prefix ++
        (match builder_params with
         | Some params -> pp_def_signature builder_name params builder_ret_annot
         | None ->
             pp_unannotated_def_signature builder_name (List.rev params')
               builder_ret_annot) ++ fnl () ++
        str "    " ++ pp_return_body state env' 4 body ++ fnl ()
      in
      builder ++ fnl () ++ fnl () ++ facade_prefix ++
      (match facade_params with
       | Some params -> pp_def_signature ~is_async:true name params facade_ret_annot
       | None ->
           pp_unannotated_def_signature ~is_async:true name (List.rev params')
             facade_ret_annot) ++ fnl () ++
      str "    " ++ facade_call pp_pyid visible_params_rev ++ fnl ()

let pp_top_level_record_value state env name typ r args =
  let fields = get_record_fields (State.get_table state) r in
  let pp_field_arg (index, value) =
    str "    " ++ pp_field_name state r fields index ++ str "=" ++ pp_expr state env value
  in
  str name ++ str ": " ++ pp_type state typ ++ str " = " ++
  str (pp_global state Cons r) ++ str "(" ++ fnl () ++
  prlist_with_sep
    (fun () -> str "," ++ fnl ())
    pp_field_arg
    (List.mapi (fun index value -> (index, value)) args) ++
  str "," ++ fnl () ++
  str ")" ++ fnl ()

let uses_native_record_constructor state r =
  let fields = get_record_fields (State.get_table state) r in
  not (List.is_empty fields) && not (is_std_Q_make_ref r)

let pp_term_decl state env name a typ =
  let lam_ids, body = collect_lams a in
  let pp_prefix, arg_annots, ret_annot = signature_data state name typ in
  match type_ends_in_io typ with
  | Some ret_typ ->
      pp_io_term_decl state env name a typ ret_typ
  | None ->
  if List.is_empty lam_ids then
    (* Non-function value: simple assignment — unless the body is a [raise]
       expression (MLaxiom / MLexn / MLparray), which cannot appear as the
       RHS of an assignment.  In that case emit the statement bare. *)
    ( match a with
      | MLaxiom _ | MLexn _ | MLparray _ ->
          pp_expr state env a ++ fnl ()
      | MLcons (_, r, args) when uses_native_record_constructor state r ->
          pp_prefix ++ pp_top_level_record_value state env name typ r args
      | _ ->
          (match pp_function_wrapper state env name a typ with
          | Some pp -> pp
          | None ->
              pp_prefix ++
              str name ++ str ": " ++ pp_type state typ ++
              str " = " ++ pp_expr state env a ++ fnl ()) )
  else
    let params = List.map id_of_mlid lam_ids in
    let params', env' = push_vars params env in
    let visible_params_rev = List.rev (visible_params params') in
    let missing = List.length arg_annots - List.length visible_params_rev in
    if missing > 0 then
      let synthetic_ids =
        List.init missing (fun i -> Id.of_string ("arg" ^ string_of_int i))
      in
      let existing_count = List.length params in
      let _, env'' = push_vars (params @ synthetic_ids) env in
      let existing_params =
        List.mapi
          (fun i param -> (pp_param param, List.nth arg_annots i))
          visible_params_rev
      in
      let synthetic_signature_params =
        List.mapi
          (fun i synthetic_id ->
             (pp_pyid synthetic_id,
              List.nth arg_annots (List.length visible_params_rev + i)))
          synthetic_ids
      in
      let synthetic_args =
        List.init missing (fun i -> MLrel (existing_count + missing - i))
      in
      let body_app =
        MLapp (body, synthetic_args)
      in
      pp_prefix ++
      pp_def_signature name (existing_params @ synthetic_signature_params) ret_annot ++
      fnl () ++
      str "    " ++ pp_return_body state env'' 4 body_app ++ fnl ()
    else
      let pp_params = annotated_params_opt visible_params_rev arg_annots in
      pp_prefix ++
      (match pp_params with
       | Some params -> pp_def_signature name params ret_annot
       | None ->
           pp_unannotated_def_signature name (List.rev params') ret_annot) ++
      fnl () ++
      (* [indent=4]: the body is indented by 4 spaces inside the def; [case] arms
         at 8, case bodies at 12.  The "    " prefix handles the first line only;
         [pp_return_body] generates the rest with absolute column positions. *)
      str "    " ++ pp_return_body state env' 4 body ++ fnl ()

let pp_method_signature ?(is_async=false) name params ret_annot =
  let def_prefix = if is_async then str "async def " else str "def " in
  if List.length params >= 1 then
    def_prefix ++ str name ++ str "(" ++ fnl () ++
    str "    self," ++ fnl () ++
    prlist_with_sep
      (fun () -> str "," ++ fnl ())
      (fun (param, annot) -> str "    " ++ param ++ str ": " ++ annot)
      params ++
    str "," ++ fnl () ++
    str ") -> " ++ ret_annot ++ str ":"
  else
    def_prefix ++ str name ++ str "(self) -> " ++ ret_annot ++ str ":"

let pp_unannotated_method_signature ?(is_async=false) name params ret_annot =
  let def_prefix = if is_async then str "async def " else str "def " in
  let pp_tail =
    if List.is_empty params then
      mt ()
    else
      str ", " ++
      prlist_with_sep (fun () -> str ", ") pp_param params
  in
  def_prefix ++ str name ++ str "(self" ++ pp_tail ++ str ") -> " ++
  ret_annot ++ str ":"

let pp_method_term_decl state env method_name a typ =
  let lam_ids, body = collect_lams a in
  let args, ret = type_decomp typ in
  match lam_ids, args with
  | _ :: _, _ :: method_args ->
      let method_typ = arrow_type method_args ret in
      let pp_prefix, arg_annots, ret_annot =
        signature_data state method_name method_typ
      in
      let params = List.map id_of_mlid lam_ids in
      let params', env' = push_vars params env in
      let visible_params_rev = List.rev (visible_params params') in
      (match visible_params_rev with
       | [] -> extraction_diagnostic_error "PYEX040"
       | self_param :: method_params ->
           let self_alias = Pp.string_of_ppcmds (pp_param self_param) in
           let pp_params = annotated_params_opt method_params arg_annots in
           pp_prefix ++
           (match pp_params with
            | Some params -> pp_method_signature method_name params ret_annot
            | None ->
                pp_unannotated_method_signature method_name method_params ret_annot) ++
           fnl () ++
           (if String.equal self_alias "self" then mt ()
            else str "    " ++ str self_alias ++ str " = self" ++ fnl ()) ++
           str "    " ++ pp_return_body state env' 4 body ++ fnl ())
  | _ -> extraction_diagnostic_error "PYEX040"

let underscore_prefix name =
  match String.index_opt name '_' with
  | None -> None
  | Some index when index > 0 -> Some (String.sub name 0 index)
  | Some _ -> None

let consistent_field_prefix field_names =
  match List.filter_map underscore_prefix field_names with
  | [] -> None
  | prefix :: rest when List.for_all (String.equal prefix) rest -> Some prefix
  | _ -> None

let method_target_of_term_decl state record_class_prefixes record_field_names = function
  | Dterm (r, _a, typ) -> (
      if is_custom r then None else
      let args, _ret = type_decomp typ in
      match args with
      | [] -> None
      | Tglob (self_ref, _) :: _ ->
          let class_name = capitalize_first (pp_global state Term self_ref) in
          let source_name = pp_global state Term r in
          if List.mem source_name record_field_names then None
          else
            (match List.assoc_opt class_name record_class_prefixes, String.index_opt source_name '_' with
             | Some prefix, Some index when String.equal prefix (String.sub source_name 0 index) ->
                 let method_name =
                   String.sub source_name (index + 1)
                     (String.length source_name - index - 1)
                 in
                 Some (class_name, method_name)
             | _ -> None)
      | _ :: _ -> None)
  | _ -> None

let record_class_name_of_decl state = function
  | Dind ind -> (
      match ind.ind_kind with
      | Record _ when Array.length ind.ind_packets > 0 ->
          Some (capitalize_first (pp_global state Term ind.ind_packets.(0).ip_typename_ref))
      | _ -> None)
  | _ -> None

let record_field_names_of_decl state = function
  | Dind ind -> (
      match ind.ind_kind with
      | Record fields when Array.length ind.ind_packets > 0 ->
          let packet = ind.ind_packets.(0) in
          let ind_kn = kn_of_ind packet.ip_typename_ref in
          Some
            (List.filter_map
               (function
                 | Some r' -> Some (pp_global_with_key state Term ind_kn r')
                 | None -> None)
               fields)
      | _ -> None)
  | _ -> None

let record_class_prefix_of_decl state decl =
  match record_class_name_of_decl state decl, record_field_names_of_decl state decl with
  | Some class_name, Some field_names -> Option.map (fun prefix -> (class_name, prefix)) (consistent_field_prefix field_names)
  | _ -> None

(*s Inductive type emission as Python dataclasses.
    Each live constructor becomes a frozen [@dataclass] class whose fields
    are the constructor arguments.  Record types use the declared field
    names; all others use positional names [arg0], [arg1], ….

    Standard and Coinductive inductives also get a shared base class so that
    Python's [match]/[case] discriminator can be written against the type
    name rather than enumerating every constructor:

      class Nat:
          pass

      @dataclass(frozen=True)
      class Nat_O(Nat):
          pass

      @dataclass(frozen=True)
      class Nat_S(Nat):
          arg0: nat          # typed via pp_type

    Records have a single constructor that *is* the type, so no separate
    base class is emitted there. *)

(** Emit one [@dataclass(frozen=True)] class for constructor [j] of [packet].
    If [base_opt] is [Some base], the class inherits from [base]. *)
let pp_type_shifted_with state shift pp_tvar typ =
  pp_type_with state (fun i -> pp_tvar (max 0 (i - shift))) typ

let pp_type_shifted state shift =
  pp_type_shifted_with state shift typevar_name

let pp_one_cons state ?(typevar_shift=1) ?(pp_tvar=typevar_name)
    ?class_name packet fields_opt base_opt j =
  let cname =
    match class_name with
    | Some name -> name
    | None -> pp_global state Cons packet.ip_consnames_ref.(j)
  in
  let nargs  = List.length packet.ip_types.(j) in
  let ind_kn = kn_of_ind packet.ip_typename_ref in
  let kernel_names =
    match fields_opt with
    | None   -> cons_arg_names_from_kernel packet j nargs
    | Some _ -> []
  in
  let field_name i =
    match fields_opt with
    | Some fds ->
        ( match List.nth fds i with
          | Some r' -> pp_global_with_key state Term ind_kn r'
          | None    -> Printf.sprintf "arg%d" i )
    | None ->
        ( match List.nth kernel_names i with
          | Some name -> name
          | None      -> Printf.sprintf "arg%d" i )
  in
  let pp_bases = match base_opt with
    | None      -> mt ()
    | Some base -> str "(" ++ str base ++ str ")"
  in
  str "@dataclass(frozen=True)" ++ fnl () ++
  str "class " ++ str cname ++ pp_bases ++ str ":" ++ fnl () ++
  ( if nargs = 0 then str "    pass" ++ fnl ()
    else
      prlist_with_sep mt
        (fun i ->
           str "    " ++ str (field_name i) ++ str ": " ++
           pp_type_shifted_with state typevar_shift pp_tvar (List.nth packet.ip_types.(j) i) ++ fnl ())
        (List.init nargs (fun i -> i)) )

let pp_coinductive_wrapper state packet step_type_expr pp_tvar tvars =
  let tname = packet_name state packet in
  let pp_generic_base =
    if List.is_empty tvars then
      mt ()
    else
      str "(Generic[" ++
      prlist_with_sep (fun () -> str ", ") str tvars ++
      str "])"
  in
  let pp_iter =
    match packet_stream_payload_type state packet with
    | None ->
        mt ()
    | Some payload_ty ->
        fnl () ++
        str "    def __iter__(self) -> Iterator[" ++
        pp_type_shifted_with state 2 pp_tvar payload_ty ++ str "]:" ++ fnl () ++
        str "        current = self" ++ fnl () ++
        str "        while True:" ++ fnl () ++
        str "            unfolded = coforce(current)" ++ fnl () ++
        str "            yield unfolded.arg0" ++ fnl () ++
        str "            current = unfolded.arg1" ++ fnl ()
  in
  str "class " ++ str tname ++ pp_generic_base ++ str ":" ++ fnl () ++
  str "    def __init__(self, force: Callable[[], " ++ str step_type_expr ++ str "]) -> None:" ++ fnl () ++
  str "        self._force = force" ++ fnl () ++
  str "    def __call__(self) -> " ++ str step_type_expr ++ str ":" ++ fnl () ++
  str "        return self._force()" ++ fnl () ++
  pp_iter

let pp_ind_decl state (ind : ml_ind) =
  match ind.ind_kind with
  | Singleton ->
      (* Constructor is transparent at expression level; emit a comment. *)
      let p = ind.ind_packets.(0) in
      if p.ip_logical then
        str "# " ++ Id.print p.ip_typename ++ str ": logical inductive" ++ fnl ()
      else if is_custom p.ip_typename_ref || is_std_remapped_type_ref p.ip_typename_ref then
        str "# " ++ Id.print p.ip_typename ++
        str ": remapped to Python primitive" ++ fnl ()
      else
        str "# " ++ Id.print p.ip_typename ++
        str ": singleton inductive, constructor was " ++
        Id.print p.ip_consnames.(0) ++ fnl ()
  | Record fields ->
      let p = ind.ind_packets.(0) in
      if p.ip_logical then
        str "# " ++ Id.print p.ip_typename ++ str ": logical record" ++ fnl ()
      else if is_custom p.ip_typename_ref || is_std_remapped_type_ref p.ip_typename_ref then
        str "# " ++ Id.print p.ip_typename ++
        str ": remapped to Python primitive" ++ fnl ()
      else
        let tvars = List.init ind.ind_nparams typevar_name in
        (* Emit TypeVar declarations once, before the dataclass. *)
        let pp_typevars = pp_typevar_decls (List.init ind.ind_nparams Fun.id) in
        (* For a parameterised record, inherit [Generic[_A, _B, …]] so that
           type-checkers can track type arguments.  Unparameterised records
           keep the plain dataclass (no base class). *)
        let base_opt =
          if List.is_empty tvars then None
          else
            Some ("Generic[" ^ String.concat ", " tvars ^ "]")
        in
        pp_typevars ++
        pp_one_cons state ~typevar_shift:2 p (Some fields) base_opt 0
  | Standard ->
      let tvars = List.init ind.ind_nparams typevar_name in
      (* TypeVar declarations are emitted once, before all packets, because
         mutual inductives share the same type parameters.  For example,
         [tree A] and [forest A] in a mutual definition both use [_A]. *)
      let pp_typevars = pp_typevar_decls (List.init ind.ind_nparams Fun.id) in
      let pp_packet p =
        if p.ip_logical then
          str "# " ++ Id.print p.ip_typename ++ str ": logical inductive" ++ fnl ()
        else if is_custom p.ip_typename_ref || is_std_remapped_type_ref p.ip_typename_ref then
          str "# " ++ Id.print p.ip_typename ++
          str ": remapped to Python primitive" ++ fnl ()
        else
          let tname = capitalize_first (pp_global state Term p.ip_typename_ref) in
          let n = Array.length p.ip_types in
          (* Shared base class.  For parameterised inductives it inherits
             [Generic[_A, …]] so that type-checkers can track type arguments. *)
          let pp_base =
            if List.is_empty tvars then
              str "class " ++ str tname ++ str ":" ++ fnl () ++
              str "    pass" ++ fnl ()
            else
              str "class " ++ str tname ++
              str "(Generic[" ++
              prlist_with_sep (fun () -> str ", ") str tvars ++
              str "]):" ++ fnl () ++
              str "    pass" ++ fnl ()
          in
          (* Constructor dataclasses, each inheriting from the base class. *)
          let pp_cons =
            prlist_with_sep (fun () -> fnl () ++ fnl ())
              (fun j ->
                 let base =
                   if List.is_empty tvars then tname
                   else
                     tname ^ "[" ^ String.concat ", " tvars ^ "]"
                 in
                 pp_one_cons state p None (Some base) j)
              (List.init n (fun j -> j))
          in
          (* Union alias for type-checker use: [TypeNameT = Con1 | Con2 | …].
             PEP 604 union syntax (Python 3.10+, fine — we target 3.14t). *)
          let cons_names =
            List.init n (fun j -> pp_global state Cons p.ip_consnames_ref.(j))
          in
          let pp_union =
            if List.is_empty tvars then
              let union_text = String.concat " | " cons_names in
              let alias = tname ^ "T = " ^ union_text in
              if String.length alias <= 88 then
                str alias ++ fnl ()
              else
                str (tname ^ "T = (") ++ fnl () ++
                str ("    " ^ union_text) ++ fnl () ++
                str ")" ++ fnl ()
            else
              mt ()
          in
          pp_base ++ fnl () ++ fnl () ++
          pp_cons ++ fnl () ++
          (if List.is_empty tvars then fnl () ++ pp_union else mt ())
      in
      pp_typevars ++
      prlist_with_sep (fun () -> fnl () ++ fnl ())
        pp_packet
        (Array.to_list ind.ind_packets)
  | Coinductive ->
      let pp_packet p =
        if p.ip_logical then
          str "# " ++ Id.print p.ip_typename ++ str ": logical coinductive" ++ fnl ()
        else if is_custom p.ip_typename_ref || is_std_remapped_type_ref p.ip_typename_ref then
          str "# " ++ Id.print p.ip_typename ++
          str ": remapped to Python primitive" ++ fnl ()
        else
          let tname = packet_name state p in
          let local_tvar_name i =
            Printf.sprintf "_%sT%d" tname i
          in
          let local_tvars = List.init ind.ind_nparams local_tvar_name in
          let pp_typevars =
            if List.is_empty local_tvars then mt ()
            else
              prlist_with_sep mt
                (fun tv ->
                   str tv ++ str " = TypeVar(\"" ++ str tv ++ str "\")" ++ fnl ())
                local_tvars ++ fnl ()
          in
          let ctor_base_opt =
            if List.is_empty local_tvars then None
            else
              Some ("Generic[" ^ String.concat ", " local_tvars ^ "]")
          in
          let n = Array.length p.ip_types in
          let cons_names =
            List.init n (fun j -> pp_global state Cons p.ip_consnames_ref.(j))
          in
          let step_members =
            if List.is_empty local_tvars then
              cons_names
            else
              List.map
                (fun name ->
                   name ^ "[" ^ String.concat ", " local_tvars ^ "]")
                cons_names
          in
          let step_type_expr = String.concat " | " step_members in
          let pp_cons =
            prlist_with_sep (fun () -> fnl () ++ fnl ())
              (fun j ->
                 pp_one_cons state ~typevar_shift:2 ~pp_tvar:local_tvar_name
                   p None ctor_base_opt j)
              (List.init n (fun j -> j))
          in
          pp_typevars ++
          pp_cons ++ fnl () ++ fnl () ++
          pp_coinductive_wrapper state p step_type_expr local_tvar_name local_tvars
      in
      prlist_with_sep (fun () -> fnl () ++ fnl ())
        pp_packet
        (Array.to_list ind.ind_packets)

(*s Declaration printer. *)

let pp_decl state = function
  | Dind  ind   -> pp_ind_decl state ind
  | Dtype (r, _, _) when is_std_real_type_ref r ->
      extraction_diagnostic_error "PYEX041"
  | Dtype (r, _, _) when is_custom r || is_std_remapped_type_ref r -> mt ()
  | Dtype _     -> fnl () ++ fnl () ++ diagnostic_comment "PYEX003"
  | Dterm (r, a, typ) ->
      if is_prop_type typ then mt ()
      else if is_runtime_marker_ref r then mt ()
      else if is_inline_custom r then mt ()
      else if is_std_primitive_compare_ref r then mt ()
      else if is_std_collection_term_ref r then mt ()
      else if is_custom r then
        str (pp_global state Term r) ++ str " = " ++
        str (find_custom r) ++ fnl ()
      else
        let () = validate_prop_discipline_decl (Dterm (r, a, typ)) in
        let env = empty_env state () in
        fnl () ++ fnl () ++ pp_term_decl state env (pp_global state Term r) a typ
  | Dfix (rv, defs, typs) ->
      (* Each function in the fix block is named globally; the bodies use
         [MLglob] references for mutual recursion, so [empty_env] suffices. *)
      let env = empty_env state () in
      let pp_one i =
        if is_prop_type typs.(i) then mt ()
        else if is_runtime_marker_ref rv.(i) then mt ()
        else if is_inline_custom rv.(i) then mt ()
        else if is_std_primitive_compare_ref rv.(i) then mt ()
        else if is_std_collection_term_ref rv.(i) then mt ()
        else if is_custom rv.(i) then
          str (pp_global state Term rv.(i)) ++ str " = " ++
          str (find_custom rv.(i)) ++ fnl ()
        else
          let () =
            validate_prop_discipline_decl
              (Dterm (rv.(i), defs.(i), typs.(i)))
          in
          fnl () ++ fnl () ++
          pp_term_decl state env (pp_global state Term rv.(i)) defs.(i) typs.(i)
      in
      prlist_with_sep mt pp_one (List.init (Array.length rv) (fun i -> i))

(*s Module structure walker.
    [State.with_visibility] must be called around each module so that
    [pp_global] can resolve names relative to the current module path. *)

let module_binding_name state l =
  pp_module state (MPdot (State.get_top_visible_mp state, l))

let rec module_type_annotation state = function
  | MTident mp ->
      str (pp_module state mp)
  | MTsig _ ->
      str "object"
  | MTwith (mt, _) ->
      module_type_annotation state mt
  | MTfunsig (_mbid, arg_mt, ret_mt) ->
      str "Callable[[" ++
      module_type_annotation state arg_mt ++
      str "], " ++
      module_type_annotation state ret_mt ++
      str "]"

let decl_export_names state = function
  | Dterm (r, _, typ) ->
      if is_prop_type typ || is_inline_custom r || is_runtime_marker_ref r ||
         is_std_collection_term_ref r then []
      else [pp_global state Term r]
  | Dfix (rv, _, typs) ->
      List.init (Array.length rv) Fun.id
      |> List.filter (fun i ->
           not (is_prop_type typs.(i)) &&
           not (is_runtime_marker_ref rv.(i)) &&
           not (is_inline_custom rv.(i)) &&
           not (is_std_collection_term_ref rv.(i)))
      |> List.map (fun i -> pp_global state Term rv.(i))
  | Dtype (r, _, _) ->
      if is_custom r || is_std_remapped_type_ref r then [] else [pp_global state Type r]
  | Dind ind ->
      let packet_names packet =
        if is_std_remapped_type_ref packet.ip_typename_ref then []
        else
          let tname = capitalize_first (pp_global state Term packet.ip_typename_ref) in
          let cnames =
            Array.to_list packet.ip_consnames_ref
            |> List.map (pp_global state Cons)
          in
          tname :: cnames
      in
      Array.to_list ind.ind_packets |> List.concat_map packet_names

let pp_decl_exports target names indent =
  prlist
    (fun name ->
       str (indent_string indent) ++
       str target ++ str "." ++ str name ++ str " = " ++ str name ++ fnl ())
    names

let pp_module_sig_member state = function
  | (l, Spec (Sval (r, typ))) ->
      str "    @property" ++ fnl () ++
      str "    def " ++ str (pp_global state Term r) ++ str "(self) -> " ++
      pp_type state typ ++ str ": ..." ++ fnl ()
  | (l, Spec (Stype (r, _, _))) ->
      str "    @property" ++ fnl () ++
      str "    def " ++ str (pp_global state Type r) ++ str "(self) -> object: ..." ++ fnl ()
  | (l, Spec (Sind ind)) ->
      let packet =
        if Array.length ind.ind_packets = 0 then []
        else
          Array.to_list ind.ind_packets
          |> List.map (fun p ->
               "    @property\n    def " ^
               capitalize_first (pp_global state Term p.ip_typename_ref) ^
               "(self) -> object: ...\n")
      in
      str (String.concat "" packet)
  | (l, Smodule mt) ->
      let annot =
        match mt with
        | MTfunsig (_, arg_mt, _) ->
            "Callable[[" ^ Pp.string_of_ppcmds (module_type_annotation state arg_mt) ^
            "], " ^ module_binding_name state l ^ "_Result]"
        | _ ->
            module_binding_name state l ^ "_Module"
      in
      str "    @property" ++ fnl () ++
      str "    def " ++ str (module_binding_name state l) ++ str "(self) -> " ++
      str annot ++ str ": ..." ++ fnl ()
  | (l, Smodtype _mt) ->
      str "    @property" ++ fnl () ++
      str "    def " ++ str (module_binding_name state l) ++ str "(self) -> object: ..." ++ fnl ()

let pp_protocol_decl_from_sig state name msig =
  str "class " ++ str name ++ str "(Protocol):" ++ fnl () ++
  (if List.is_empty msig then
     str "    pass" ++ fnl ()
   else
     prlist (pp_module_sig_member state) msig) ++
  fnl () ++ fnl ()

let rec pp_named_module_type_decl state name = function
  | MTsig (mp, msig) ->
      State.with_visibility state mp [] (fun state ->
        pp_protocol_decl_from_sig state name msig)
  | MTident mp ->
      str name ++ str " = " ++ str (pp_module state mp) ++ fnl () ++ fnl ()
  | MTwith (mt, _) ->
      pp_named_module_type_decl state name mt
  | MTfunsig (_mbid, arg_mt, ret_mt) ->
      str "class " ++ str name ++ str "(Protocol):" ++ fnl () ++
      str "    def __call__(self, arg0: " ++ module_type_annotation state arg_mt ++
      str ", /) -> " ++ module_type_annotation state ret_mt ++ str ": ..." ++ fnl () ++ fnl ()

let rec ensure_module_type_name state fallback = function
  | MTident mp ->
      mt (), pp_module state mp
  | MTsig (mp, msig) ->
      State.with_visibility state mp [] (fun state ->
        pp_protocol_decl_from_sig state fallback msig), fallback
  | MTwith (mt, _) ->
      ensure_module_type_name state fallback mt
  | MTfunsig (_mbid, arg_mt, ret_mt) ->
      let pp =
        str "class " ++ str fallback ++ str "(Protocol):" ++ fnl () ++
        str "    def __call__(self, arg0: " ++ module_type_annotation state arg_mt ++
        str ", /) -> " ++ module_type_annotation state ret_mt ++ str ": ..." ++ fnl () ++ fnl ()
      in
      pp, fallback

let rec pp_module_expr_value state indent local_name = function
  | MEident mp ->
      str (indent_string indent) ++ str local_name ++ str " = " ++ str (pp_module state mp) ++ fnl ()
  | MEstruct (mp, sel) ->
      str (indent_string indent) ++ str local_name ++ str ": Any = __ModuleNamespace()" ++ fnl () ++
      State.with_visibility state mp [] (fun state ->
        prlist (pp_module_structure_elem_into state (indent + 0) local_name) sel)
  | MEapply (me, me_arg) ->
      let pp_fun, fun_name = pp_module_expr_callable state indent (local_name ^ "_fun") me in
      let pp_arg, arg_name = pp_module_expr_arg state indent (local_name ^ "_arg") me_arg in
      pp_fun ++ pp_arg ++
      str (indent_string indent) ++ str local_name ++ str " = " ++ str fun_name ++ str "(" ++ str arg_name ++ str ")" ++ fnl ()
  | MEfunctor (mbid, mt, me) ->
      let param_name = pp_module state (MPbound mbid) in
      let pp_arg_mt, arg_annot = ensure_module_type_name state (local_name ^ "_Arg") mt in
      let ret_name = local_name ^ "_Result" in
      let ret_pp, ret_annot =
        ensure_module_type_name state ret_name (mtyp_of_mexpr me)
      in
      let nested_state = State.with_visibility state (State.get_top_visible_mp state) [mbid] Fun.id in
      pp_arg_mt ++ ret_pp ++
      str (indent_string indent) ++ str "def " ++ str local_name ++ str "(" ++
      str param_name ++ str ": " ++ str arg_annot ++ str ") -> " ++ str ret_annot ++ str ":" ++ fnl () ++
      pp_module_expr_return nested_state (indent + 4) "__result" me ++
      str (indent_string (indent + 4)) ++ str "return cast(" ++ str ret_annot ++ str ", __result)" ++ fnl ()

and pp_module_expr_callable state indent local_name = function
  | MEident mp ->
      mt (), pp_module state mp
  | MEfunctor _ as me ->
      pp_module_expr_value state indent local_name me, local_name
  | MEapply _ | MEstruct _ as me ->
      pp_module_expr_value state indent local_name me, local_name

and pp_module_expr_arg state indent local_name = function
  | MEident mp ->
      mt (), pp_module state mp
  | me ->
      pp_module_expr_value state indent local_name me, local_name

and pp_module_expr_return state indent local_name = function
  | MEstruct (mp, sel) ->
      str (indent_string indent) ++ str local_name ++ str ": Any = __ModuleNamespace()" ++ fnl () ++
      State.with_visibility state mp [] (fun state ->
        prlist (pp_module_structure_elem_into state indent local_name) sel)
  | me ->
      pp_module_expr_value state indent local_name me

and pp_module_structure_elem_into state indent target = function
  | (_, SEdecl d) ->
      indent_pp indent (pp_decl state d) ++
      pp_decl_exports target (decl_export_names state d) indent
  | (l, SEmodule m) ->
      let name = module_binding_name state l in
      if is_std_collection_module_name name then mt ()
      else
        pp_named_module_binding state indent name m ++
        str (indent_string indent) ++ str target ++ str "." ++ str name ++ str " = " ++ str name ++ fnl ()
  | (l, SEmodtype mt) ->
      let name = module_binding_name state l in
      indent_pp indent (pp_named_module_type_decl state name mt) ++
      str (indent_string indent) ++ str target ++ str "." ++ str name ++ str " = " ++ str name ++ fnl ()

and pp_named_module_binding state indent name m =
  match m.ml_mod_expr, m.ml_mod_type with
  | MEfunctor (mbid, arg_mt, body), MTfunsig (_, _, ret_mt) ->
      let param_name = pp_module state (MPbound mbid) in
      let arg_pp, arg_annot = ensure_module_type_name state (name ^ "_Arg") arg_mt in
      let ret_pp, ret_annot = ensure_module_type_name state (name ^ "_Result") ret_mt in
      indent_pp indent arg_pp ++
      indent_pp indent ret_pp ++
      str (indent_string indent) ++ str "def __" ++ str name ++ str "_impl(" ++
      str param_name ++ str ": " ++ str arg_annot ++ str ") -> " ++ str ret_annot ++ str ":" ++ fnl () ++
      pp_module_expr_return
        (State.with_visibility state (State.get_top_visible_mp state) [mbid] Fun.id)
        (indent + 4) "__result" body ++
      str (indent_string (indent + 4)) ++ str "return cast(" ++ str ret_annot ++ str ", __result)" ++ fnl () ++
      str (indent_string indent) ++ str "__" ++ str name ++ str "_cache: dict[int, " ++ str ret_annot ++ str "] = {}" ++ fnl () ++
      str (indent_string indent) ++ str "def " ++ str name ++ str "(" ++
      str param_name ++ str ": " ++ str arg_annot ++ str ") -> " ++ str ret_annot ++ str ":" ++ fnl () ++
      str (indent_string (indent + 4)) ++ str "return __apply_applicative(__" ++
      str name ++ str "_cache, __" ++ str name ++ str "_impl, " ++ str param_name ++ str ")" ++ fnl ()
  | _, _ ->
      let type_pp, type_name =
        ensure_module_type_name state (name ^ "_Module") m.ml_mod_type
      in
      let temp_name = "__" ^ name ^ "_value" in
      indent_pp indent type_pp ++
      pp_module_expr_value state indent temp_name m.ml_mod_expr ++
      str (indent_string indent) ++ str name ++ str ": " ++ str type_name ++
      str " = cast(" ++ str type_name ++ str ", " ++ str temp_name ++ str ")" ++ fnl ()

let pp_structure_elem state = function
  | (_, SEdecl d) ->
      pp_decl state d
  | (l, SEmodule m) ->
      let name = module_binding_name state l in
      if is_std_collection_module_name name then mt ()
      else pp_named_module_binding state 0 name m ++ fnl ()
  | (l, SEmodtype mt) ->
      let name = module_binding_name state l in
      pp_named_module_type_decl state name mt

let pp_structure_sel state sel =
  let methods_by_class = Hashtbl.create 8 in
  let class_names =
    List.filter_map
      (function
        | (_, SEdecl d) -> record_class_name_of_decl state d
        | (_, SEmodule _) | (_, SEmodtype _) -> None)
      sel
  in
  let class_prefixes =
    List.filter_map
      (function
        | (_, SEdecl d) -> record_class_prefix_of_decl state d
        | (_, SEmodule _) | (_, SEmodtype _) -> None)
      sel
  in
  let record_field_names =
    List.concat
      (List.filter_map
         (function
           | (_, SEdecl d) -> record_field_names_of_decl state d
           | (_, SEmodule _) | (_, SEmodtype _) -> None)
         sel)
  in
  List.iter
      (function
      | (_, SEdecl (Dterm (r, _a, _typ) as d)) -> (
          match method_target_of_term_decl state class_prefixes record_field_names d with
          | Some (class_name, method_name) when List.mem class_name class_names ->
              let entries =
                match Hashtbl.find_opt methods_by_class class_name with
                | Some entries -> entries
                | None -> []
              in
              Hashtbl.replace methods_by_class class_name ((method_name, d) :: entries)
          | _ -> ())
      | (_, SEdecl _d) -> ()
      | (_, SEmodule _) | (_, SEmodtype _) -> ())
    sel;
  let method_targets =
    Hashtbl.to_seq methods_by_class
    |> List.of_seq
    |> List.concat_map
         (fun (class_name, methods) ->
            List.map
              (fun (method_name, decl) ->
                 match decl with
                 | Dterm (r, _a, _typ) ->
                     (pp_global state Term r, (class_name, method_name))
                 | _ -> extraction_diagnostic_error "PYEX040")
              methods)
  in
  let prior_method_targets = !active_method_targets in
  active_method_targets := method_targets;
  let rendered =
  let pp_record_methods class_name =
    match Hashtbl.find_opt methods_by_class class_name with
    | None -> mt ()
    | Some methods ->
        fnl () ++
        prlist_with_sep
          (fun () -> fnl ())
          (fun (method_name, decl) ->
             match decl with
             | Dterm (_r, a, typ) ->
                 indent_pp 4 (pp_method_term_decl state (empty_env state ()) method_name a typ)
             | _ -> mt ())
          (List.rev methods)
  in
  prlist
    (function
      | (_, SEdecl d) -> (
          match method_target_of_term_decl state class_prefixes record_field_names d with
          | Some (class_name, _method_name) when List.mem class_name class_names ->
              mt ()
          | _ ->
              pp_decl state d ++
              (match record_class_name_of_decl state d with
               | Some class_name -> pp_record_methods class_name
               | None -> mt ()))
      | (l, SEmodule m) ->
          let name = module_binding_name state l in
          if is_std_collection_module_name name then mt ()
          else pp_named_module_binding state 0 name m ++ fnl ()
      | (l, SEmodtype mt) ->
          let name = module_binding_name state l in
          pp_named_module_type_decl state name mt)
    sel
  in
  active_method_targets := prior_method_targets;
  rendered

let pp_struct state struc =
  let pp_mod (mp, sel) =
    State.with_visibility state mp [] (fun state ->
      pp_structure_sel state sel)
  in
  prlist pp_mod struc

(*s No interface file for Python. *)

let sig_preamble _state _name _comment _used_modules _safe = mt ()

let pp_sig _state _sig = mt ()

(*s The language descriptor record registered with the extraction framework. *)

let python_descr : Common.State.t language_descr = {
  keywords;
  file_suffix    = ".py";
  file_naming;
  preamble;
  pp_struct;
  sig_suffix     = None;
  sig_preamble;
  pp_sig;
  pp_decl;
}
