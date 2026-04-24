# Rocq → Python Extraction: Phase 0 Design

**Status:** Draft — MVP Phase 0 (closes #711)
**Parent:** #710

This document catalogues every node in Rocq's MiniML intermediate representation
and specifies the target Python construct for each.  It is the design contract for
Phase 1 (the `python.ml` extraction plugin) and gates all later phases.

Sources read:

- `plugins/extraction/miniml.mli` — IR type definitions
- `plugins/extraction/ocaml.ml` — OCaml backend (primary reference)
- `plugins/extraction/haskell.ml` — Haskell backend
- `plugins/extraction/scheme.ml` — Scheme backend
- `plugins/extraction/extract_env.ml` — module-level emission

---

## 0. Python Target

**The sole target is Python 3.14t** (the free-threaded build, no GIL).

All generated code is written for 3.14t and makes no attempt at backwards
compatibility with older Python versions.  Specifically:

- `match`/`case` structural pattern matching (3.10+) is used freely.
- PEP 604 union syntax (`A | B`) is used in type annotations and union
  aliases (3.10+).
- `from typing import Generic, TypeVar` is used for parameterised base
  classes; `typing.Callable` is used for arrow-type annotations.
- Annotation evaluation follows 3.14 semantics: annotations are **not**
  evaluated eagerly by default (PEP 649 / deferred evaluation), so forward
  references in dataclass field annotations work without quoting or
  `from __future__ import annotations`.
- The free-threaded runtime (`-Xgil=0`) is assumed; the generated code does
  not contain GIL-dependent concurrency assumptions.

The Dockerfile and CI both pin to a compatible Rocq + Python build.

---

## 1. Background: MiniML

Rocq's extraction pipeline first elaborates Gallina terms into **MiniML**, a
typed lambda-calculus that is simpler than Gallina but richer than any single
target language.  Each backend (`ocaml.ml`, `haskell.ml`, …) is an OCaml
pretty-printer from MiniML to its target syntax.  Our Python backend will be
another such pretty-printer, implemented as `python.ml` and registered with
`Extraction Language Python`.

MiniML has four primary namespaces:

| Namespace | OCaml type | Role |
|-----------|------------|------|
| Terms | `ml_ast` | Computations |
| Types | `ml_type` | Type annotations (advisory in a dynamic language) |
| Patterns | `ml_pattern` | Match arms |
| Declarations | `ml_decl` | Top-level definitions |

---

## 2. Term Nodes (`ml_ast`)

### 2.1 `MLrel n`

**What it is:** A de Bruijn variable reference.  `n = 1` is the innermost
binder; `n = 2` is the next one out.

**Python target:** Emit the identifier from the binder environment at position
`n`.  The backend maintains an env stack; `get_db_name n env` returns the
name.  No special syntax needed — just the bare identifier.

```python
# MLrel 1 in scope [x, y]  →
x
```

---

### 2.2 `MLapp (f, args)`

**What it is:** Function application.  `args` is a non-empty list; `f` may
itself be another `MLapp` (curried calls are nested left-associatively).

**Python target:** Flatten the curried application into a single call.  The
backend accumulates the argument list while recursing on `f` until `f` is not
an `MLapp`, then emits `f(a, b, c)`.

```python
# MLapp(MLapp(f, [a]), [b])  →
f(a, b)
```

---

### 2.3 `MLlam (id, body)`

**What it is:** Lambda abstraction.  Consecutive `MLlam` nodes appear as
nested, not flat.

**Python target:** Collect all leading `MLlam` nodes (via `collect_lams`),
then emit a single `lambda` expression with the full parameter list.

```python
# MLlam(x, MLlam(y, body))  →
lambda x, y: body
```

When a lambda appears at statement level (inside a `Dterm` or `MLletin`),
prefer a `def` statement over a `lambda` expression to support multi-statement
bodies and recursive references.

```python
# Dterm(f, MLlam(x, body), _)  →
def f(x):
    return body
```

---

### 2.4 `MLletin (id, a1, a2)`

**What it is:** Non-recursive let binding.  `a1` is the bound expression;
`a2` is the body.

**Python target:** An assignment followed by the body expression.  In a
statement context this is a plain assignment; in an expression context emit a
walrus operator (`:=`) or lift to a helper lambda.

Prefer the statement form for top-level emission:

```python
# MLletin(x, e1, e2)  →
x = e1
... e2 ...
```

---

### 2.5 `MLglob r`

**What it is:** A global reference — a constant, constructor, or module path.

**Python target:** The qualified Python name produced by the name-mangling
table (see §6).  No parentheses; just the identifier or dotted path.

```python
# MLglob Nat.succ  →
Nat.succ
```

---

### 2.6 `MLcons (ty, r, args)`

**What it is:** Constructor application, carrying the type `ty` (for dispatch
on inductive kind), the constructor reference `r`, and a list of arguments.

The `inductive_kind` of `ty` determines the emission form:

| `inductive_kind` | Python target |
|-----------------|---------------|
| `Standard` | dataclass instantiation: `ConstrName(a, b)` |
| `Singleton` | emit the single argument directly: `a` (newtype erasure) |
| `Record fields` | dataclass instantiation with keyword args: `T(field1=a, field2=b)` |
| `Coinductive` | thunk: `lambda: ConstrName(a, b)` (lazy by convention) |

```python
# MLcons(Standard, Nat.S, [n])  →
Nat.S(n)

# MLcons(Singleton, Box, [x])  →
x

# MLcons(Record [field_a; field_b], Point, [a; b])  →
Point(field_a=a, field_b=b)

# MLcons(Coinductive, Stream.Cons, [h; t])  →
lambda: Stream.Cons(h, t)
```

---

### 2.7 `MLtuple l`

**What it is:** A tuple literal of two or more elements.  (Scheme does not
support this node; OCaml and Haskell do.)

**Python target:** A Python tuple literal.

```python
# MLtuple [a; b; c]  →
(a, b, c)
```

---

### 2.8 `MLcase (ty, scrutinee, branches)`

**What it is:** Pattern match.  `branches` is an array of `ml_branch`
records, each of the form `(binder_idents, pattern, body)`.

The `inductive_kind` of `ty` selects an optimised emission form:

| Condition | Python target |
|-----------|---------------|
| Two branches, `Pcons(True,[])` / `Pcons(False,[])` | `body_t if scrutinee else body_f` |
| Singleton inductive | bind the single field directly, no match |
| Record inductive, single branch | `field = scrutinee.field_name` projections |
| General case | `match scrutinee:` with `case` arms (Python 3.10+) |

```python
# boolean case  →
body_t if scrutinee else body_f

# general match  →
match scrutinee:
    case ConstrName(x, y):
        body
    case _:
        wildcard_body
```

---

### 2.9 `MLfix (i, ids, defs)`

**What it is:** Mutual fixpoint.  `ids` is an array of function names; `defs`
is a corresponding array of bodies (each a `MLlam`).  `i` is the index of the
function to "return" — the one that is called at this point in the program.

**Python target:** A block of `def` statements (mutually recursive via Python
closure rules) followed by returning/calling `ids[i]`.

```python
# MLfix(0, [f; g], [def_f; def_g])  →
def f(...):
    return def_f_body

def g(...):
    return def_g_body

# then expression context uses f
```

For `Dfix` (top-level mutual recursion) the `def` statements are emitted at
module scope in dependency order with no enclosing scope.

---

### 2.10 `MLexn s`

**What it is:** An unreachable branch or internal error (corresponds to Rocq's
`assert False` in proofs, or an absurd pattern).

**Python target:** `raise RuntimeError(s)`.  Never `assert False` — Python
assertions can be suppressed with `-O`.

```python
# MLexn "Non-exhaustive pattern"  →
raise RuntimeError("Non-exhaustive pattern")
```

---

### 2.11 `MLdummy k`

**What it is:** An erased term — a placeholder for a logical argument (a
`Prop`-sorted term, an implicit type argument, or similar).  The `kill_reason`
tag explains why it was erased.

**Python target:** The sentinel `__`.  Define once at module top:

```python
__ = None  # erased logical argument
```

Emit `__` at every use site.  For erased binders in lambdas/branches, use `_`
(the Python throwaway).

---

### 2.12 `MLaxiom s`

**What it is:** A Rocq axiom that has no computational realization.  The
string `s` is the axiom's name.

**Python target:** A function that raises `NotImplementedError`, consistent
with OCaml's `failwith "AXIOM TO BE REALIZED"`.

```python
# MLaxiom "Funext"  →
raise NotImplementedError("AXIOM TO BE REALIZED: Funext")
```

---

### 2.13 `MLmagic a`

**What it is:** An identity coercion (`Obj.magic` in OCaml, `unsafeCoerce` in
Haskell).  Used when the type checker needs help but the value is unchanged.

**Python target:** Emit the inner expression `a` unchanged.  Python is
dynamically typed; no cast is needed.

```python
# MLmagic e  →
e
```

---

### 2.14 `MLuint i`

**What it is:** A 63-bit unsigned machine integer literal.

**Python target:** A plain Python `int` literal.  Python integers are
arbitrary-precision; no overflow risk.

```python
# MLuint 42  →
42
```

---

### 2.15 `MLfloat f`

**What it is:** A 64-bit IEEE-754 float literal.

**Python target:** A Python `float` literal.  Use `repr()` formatting to
preserve all significant bits.

```python
# MLfloat 3.14  →
3.14
```

---

### 2.16 `MLstring s`

**What it is:** A Rocq primitive byte-string literal.

**Python target:** A Python `bytes` literal.

```python
# MLstring "hello"  →
b"hello"
```

---

### 2.17 `MLparray (elems, default)`

**What it is:** A persistent array (Rocq's `PArray`) — a purely functional
array with efficient functional update.  Only OCaml supports this natively via
`ExtrNative.of_array`.

**Python target:** Phase 0 declares this **out of scope for MVP**.  Emit a
`NotImplementedError` at runtime:

```python
raise NotImplementedError("MLparray: persistent arrays not yet supported")
```

Band A will add a `PArray` class backed by a Python tuple (copy-on-write) once
the basic extraction path is stable.

---

## 3. Type Nodes (`ml_type`)

Python is dynamically typed.  Type nodes are used only to drive **emission
decisions** (e.g. choosing between singleton vs. standard constructor form)
and **`typing` annotations** in the generated stub.  They are never required
for correctness.

| Node | Emission decision | Optional annotation |
|------|------------------|---------------------|
| `Tarr(t1, t2)` | n/a | `Callable[[t1], t2]` |
| `Tglob(r, [])` | look up `r` in the inductive table | `TypeName` |
| `Tglob(r, args)` | same | `TypeName[arg1, arg2]` |
| `Tvar i` | n/a | `TypeVar` (named by convention) |
| `Tvar' i` | same as `Tvar` | same |
| `Tdummy _` | erased | omit annotation |
| `Tunknown` | n/a | `Any` |
| `Taxiom` | n/a | omit annotation |
| `Tmeta _` | internal only; never appears post-reconstruction | — |

MVP Phase 1–4 emits **no annotations** — pure untyped Python.  Phase 5 (repo
integration) adds `pyright` coverage, at which point annotations become
required for the public surface.

---

## 4. Pattern Nodes (`ml_pattern`)

Patterns appear in `MLcase` branches.  Python 3.10 structural pattern matching
(`match`/`case`) supports them directly.

| Node | Python `case` syntax |
|------|----------------------|
| `Pcons(r, [])` | `case ClassName():` |
| `Pcons(r, pats)` | `case ClassName(p1, p2):` |
| `Ptuple pats` | `case (p1, p2):` |
| `Prel n` | bind the name from binder env: `case ClassName(x):` |
| `Pwild` | `case _:` |
| `Pusual r` | shorthand for `Pcons(r, [Prel n; …; Prel 1])` — expand before emission |

`Pusual` is always expanded by the OCaml backend before reaching the
pretty-printer; our Python backend should do the same (call
`collect_pattern_vars` before `pp_pattern`).

---

## 5. Declaration Nodes (`ml_decl`)

### 5.1 `Dterm (r, ast, ty)`

A global constant or function definition.

```python
# non-function (MLlam not at top)  →
name = expr

# function (MLlam at top)  →
def name(x, y):
    return body
```

---

### 5.2 `Dtype (r, vars, ty)`

A type alias.  Emit as a Python `TypeAlias` (PEP 613).

```python
# Dtype(MyList, [a], Tglob(list, [Tvar a]))  →
from typing import TypeAlias, TypeVar
_a = TypeVar("_a")
MyList: TypeAlias = list[_a]
```

MVP may skip type alias emission and emit a comment instead.

---

### 5.3 `Dfix (refs, asts, tys)`

Mutual top-level fixpoint — same as `MLfix` but at module scope.  Emit as a
sequence of `def` statements.

```python
def f(n):
    ...g(n)...

def g(n):
    ...f(n)...
```

---

### 5.4 `Dind ml_ind`

An inductive type definition.  The `ml_ind` carries:

- `ind_kind` — how to emit
- `ind_nparams` — number of type parameters
- `ind_packets` — one per mutual inductive; each has a type name, constructor
  names, and constructor field types

Emission strategy by `inductive_kind`:

#### `Standard`

Emit a hierarchy of frozen `dataclass`es under a common base class:

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

_a = TypeVar("_a")

class Nat:
    pass

@dataclass(frozen=True)
class Nat_O(Nat):
    pass

@dataclass(frozen=True)
class Nat_S(Nat):
    arg0: Nat

# Optional union alias for type hints:
NatT = Nat_O | Nat_S
```

Constructor names are mangled: `{TypeName}_{ConstructorName}` to avoid
namespace collisions.  A module-level alias `O = Nat_O` is emitted for
ergonomics when names are unambiguous.

#### `Singleton`

A newtype wrapper — one constructor, one field.  **Erased entirely**: the
constructor and its single argument are inlined at use sites (handled by
`MLcons` Singleton case in §2.6).  No class is emitted.

#### `Record fields`

A `dataclass` where fields are named:

```python
@dataclass(frozen=True)
class Point:
    x: float
    y: float
```

Anonymous fields (`None` in `ip_sign`) get positional names `arg0`, `arg1`, ….

#### `Coinductive`

A lazy / stream type.  Emit the same `dataclass` hierarchy but wrap
constructor arguments in `Callable[[], T]` (thunks) to allow infinite
structures.

```python
@dataclass(frozen=True)
class Stream_Cons:
    head: int
    tail: "Callable[[], StreamT]"
```

---

## 6. Module Structure

MiniML modules map to Python modules (files) under a package.  The extraction
plugin controls the file layout via `ml_structure`.

| MiniML | Python |
|--------|--------|
| `MEstruct` (a flat module) | a `.py` file |
| `MEfunctor` | a factory function returning a module-object (Phase 1: raise `NotImplementedError`) |
| `MEident r` | `from path import *` alias |
| `MEapply` | call the factory function |

MVP (Phases 1–4) targets **flat extraction** — `Recursive Extraction` produces
a single `.py` file.  Multi-file module extraction is deferred to Phase 5.

---

## 7. Name Mangling

MiniML identifiers are Rocq fully-qualified names (`Coq.Init.Nat.S`).  Python
identifiers must be:

1. Valid Python identifiers (no `.`, no primes `'`, no unicode Rocq names).
2. Unique within a module.
3. Readable — preserve the base name where possible.

Rules (applied in order):

1. Strip the module path; keep the base name (`S`, `plus`, `eq_refl`).
2. Replace `'` with `_prime`.
3. Prepend `_` if the result is a Python keyword or built-in (`type`, `list`,
   `None`, `True`, `False`, `lambda`, …).
4. Append `_1`, `_2`, … to disambiguate clashes within the same module.
5. Constructor names within a type are prefixed with `TypeName_`:
   `Nat_O`, `Nat_S`, `List_nil`, `List_cons`.

The name-mangling table is populated in `table.ml` and consulted at every
`MLglob` and `Dterm`/`Dind` emission point.

---

## 8. The `__` Sentinel

Every Python extraction module defines:

```python
__ = None  # MiniML MLdummy: erased logical argument
```

At use sites where a lambda binder is erased, emit `_` (Python's throwaway).
At value use sites, emit `__`.  This mirrors the OCaml convention and lets
downstream code search for `__` to find erasure points.

---

## 9. Primitive Remapping (`Extract Inductive` / `Extract Constant`)

Rocq users can redirect specific types and constants to Python primitives via
extraction hints.  The `table.ml` module stores these.

The Python backend also owns a small set of Stdlib remappings that are always
available without local pragmas:

| Rocq type | Python target | Notes |
|-----------|---------------|-------|
| `Stdlib.Strings.String.string` | `str` | UTF-8 text boundary; `String` pattern splitting raises `_RocqUtf8BoundaryError` if the tail is invalid UTF-8. |
| `Stdlib.Strings.Ascii.ascii` | `int` | Byte value in `0..255`; `Ascii b0 ... b7` packs least-significant bit first. |
| `Stdlib.Init.Byte.byte` | `int` | Byte constructors `x00` through `xff` lower to integer literals. |
| primitive `%pstring` / `MLstring` | `bytes` | Byte strings are emitted as Python `bytes` literals. |
| `Stdlib.Init.Datatypes.nat` | `int` | Non-negative integer; pattern matching rejects negative Python inputs. |
| `Stdlib.Numbers.BinNums.positive` | `int` | Strictly positive integer; pattern matching rejects zero and negative Python inputs. |
| `Stdlib.Numbers.BinNums.N` | `int` | Non-negative binary natural; pattern matching rejects negative Python inputs. |
| `Stdlib.Numbers.BinNums.Z` | `int` | Arbitrary precision signed integer. |
| `Stdlib.QArith.QArith_base.Q` | `fractions.Fraction` | Rational values use Python's normalized numerator and denominator on destructuring. |
| `Stdlib.Reals.Rdefinitions.R` | unsupported | Raises `PYEX041`; no `float` mapping is provided for classical reals. |

Other remappings currently covered by acceptance tests:

| Rocq type | Python target |
|-----------|---------------|
| `Coq.Init.Datatypes.bool` | `bool` (`True`/`False`) |
| `Coq.Init.Datatypes.list` | `list` |
| `Coq.Init.Datatypes.option` | `T \| None` |
| `Coq.Init.Datatypes.prod` | `tuple[A, B]` |
| `Coq.Init.Datatypes.unit` | `None` |

These remain configured via `Extract Inductive` / `Extract Constant` pragmas
in `.v` files; `table.ml` handles those lookups.

---

## 10. Unsupported Nodes (Phase 0 Scope Boundaries)

The following nodes are **out of scope for MVP Phases 1–4** and will emit a
`NotImplementedError` placeholder.  They are tracked as future work in the
parent issue #710.

| Node / Feature | Planned band |
|----------------|-------------|
| `MLparray` | Band A (`MLparray` native array) |
| `MEfunctor` / `MEapply` (module functors) | Band A (A7) |
| `Coinductive` streams (full lazy semantics) | Band A (A6) |
| `Program Fixpoint` / `Acc`-recursion | Band A (A4) |
| Type annotations / `pyright` coverage | Phase 5 |
| Bidirectional source maps | Band B (B1) |
| `IO` / `State` monad extraction | Band A (A10) |

---

## 11. Acceptance Criteria

This document satisfies the Phase 0 acceptance criterion when:

- Every `ml_ast` variant has a named Python target (§2). ✓
- Every `ml_type` variant has a noted role (§3). ✓
- Every `ml_pattern` variant has a `case` syntax (§4). ✓
- Every `ml_decl` variant has an emission strategy (§5). ✓
- Module mapping is described (§6). ✓
- Name mangling rules are specified (§7). ✓
- Primitive remappings are listed (§9). ✓
- Out-of-scope nodes are enumerated with band assignments (§10). ✓

Phase 1 can begin: implement `python.ml` following this spec, using `ocaml.ml`
as the structural template.
