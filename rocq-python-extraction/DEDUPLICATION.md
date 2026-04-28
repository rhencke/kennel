# Rocq -> Python Extraction Deduplication Targets

This catalogue tracks the cleanup opportunities behind
FidoCanCode/home#1087. It is intentionally an audit note, not a design lock:
each target should still earn its own focused tests before the extractor
behavior changes.

## Expression precedence and associativity

The largest duplication pressure is in expression rendering. `pp_expr` currently
emits Python syntax directly from many branches, and each branch decides for
itself when to wrap operands:

- boolean lowering has a local "atomic or parenthesized" helper;
- set union, intersection, and difference always add parentheses;
- lambda-as-call-head has a one-off parenthesized case;
- lambda-lifted lets, option binds, reader binds, and custom match expressions
  manually shape low-precedence expressions;
- primitive comparisons and list append emit infix forms without a shared
  rule for child precedence.

This is the combinatorial risk called out in #1087. The next abstraction should
let expression emitters return a rendered node with precedence and associativity
metadata, then centralize the decision to parenthesize a child in a parent
context. That gives `not (x and y)`, `(x | y) & z`, `f(lambda...)`, comparisons,
addition, and future infix lowerings one rule instead of one helper per special
case.

## Primitive and collection application lowerings

`MLapp` handling contains several small dispatchers that all recognize a global
head, check arity, and return specialized Python:

- bool operations: `negb`, `andb`, `eqb`;
- primitive comparisons over nat, positive, ascii, and string;
- list append;
- positive and string maps;
- positive and string sets;
- monad and concurrency marker calls;
- direct record-field and method call rewrites.

Those dispatchers duplicate the same shape: identify the lowering, validate the
argument list, then emit an expression. Once precedence-aware expression nodes
exist, these should become a table or small registry of primitive lowering
rules. The rule should carry the operation family, accepted arity, emitted
Python form, and declaration/export suppression behavior where applicable.

## Standard-library reference predicates

The extractor has many `is_std_*_ref` helpers for type references,
constructors, and term references. They are useful, but the term predicates now
duplicate operation families across application lowering, declaration skipping,
export filtering, and remapped module handling.

A good cleanup target is to separate "reference classification" from "what this
site does with the reference". For example, classify `Bool.andb` and `Pos.eqb`
once as primitive operations, then let the expression printer, declaration
printer, and export list ask whether that classified operation is inlined. That
keeps future lowerings from requiring the same `is_std_*` check in three or four
places.

## Declaration and export filtering

`pp_decl` and `decl_export_names` both decide whether a declaration should be
emitted. Their conditions overlap heavily: prop declarations, runtime markers,
inline custom definitions, primitive operations, collection operations, and
record accessors are all suppressed in more than one place. The Dfix branch
repeats most of the Dterm branch too.

This wants a single declaration classification helper with outcomes such as
emit term, emit custom alias, suppress inline primitive, suppress erased prop,
or emit diagnostic. The declaration printer and export collector can then share
that classification instead of re-encoding it independently.

## Record method and field target registries

Record method extraction and record-field projection lowering use parallel
module-scope registries:

- `active_method_targets` maps extracted functions to instance methods;
- `active_record_field_targets` maps record accessor functions to direct field
  reads.

Both registries are populated while rendering a structure, temporarily installed
as global mutable state, and queried during expression/declaration printing.
This is a useful abstraction boundary, but the shape is duplicated and the
temporary global mutation makes the dependency harder to see. A later cleanup
should gather these into one scoped lowering environment that is passed through
the printer state, with lookup helpers for methods, fields, and future
structure-local rewrites.

## Type variable emission

TypeVar naming is centralized, but TypeVar declaration emission still appears
in several places:

- shared inductive parameter declarations;
- record declarations;
- local term signature declarations;
- protocol variance declarations;
- coinductive packet-local declarations.

The next useful abstraction is a small TypeVar declaration emitter that accepts
the name source and variance, then handles spacing and deduplication uniformly.
That would keep generated headers stable as more protocol and generic
annotations are added.

## Suggested implementation order

1. Add focused extraction tests for nested boolean, comparison, list, set, and
   lambda application precedence before changing the printer.
2. Introduce rendered expression nodes with precedence and associativity.
3. Route bool, comparison, list, and set infix lowerings through those nodes.
4. Extract a primitive-operation classifier and share it between expression
   lowering, declaration suppression, and export filtering.
5. Replace the parallel active method/field globals with one scoped lowering
   environment.
6. Fold TypeVar declaration formatting into one helper after expression and
   declaration filtering are stable.

