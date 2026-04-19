# rocq-python-extraction

Rocq → Python extraction plugin.  Registers a `Python Extraction` vernacular with
Rocq's extraction framework and emits Python 3.14t source from MiniML terms.

## Python target: 3.14t ONLY

**The sole supported Python version is Python 3.14t** (free-threaded build, no GIL).

This is a hard constraint.  Do not add compatibility shims, `from __future__`
imports, or conditional code for older Python versions.  If something requires
a workaround on Python ≤ 3.13, the answer is to use a different approach that
works cleanly on 3.14t — not to add a shim.

Specifically, the generated preamble must not contain `from __future__ import
annotations`.  Forward references in `@dataclass` field annotations are handled
natively by PEP 649 deferred annotation evaluation, which is the default on
3.14t.

The Docker CI image is pinned to Python 3.14t via uv.  Do not downgrade it or
add a fallback to a system Python.

## Testing

```bash
make test          # build + check extracted .py syntax + run round-trip assertions
make docker-test   # run the full suite inside the CI Docker image
```

`make docker-build` rebuilds the Docker image locally (needed after Dockerfile changes).

100% of round-trip assertions must pass.  Adding a new extraction function
requires a corresponding round-trip test in `dune` (the canonical home) — see
the existing `runtest` rules for the pattern.  The Makefile `test` target must
stay in sync but the assertions themselves live in `dune`.

## Building

```bash
dune build         # compile the plugin and the test theories
```

## Linting / formatting

Follow the surrounding OCaml style (no external linter is enforced; match the
style of `python.ml` and `g_python_extraction.mlg`).

## Key files

| File | Purpose |
|------|---------|
| `python.ml` | The extraction backend — MiniML → Python pretty-printer |
| `g_python_extraction.mlg` | Vernacular registration (`Python Extraction`) |
| `test/phase*.v` | Acceptance tests; each phase covers one IR feature |
| `Dockerfile` | CI image — OCaml + Rocq + Python 3.14t via uv |
| `DESIGN.md` | Full MiniML → Python mapping contract |
