# rocq-python-extraction

Rocq → Python extraction plugin.  Registers a `Python Extraction` vernacular
with Rocq's extraction framework and emits Python 3.14t source from MiniML
terms.

The Python target is **Python 3.14t only** (free-threaded, no GIL) — same
constraint as the rest of the repo. No `from __future__ import annotations` in
the generated preamble; PEP 649 deferred annotation evaluation handles forward
references in `@dataclass` fields natively on 3.14t.

Python tooling runs through buildx uv targets. The Rocq Docker CI image stays
Rocq-only; generated Python is copied into uv stages for formatting, pyright,
and pytest.

## Testing

```bash
./fido make-rocq  # regenerate committed Rocq-extracted Python
./fido ci         # format, lint, typecheck, generated typecheck, tests
```

`./fido make-rocq` runs the `make-rocq` buildx bake target. BuildKit rebuilds
the Rocq image and dependent layers when their Dockerfile or inputs change.
100% of round-trip assertions must pass. Add new extraction checks as pytest
tests under `rocq-python-extraction/test/test_*.py`; `./fido ci` is the
canonical assertion path.

## Linting / formatting

Follow the surrounding OCaml style (no external linter is enforced; match the
style of `python.ml` and `g_python_extraction.mlg`).

## Key files

| File | Purpose |
|------|---------|
| `python.ml` | The extraction backend — MiniML → Python pretty-printer |
| `g_python_extraction.mlg` | Vernacular registration (`Python Extraction`) |
| `test/python.v` + `test/*.v` feature files | Acceptance tests; `python.v` is the umbrella entrypoint over feature-scoped theories |
| `Dockerfile` | CI image — OCaml + Rocq toolchain |
| `DESIGN.md` | Full MiniML → Python mapping contract |
| `DIAGNOSTICS.md` | Diagnostic and troubleshooting guide |
