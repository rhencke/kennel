# kennel

GitHub webhook listener and Fido runner.

## Fido launcher

Use the root launcher to run project commands inside the buildx uv image:

```bash
./fido help
./fido up
./fido up --detach
./fido down
./fido status
./fido tests
./fido pytest tests/test_build_wrapper.py -q
```

`./fido` builds the `fido` target from `models/Dockerfile` with `docker buildx
build`, loads it as `fido:local`, then runs it with Docker.
`./fido help` lists the project commands. `./fido up` runs the kennel server,
`./fido up --detach` runs it as a named Docker container, and `./fido down`
stops that detached container. Other friendly commands such as `./fido status`,
`./fido task`, and `./fido sync-tasks` map to dedicated `pyproject.toml`
scripts. Unknown commands are passed to `uv run` unchanged. The repository is
bind-mounted at `/workspace`, and the container runs with the caller's UID/GID
so files written through bind mounts keep the host user's ownership. `./fido
up` also bind-mounts the 0600 secret file read-only at
`/run/secrets/kennel-secret`, plus the host workspace and log directories. Runs
use host networking so `./fido up` exposes the webhook server normally and
`./fido status` can reach it from the same network namespace.

Set `FIDO_IMAGE=...` to override the local image tag, and
`FIDO_CONTAINER=...` to override the detached container name. Set
`FIDO_SECRET=...` to override the host secret file used by `./fido up`.

## Rocq model builds

Use the root build helper:

```bash
./build
```

`./build` is the canonical Rocq model generation entry point. It runs
`docker buildx build`, extracts `models/*.v`, formats the generated Python in
an Astral `uv` Python image, and writes the committed output to
`kennel/models_generated/`.

The helper keeps build work inside buildx:

- Rocq/Dune run in an internal `rocq-python-extraction:ci` image that `./build`
  builds with buildx from `rocq-python-extraction/Dockerfile`. Set
  `ROCQ_IMAGE=...` to test another image and skip the internal image build.
- Python formatting runs in `ghcr.io/astral-sh/uv:python3.14-bookworm-slim`.
- The Rocq image build cache is exported through buildx to
  `.cache/rocq-models/image`.
- The model Dockerfile build cache is exported through buildx to
  `.cache/rocq-models/buildx` when the active builder supports external cache
  export. This includes uv cache mounts used by the `ci` target.
- The Dune `_build` cache is exported through buildx to
  `.cache/rocq-models/context/_build`.
- A manifest in `.cache/rocq-models/manifest.sha` gives unchanged local runs a
  fast pre-build exit.

Extra arguments are passed through to `docker buildx build` before the final
context argument:

```bash
./build --no-cache
./build --target format
./build --target lint
./build --target typecheck
./build --target generated-typecheck
./build --target test
./build --target ci
```

All Python checks run inside buildx targets. Rocq test artifacts are produced
by a Rocq stage, then ruff format, ruff lint, pyright, generated pyright, and
pytest run as separate uv stages. The aggregate `ci` target is a scratch
meta-target that depends on marker files from those stages, so BuildKit can run
independent checks in parallel. The uv dependency layer follows Astral's Docker
pattern: `pyproject.toml`, `uv.lock`, and `.python-version` are bind-mounted
into a `uv sync --frozen --no-install-project` layer before the source tree is
copied. The pre-commit hook and CI both call `./build --target ci`.

The internal smart-output mode is available for buildx artifact producers:

```bash
./build --smart-output DIR -- <command> [args...]
```

If the command emits `DIR/.build-files`, the helper removes files in `DIR` that
are not listed. Without a manifest, it falls back to ctime cleanup for files
older than the command start marker.
