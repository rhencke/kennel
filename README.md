# Fido

GitHub webhook listener and Fido runner.

## Fido launcher

Use the root launcher to run project commands inside the buildx uv image:

```bash
./fido help
./fido up
./fido down
./fido warm
./fido status
./fido tests  # focused pytest convenience; use ./fido warm before commits
./fido ruff format .
./fido pytest tests/test_build_wrapper.py -q
```

`./fido` builds the `fido` bake target from `docker-bake.hcl` with `docker
buildx bake`, loads it as `fido:local`, then runs it with Docker.
`./fido help` lists the project commands. `./fido up` runs the Fido server,
then supervises it in the foreground with `docker run --rm`. `./fido up`
appends supervisor and container stdout/stderr to `~/log/fido.log`; override
with `FIDO_LOG=...`. On update exits from the app, `./fido up` syncs the runner
clone, rebuilds the image, and starts again. It exits normally on ordinary
shutdown signals. `./fido down` stops the named container gracefully; `--rm`
lets Docker remove it after the stop.
`./fido warm` builds the `warm` bake group, which depends on full CI and the
production runtime image cache, so local and CI runs populate those cache
families through one command. The `fido-test` image is built on demand for ad hoc local
commands.

Project commands such as `./fido status`, `./fido task`, and `./fido
sync-tasks` map to dedicated `pyproject.toml` scripts in the production image.
Unknown commands, such as `./fido ruff format .`, run through the `fido-test` image and
are passed to containerized `uv run` unchanged; do not use host `uv` for normal
project checks. The production runtime image installs only production Python
dependencies, plus pinned Node CLI tools from
`package-lock.json` (`claude` and
`copilot`) and the GitHub CLI. Node dependencies are built in their own Docker
stage with an npm cache mount, so changing application code does not rerun
`npm ci`.

The repository is bind-mounted at `/workspace`, and the container runs with the
caller's UID/GID so files written through bind mounts keep the host user's
ownership. `./fido up` bind-mounts the 0600 secret file read-only at
`/run/secrets/fido-secret`. All runs also mount the host workspace and, when
present, `.claude`, `.claude.json`, `.config/gh`, and `.cache/copilot` at the
same absolute paths. Logs go to stdout/stderr for Docker or systemd to capture.
Runs use host networking so `./fido up` exposes the webhook server normally and
`./fido status` can reach it from the same network namespace.

The bake target hides the Dockerfile path and stage selection, so the launcher
only supplies runtime variables such as the image tag and host UID/GID. Set
`FIDO_IMAGE=...` to override the local image tag,
`FIDO_CONTAINER=...` to override the container name, and `FIDO_SECRET=...` to
override the host secret file used by `./fido up`. Use `FIDO_AUTO_UPDATE=0` or
`./fido --no-auto-update up ...` for dirty local testing; production startup
cleans the runner clone with `git reset --hard` and `git clean -fd -e .cache/`
before syncing `origin/main`.

## Rocq model builds

Use the Fido launcher:

```bash
./fido make-rocq
```

`./fido make-rocq` is the canonical Rocq model generation entry point. It runs
`docker buildx build`, extracts `models/*.v`, formats the generated Python in
an Astral `uv` Python image, and writes the committed output to
`src/fido/rocq/`.

The helper keeps build work inside buildx:

- Rocq/Dune run in an internal `rocq-python-extraction:ci` image that `./fido make-rocq`
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
./fido make-rocq --no-cache
./fido make-rocq --target format
./fido make-rocq --target lint
./fido make-rocq --target typecheck
./fido make-rocq --target generated-typecheck
./fido make-rocq --target test
```

All Python checks run inside buildx bake targets. Rocq test artifacts are
produced by a Rocq stage, then ruff format, ruff lint, pyright, generated
pyright, and pytest run as separate uv stages. The `ci` and `warm` bake groups
are meta groups over those real targets, so BuildKit can run independent checks
in parallel without scratch sentinel targets. The uv dependency layers follow
Astral's Docker pattern: `pyproject.toml`, `uv.lock`, and `.python-version` are
copied into `uv sync --frozen --no-install-project` layers before application
inputs are copied. The pre-commit hook and CI both call `./fido warm`.

The internal smart-output mode is available for buildx artifact producers:

```bash
./fido make-rocq --smart-output DIR -- <command> [args...]
```

If the command emits `DIR/.build-files`, the helper removes files in `DIR` that
are not listed. Without a manifest, it falls back to ctime cleanup for files
older than the command start marker.
