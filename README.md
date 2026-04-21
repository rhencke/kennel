# kennel

GitHub webhook listener and Fido runner.

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
- The Dune `_build` cache is exported through buildx to
  `.cache/rocq-models/context/_build`.
- A manifest in `.cache/rocq-models/manifest.sha` gives unchanged local runs a
  fast pre-build exit.

Extra arguments are passed through to `docker buildx build` before the final
context argument:

```bash
./build --no-cache
```

The internal smart-output mode is available for buildx artifact producers:

```bash
./build --smart-output DIR -- <command> [args...]
```

If the command emits `DIR/.build-files`, the helper removes files in `DIR` that
are not listed. Without a manifest, it falls back to ctime cleanup for files
older than the command start marker.
