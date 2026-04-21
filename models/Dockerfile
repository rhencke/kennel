# syntax=docker/dockerfile:1

ARG ROCQ_IMAGE=rocq-python-extraction:ci
FROM ${ROCQ_IMAGE} AS extract

WORKDIR /workspace
COPY --chown=opam:opam . .
COPY --from=rocq_models_cache --chown=opam:opam /_build/. /workspace/_build/

RUN targets="" \
    && for model in models/*.v; do targets="$targets ${model%.v}.vo"; done \
    && opam exec -- dune build $targets \
    && mkdir -p /tmp/model-output/kennel/models_generated \
    && py_files="$(find /workspace/_build/default -maxdepth 1 -name '*.py' -print)" \
    && if [ -z "$py_files" ]; then find /workspace/_build/default -maxdepth 2 -type f -print; exit 1; fi \
    && cp $py_files /tmp/model-output/kennel/models_generated/ \
    && printf '%s\n' \
        '"""Python extracted from Rocq models in models/.' \
        '' \
        'Rebuild with:  ./build' \
        'Source:        models/*.v (Rocq)' \
        'Generator:     rocq-python-extraction plugin' \
        '"""' \
        > /tmp/model-output/kennel/models_generated/__init__.py

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS format

WORKDIR /workspace
COPY pyproject.toml uv.lock ./
COPY --from=extract /tmp/model-output/kennel/models_generated /workspace/kennel/models_generated

RUN uv run --no-project --with ruff==0.15.9 ruff format /workspace/kennel/models_generated \
    && cd /workspace/kennel/models_generated \
    && find . -type f ! -name ".build-files" -printf "%P\n" | LC_ALL=C sort > .build-files

FROM scratch AS export

COPY --from=format /workspace/kennel/models_generated /kennel/models_generated
COPY --from=extract /workspace/_build /.cache/rocq-models/context/_build
