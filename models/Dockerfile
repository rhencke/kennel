# syntax=docker/dockerfile:1

ARG ROCQ_IMAGE=rocq-python-extraction:ci

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS python-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libatomic1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/tmp/kennel-venv
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

FROM python-base AS fido

WORKDIR /workspace

ENV HOME=/tmp
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_PROJECT_ENVIRONMENT=/opt/kennel-venv
ENV UV_PYTHON_CACHE_DIR=/tmp/uv-cache/python
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN --mount=type=cache,target=/tmp/uv-cache \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    mkdir -p /tmp/uv-cache \
    && uv sync --frozen --no-install-project \
    && chmod -R a+rwX /opt/kennel-venv /opt/uv-python /tmp/uv-cache

ENTRYPOINT ["uv", "run"]

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

FROM python-base AS model-format

WORKDIR /workspace
COPY .python-version pyproject.toml uv.lock ./
COPY --from=extract /tmp/model-output/kennel/models_generated /workspace/kennel/models_generated

RUN uv run --no-project --with ruff==0.15.9 ruff format /workspace/kennel/models_generated \
    && cd /workspace/kennel/models_generated \
    && find . -type f ! -name ".build-files" -printf "%P\n" | LC_ALL=C sort > .build-files

FROM scratch AS export

COPY --from=model-format /workspace/kennel/models_generated /kennel/models_generated
COPY --from=extract /workspace/_build /.cache/rocq-models/context/_build

FROM extract AS test-extract

RUN opam exec -- dune build rocq-python-extraction/test/python.vo \
    && mkdir -p /tmp/pytest-artifacts /tmp/pyright-artifacts \
    && while IFS= read -r name; do \
        cp "/workspace/_build/default/$name" "/tmp/pytest-artifacts/$name"; \
    done < rocq-python-extraction/test/generated_pytest_targets.txt \
    && while IFS= read -r name; do \
        cp "/workspace/_build/default/$name" "/tmp/pyright-artifacts/$name"; \
    done < rocq-python-extraction/test/generated_pyright_targets.txt

FROM python-base AS python-deps

WORKDIR /workspace

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project \
    && touch /tmp/python-deps-ready

FROM python-base AS python-check-base

WORKDIR /workspace
COPY --from=python-deps /tmp/python-deps-ready /tmp/python-deps-ready
COPY . .
COPY --from=model-format /workspace/kennel/models_generated /workspace/kennel/models_generated

ENV KENNEL_ROCQ_PYTEST_ARTIFACTS=prepared
ENV KENNEL_SKIP_DOCKER_HELPER_TESTS=1

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pyright-python \
    uv sync --frozen

FROM python-check-base AS format

RUN --mount=type=cache,target=/root/.cache/uv \
    uv run ruff format --check . \
    && touch /tmp/format-ready

FROM python-check-base AS lint

RUN --mount=type=cache,target=/root/.cache/uv \
    uv run ruff check . \
    && touch /tmp/lint-ready

FROM python-check-base AS typecheck

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pyright-python \
    uv run pyright \
    && touch /tmp/typecheck-ready

FROM python-check-base AS generated-typecheck

COPY --from=test-extract /tmp/pytest-artifacts /workspace/_build/default
COPY --from=test-extract /tmp/pyright-artifacts /tmp/generated-pyright

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pyright-python \
    ./rocq-python-extraction/run_generated_pyright.sh /tmp/generated-pyright \
    && touch /tmp/generated-typecheck-ready

FROM python-check-base AS test

COPY --from=test-extract /tmp/pytest-artifacts /workspace/_build/default

RUN --mount=type=cache,target=/root/.cache/uv \
    uv run tests \
    && touch /tmp/test-ready

FROM scratch AS ci

COPY --from=format /tmp/format-ready /format-ready
COPY --from=lint /tmp/lint-ready /lint-ready
COPY --from=typecheck /tmp/typecheck-ready /typecheck-ready
COPY --from=generated-typecheck /tmp/generated-typecheck-ready /generated-typecheck-ready
COPY --from=test /tmp/test-ready /test-ready
