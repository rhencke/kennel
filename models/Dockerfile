# syntax=docker/dockerfile:1

ARG ROCQ_IMAGE=rocq-python-extraction:ci

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS python-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libatomic1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/tmp/fido-venv
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

FROM python-base AS fido-python-prod

WORKDIR /workspace

ENV HOME=/tmp
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_PROJECT_ENVIRONMENT=/opt/fido-venv
ENV UV_PYTHON_CACHE_DIR=/tmp/uv-cache/python
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

COPY .python-version pyproject.toml uv.lock ./

RUN --mount=type=cache,id=fido-uv-prod,target=/tmp/uv-cache \
    mkdir -p /tmp/uv-cache \
    && uv sync --frozen --no-dev --no-install-project \
    && chmod -R a+rwX /opt/fido-venv /opt/uv-python /tmp/uv-cache

FROM fido-python-prod AS fido-python-dev

RUN --mount=type=cache,id=fido-uv-dev,target=/tmp/uv-cache \
    uv sync --frozen --no-install-project \
    && chmod -R a+rwX /opt/fido-venv /opt/uv-python /tmp/uv-cache

FROM node:24-bookworm-slim AS node-runtime

FROM fido-python-prod AS fido-base

ARG FIDO_UID=1000
ARG FIDO_GID=1000
ARG FIDO_USER=fido
ARG FIDO_HOME=/home/fido
ARG GH_VERSION=2.89.0

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -fsSL -o /tmp/gh.deb "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.deb" \
    && apt-get install -y --no-install-recommends /tmp/gh.deb \
    && rm -f /tmp/gh.deb \
    && rm -rf /var/lib/apt/lists/*

FROM node-runtime AS node-tools

WORKDIR /workspace

COPY package.json package-lock.json ./

RUN --mount=type=cache,id=fido-npm,target=/root/.npm \
    npm ci --omit=dev --ignore-scripts \
    && npm rebuild @anthropic-ai/claude-code

FROM fido-base AS fido

ARG FIDO_UID=1000
ARG FIDO_GID=1000
ARG FIDO_USER=fido
ARG FIDO_HOME=/home/fido

COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-tools /workspace/node_modules /opt/fido-node-tools/node_modules

RUN groupadd --gid "$FIDO_GID" "$FIDO_USER" \
    && useradd --uid "$FIDO_UID" --gid "$FIDO_GID" --home-dir "$FIDO_HOME" --shell /bin/bash --create-home "$FIDO_USER" \
    && mkdir -p "$FIDO_HOME/.config" "$FIDO_HOME/.cache" "$FIDO_HOME/workspace" /tmp/uv-cache \
    && ln -sf /opt/fido-node-tools/node_modules/.bin/claude /usr/local/bin/claude \
    && ln -sf /opt/fido-node-tools/node_modules/.bin/copilot /usr/local/bin/copilot \
    && chown -R "$FIDO_UID:$FIDO_GID" "$FIDO_HOME" /tmp/uv-cache

ENV HOME=$FIDO_HOME

ENTRYPOINT ["uv", "run"]

FROM fido AS fido-test

ARG FIDO_UID=1000
ARG FIDO_GID=1000

COPY --from=fido-python-dev /opt/fido-venv /opt/fido-venv
COPY --from=fido-python-dev /opt/uv-python /opt/uv-python

FROM ${ROCQ_IMAGE} AS rocq-plugin-source

WORKDIR /workspace
USER root
RUN chown opam:opam /workspace
USER opam

COPY --chown=opam:opam dune-workspace ./
COPY --chown=opam:opam \
    rocq-python-extraction/dune-project \
    rocq-python-extraction/dune \
    rocq-python-extraction/rocq-python-extraction.opam \
    rocq-python-extraction/META.rocq-python-extraction.template \
    rocq-python-extraction/g_python_extraction.mlg \
    rocq-python-extraction/python.ml \
    rocq-python-extraction/

FROM ${ROCQ_IMAGE} AS rocq-toolchain

FROM python-base AS rocq-repl

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        gcc \
        libc6-dev \
        libgmp-dev \
        linux-libc-dev \
        make \
        pkg-config \
        tar \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash opam

COPY --from=rocq-toolchain --chown=opam:opam /home/opam/.opam/5.3 /home/opam/.opam/5.3
COPY --from=rocq-toolchain /usr/local/bin/opam /usr/local/bin/opam

ENV PATH="/home/opam/.opam/5.3/bin:${PATH}"
ENV OPAM_SWITCH_PREFIX="/home/opam/.opam/5.3"
ENV CAML_LD_LIBRARY_PATH="/home/opam/.opam/5.3/lib/stublibs"
ENV OCAML_TOPLEVEL_PATH="/home/opam/.opam/5.3/lib/toplevel"
ENV OCAMLTOP_INCLUDE_PATH="/home/opam/.opam/5.3/lib/toplevel"

FROM rocq-plugin-source AS extract

COPY --chown=opam:opam models/dune-project models/dune models/*.v models/
COPY --from=rocq_models_cache --chown=opam:opam /_build/. /workspace/_build/

RUN targets="" \
    && for model in models/*.v; do targets="$targets ${model%.v}.vo"; done \
    && opam exec -- dune build $targets \
    && mkdir -p /tmp/model-output/src/fido/rocq \
    && py_files="$(find /workspace/_build/default -maxdepth 1 -name '*.py' -print)" \
    && if [ -z "$py_files" ]; then find /workspace/_build/default -maxdepth 2 -type f -print; exit 1; fi \
    && cp $py_files /tmp/model-output/src/fido/rocq/ \
    && find /workspace/_build/default -maxdepth 1 -name '*.pymap' -exec cp {} /tmp/model-output/src/fido/rocq/ \; \
    && printf '%s\n' \
        '"""Python extracted from Rocq models in models/.' \
        '' \
        'Rebuild with:  ./fido make-rocq' \
        'Source:        models/*.v (Rocq)' \
        'Generator:     rocq-python-extraction plugin' \
        '"""' \
        > /tmp/model-output/src/fido/rocq/__init__.py

FROM python-base AS model-format

WORKDIR /workspace
COPY .python-version pyproject.toml uv.lock ./
COPY --from=extract /tmp/model-output/src/fido/rocq /workspace/src/fido/rocq

RUN uv run --no-project --with ruff==0.15.9 ruff format /workspace/src/fido/rocq \
    && cd /workspace/src/fido/rocq \
    && find . -type f ! -name ".build-files" -printf "%P\n" | LC_ALL=C sort > .build-files

FROM scratch AS export

COPY --from=model-format /workspace/src/fido/rocq /src/fido/rocq
COPY --from=extract /workspace/_build /.cache/rocq-models/context/_build

FROM rocq-plugin-source AS test-extract

COPY --chown=opam:opam \
    rocq-python-extraction/test/dune \
    rocq-python-extraction/test/*.v \
    rocq-python-extraction/test/generated_pytest_targets.txt \
    rocq-python-extraction/test/generated_pyright_targets.txt \
    rocq-python-extraction/test/

RUN opam exec -- dune build rocq-python-extraction/test/python.vo \
    && mkdir -p /tmp/pytest-artifacts /tmp/pyright-artifacts \
    && while IFS= read -r name; do \
        cp "/workspace/_build/default/$name" "/tmp/pytest-artifacts/$name"; \
    done < rocq-python-extraction/test/generated_pytest_targets.txt \
    && while IFS= read -r name; do \
        cp "/workspace/_build/default/$name" "/tmp/pyright-artifacts/$name"; \
    done < rocq-python-extraction/test/generated_pyright_targets.txt \
    && if opam exec -- dune build rocq-python-extraction/test/reals_unsupported.vo > /tmp/reals-unsupported.log 2>&1; then \
        cat /tmp/reals-unsupported.log; \
        echo "expected reals_unsupported.v extraction to fail with PYEX041" >&2; \
        exit 1; \
    fi \
    && grep -F "PYTHON_EXTRACTION_DIAGNOSTIC_JSON:" /tmp/reals-unsupported.log \
    && grep -F '"code": "PYEX041"' /tmp/reals-unsupported.log

FROM python-base AS python-deps

WORKDIR /workspace
COPY .python-version pyproject.toml uv.lock ./

RUN --mount=type=cache,id=fido-uv-check,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

FROM python-deps AS python-check-base

WORKDIR /workspace
COPY .dockerignore .lsp.json .python-version docker-bake.hcl dune-workspace fido package.json package-lock.json pyproject.toml pyrightconfig.json uv.lock ./
COPY .githooks/pre-commit .githooks/pre-commit
COPY .github/workflows/ci.yml .github/workflows/
COPY tools/gen_workflows.py tools/build_graph.sh tools/
COPY models/Dockerfile models/Dockerfile
COPY models/dune-project models/dune models/*.v models/
COPY src src
COPY tests tests
COPY rocq-python-extraction/Dockerfile \
    rocq-python-extraction/DIAGNOSTICS.md \
    rocq-python-extraction/META.rocq-python-extraction.template \
    rocq-python-extraction/dune \
    rocq-python-extraction/dune-project \
    rocq-python-extraction/export_pytest_generated.sh \
    rocq-python-extraction/g_python_extraction.mlg \
    rocq-python-extraction/python.ml \
    rocq-python-extraction/rocq-python-extraction.opam \
    rocq-python-extraction/run_generated_pyright.sh \
    rocq-python-extraction/run_in_docker.sh \
    rocq-python-extraction/
COPY rocq-python-extraction/test/*.py rocq-python-extraction/test/
COPY --from=model-format /workspace/src/fido/rocq /workspace/src/fido/rocq

ENV FIDO_ROCQ_PYTEST_ARTIFACTS=prepared
ENV FIDO_SKIP_DOCKER_HELPER_TESTS=1

RUN --mount=type=cache,id=fido-uv-check,target=/root/.cache/uv \
    --mount=type=cache,id=fido-pyright,target=/root/.cache/pyright-python \
    uv sync --frozen

FROM python-check-base AS format

RUN --mount=type=cache,id=fido-uv-check,target=/root/.cache/uv \
    uv run ruff format --check .

FROM python-check-base AS lint

RUN --mount=type=cache,id=fido-uv-check,target=/root/.cache/uv \
    uv run ruff check .

FROM python-check-base AS typecheck

RUN --mount=type=cache,id=fido-uv-check,target=/root/.cache/uv \
    --mount=type=cache,id=fido-pyright,target=/root/.cache/pyright-python \
    uv run pyright

FROM python-deps AS python-test-base

WORKDIR /workspace
COPY .dockerignore .lsp.json .python-version docker-bake.hcl dune-workspace fido package.json package-lock.json pyproject.toml pyrightconfig.json uv.lock ./
COPY .githooks/pre-commit .githooks/pre-commit
COPY .github/workflows/ci.yml .github/workflows/
COPY tools/gen_workflows.py tools/build_graph.sh tools/
COPY models/Dockerfile models/Dockerfile
COPY models/dune-project models/dune models/*.v models/
COPY src src
COPY rocq-python-extraction/Dockerfile \
    rocq-python-extraction/DIAGNOSTICS.md \
    rocq-python-extraction/META.rocq-python-extraction.template \
    rocq-python-extraction/dune \
    rocq-python-extraction/dune-project \
    rocq-python-extraction/export_pytest_generated.sh \
    rocq-python-extraction/g_python_extraction.mlg \
    rocq-python-extraction/python.ml \
    rocq-python-extraction/rocq-python-extraction.opam \
    rocq-python-extraction/run_generated_pyright.sh \
    rocq-python-extraction/run_in_docker.sh \
    rocq-python-extraction/
COPY --from=model-format /workspace/src/fido/rocq /workspace/src/fido/rocq

ENV FIDO_ROCQ_PYTEST_ARTIFACTS=prepared
ENV FIDO_SKIP_DOCKER_HELPER_TESTS=1

RUN --mount=type=cache,id=fido-uv-check,target=/root/.cache/uv \
    uv sync --frozen

FROM python-test-base AS generated-typecheck

COPY --from=test-extract /tmp/pytest-artifacts /workspace/_build/default
COPY --from=test-extract /tmp/pyright-artifacts /tmp/generated-pyright
COPY rocq-python-extraction/test/pyright_*.py rocq-python-extraction/test/

RUN --mount=type=cache,id=fido-uv-check,target=/root/.cache/uv \
    --mount=type=cache,id=fido-pyright,target=/root/.cache/pyright-python \
    ./rocq-python-extraction/run_generated_pyright.sh /tmp/generated-pyright

FROM python-test-base AS test-unit

COPY tests tests

RUN --mount=type=cache,id=fido-uv-check,target=/root/.cache/uv \
    uv run tests-unit

FROM python-test-base AS test-rocq-generated

COPY --from=test-extract /tmp/pytest-artifacts /workspace/_build/default
COPY rocq-python-extraction/test/*.py rocq-python-extraction/test/

RUN --mount=type=cache,id=fido-uv-check,target=/root/.cache/uv \
    uv run tests-rocq-generated
