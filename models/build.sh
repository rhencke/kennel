#!/usr/bin/env bash
# Extract Rocq models to Python and deposit in kennel/models_generated/.
#
# Usage:
#   models/build.sh          (from repo root or any subdirectory)
#   make models              (via top-level Makefile — added in next task)
#
# Requirements:
#   docker  — with buildx support (same image used by rocq-python-extraction CI)
#   uv      — for `uv run ruff format` on the extracted output
#
# What it does:
#   1. Builds the rocq-python-extraction Docker image (cached after first run).
#   2. Inside the container, builds the unified dune workspace:
#        rocq-python-extraction/ (plugin)  →  models/ (theory)
#      dune resolves the plugin dependency within the workspace — no separate
#      install step needed.
#   3. Collects the .py side-effect files produced by `Python Extraction`.
#   4. Applies `ruff format` to make generated code match kennel's style.
#   5. Copies formatted files to kennel/models_generated/.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

echo "==> building extraction plugin Docker image"
docker buildx build --load -t rocq-python-extraction:ci rocq-python-extraction/

OUTDIR=$(mktemp -d)
trap 'rm -rf "$OUTDIR"' EXIT

echo "==> running extraction (first run may take a few minutes)"
docker run --rm \
  -v "$REPO:/src:ro" \
  -v "$OUTDIR:/out" \
  rocq-python-extraction:ci \
  bash -euo pipefail -c '
    cp -r /src /tmp/work
    chmod -R u+w /tmp/work
    cd /tmp/work

    # Build the unified workspace: dune handles plugin → theory ordering.
    opam exec -- dune build models/

    # Python Extraction side-effects land in _build/default/models/.
    find _build/default/models -maxdepth 1 -name "*.py" -exec cp {} /out/ \;
  '

echo "==> applying ruff format to extracted Python"
mkdir -p kennel/models_generated
COUNT=0
for f in "$OUTDIR"/*.py; do
  [ -f "$f" ] || continue
  uv run ruff format "$f"
  cp "$f" kennel/models_generated/
  COUNT=$((COUNT + 1))
done

echo "models build complete: $COUNT file(s) → kennel/models_generated/"
