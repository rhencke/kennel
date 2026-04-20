#!/bin/sh
set -eu

KEEP_WORKSPACE=0
OUT_DIR=
TARGETS_FILE=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --keep-workspace)
      KEEP_WORKSPACE=1
      shift
      ;;
    --output-dir)
      OUT_DIR=$2
      shift 2
      ;;
    --targets-file)
      TARGETS_FILE=$2
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [ -n "$OUT_DIR" ] && [ -z "$TARGETS_FILE" ]; then
  echo "--output-dir requires --targets-file" >&2
  exit 2
fi

if [ -n "$TARGETS_FILE" ] && [ -z "$OUT_DIR" ]; then
  echo "--targets-file requires --output-dir" >&2
  exit 2
fi

if [ -n "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
fi

if [ "$#" -lt 2 ]; then
  echo "usage: $0 [--keep-workspace] [--output-dir DIR --targets-file FILE] <workdir-under-/tmp/work> <command> [args...]" >&2
  exit 2
fi

WORKDIR=$1
shift

if [ -n "$OUT_DIR" ]; then
  CID=$(
    docker create -v "$PWD:/src:ro" rocq-python-extraction:ci \
      bash -euo pipefail -c '
        mkdir -p /tmp/work
        tar -C /src --exclude=.git --exclude=.ruff_cache --exclude=.pytest_cache -cf - . | tar -C /tmp/work -xf -
        chmod -R u+w /tmp/work
        if [ "$1" = 0 ]; then
          rm -f /tmp/work/dune-workspace
        fi
        cd "/tmp/work/$2"
        shift 2
        "$@"
      ' bash "$KEEP_WORKSPACE" "$WORKDIR" "$@"
  )
  cleanup() {
    docker rm -f "$CID" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT HUP INT TERM
  docker start -a "$CID"
  while IFS= read -r name; do
    docker cp "$CID:/tmp/work/$WORKDIR/_build/default/$name" "$OUT_DIR/$name"
  done < "$TARGETS_FILE"
  cleanup
  trap - EXIT HUP INT TERM
else
  docker run --rm -v "$PWD:/src:ro" rocq-python-extraction:ci \
    bash -euo pipefail -c '
      mkdir -p /tmp/work
      tar -C /src --exclude=.git --exclude=.ruff_cache --exclude=.pytest_cache -cf - . | tar -C /tmp/work -xf -
      chmod -R u+w /tmp/work
      if [ "$1" = 0 ]; then
        rm -f /tmp/work/dune-workspace
      fi
      cd "/tmp/work/$2"
      shift 2
      "$@"
    ' bash "$KEEP_WORKSPACE" "$WORKDIR" "$@"
fi
