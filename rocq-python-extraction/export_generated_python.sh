#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <output-dir>" >&2
  exit 2
fi

OUT_DIR=$1
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

./rocq-python-extraction/run_in_docker.sh \
  --output-dir "$OUT_DIR" \
  --targets-file "$PWD/rocq-python-extraction/test/generated_pyright_targets.txt" \
  rocq-python-extraction \
  opam exec -- dune build test/phase4.vo test/phase8.vo test/phase9.vo
