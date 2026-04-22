#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <generated-python-dir>" >&2
  exit 2
fi

OUT_DIR=$1

cat >"$OUT_DIR/pyrightconfig.json" <<'EOF'
{
  "include": [
    "myopt_flatten.py",
    "mylist_is_empty.py",
    "roseforest_is_empty.py",
    "list_map.py",
    "zeros.py",
    "zeros_pair.py",
    "repeat_tree.py",
    "tree_root_of_repeat.py",
    "Phase10Mod.py",
    "pyright_list_map_check.py",
    "pyright_coinductive_check.py",
    "pyright_modules_check.py",
    "pyright_strings_bytes_check.py"
  ],
  "executionEnvironments": [
    {
      "root": ".",
      "extraPaths": ["."]
    }
  ],
  "reportUnusedImport": false,
  "reportUnusedVariable": false,
  "reportUnknownLambdaType": false,
  "reportRedeclaration": false
}
EOF

cp rocq-python-extraction/test/pyright_list_map_check.py \
  "$OUT_DIR/pyright_list_map_check.py"
cp rocq-python-extraction/test/pyright_coinductive_check.py \
  "$OUT_DIR/pyright_coinductive_check.py"
cp rocq-python-extraction/test/pyright_modules_check.py \
  "$OUT_DIR/pyright_modules_check.py"
cp rocq-python-extraction/test/pyright_strings_bytes_check.py \
  "$OUT_DIR/pyright_strings_bytes_check.py"

uv run pyright -p "$OUT_DIR/pyrightconfig.json"
