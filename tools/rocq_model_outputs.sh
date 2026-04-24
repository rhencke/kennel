#!/usr/bin/env bash
set -euo pipefail

awk '
  /^[[:space:]]*Python[[:space:]]+Extraction[[:space:]]+/ {
    name = $3
    sub(/[.]$/, "", name)
    if (name != "") {
      print name ".py"
      print name ".pymap"
    }
  }
  /^[[:space:]]*Python[[:space:]]+Module[[:space:]]+Extraction[[:space:]]+/ {
    name = $4
    sub(/[.]$/, "", name)
    if (name != "") {
      print name ".py"
      print name ".pymap"
    }
  }
  /^[[:space:]]*Python[[:space:]]+File[[:space:]]+Extraction[[:space:]]+/ {
    name = $4
    sub(/[.]$/, "", name)
    if (name != "") {
      print name ".py"
      print name ".pymap"
    }
  }
' "$@" | LC_ALL=C sort -u
