#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PERSONA_FILE="$SCRIPT_DIR/sub/persona.md"

if [[ ! -f "$PERSONA_FILE" ]]; then
  echo "persona file not found: $PERSONA_FILE" >&2
  exit 1
fi

cd "$HOME/home-runner"
export CLAUDE_CODE_NO_FLICKER=1

args=("$@")
if [[ ${#args[@]} -eq 0 ]]; then
  args=("/remote-control")
fi

exec nice -n19 claude \
  --permission-mode=bypassPermissions \
  --continue \
  --append-system-prompt "$(cat "$PERSONA_FILE")" \
  "${args[@]}"
