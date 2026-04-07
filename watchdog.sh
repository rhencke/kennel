#!/usr/bin/env bash
# watchdog.sh — kill stale fido processes and restart
# Run from cron every 5 minutes
set -euo pipefail

WORK_DIR="${1:?usage: watchdog.sh <work_dir>}"
LOCK="$(cd "$WORK_DIR" && git rev-parse --absolute-git-dir)/fido/lock"
LOG="$HOME/log/fido.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$HOME/log"
log() { printf '[%s] watchdog: %s\n' "$(date '+%H:%M:%S')" "$*" >> "$HOME/log/watchdog.log"; }

# Is fido running?
flock -n "$LOCK" true 2>/dev/null && {
  # Lock free — fido not running, cron will handle restart
  exit 0
}

# Fido is running — check staleness
if [[ -f "$LOG" ]] && ! find "$LOG" -mmin -10 | grep -q .; then
  log "fido stale (log untouched 10+ min) — killing"
  for pid in $(lsof "$LOCK" 2>/dev/null | awk 'NR>1{print $2}' | sort -u); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 2
  log "restarting"
  nohup bash "$SCRIPT_DIR/work.sh" "$WORK_DIR" >> "$LOG" 2>&1 &
fi
