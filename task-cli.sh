#!/usr/bin/env bash
# task-cli.sh — bash interface to tasks.json with flock
# Usage:
#   task-cli.sh <work_dir> add <title> [description]
#   task-cli.sh <work_dir> complete <title>
#   task-cli.sh <work_dir> list
set -euo pipefail

WORK_DIR="${1:?usage: task-cli.sh <work_dir> <command> [args...]}"
CMD="${2:?usage: task-cli.sh <work_dir> <command> [args...]}"
shift 2

TASK_FILE="$(cd "$WORK_DIR" && git rev-parse --git-dir)/fido/tasks.json"
mkdir -p "$(dirname "$TASK_FILE")"
[[ -f "$TASK_FILE" ]] || echo '[]' > "$TASK_FILE"

case "$CMD" in
  add)
    TITLE="${1:?usage: task-cli.sh <work_dir> add <title> [description]}"
    DESC="${2:-}"
    (
      flock 7
      python3 -c "
import json, sys, time
title, desc = sys.argv[1], sys.argv[2]
with open(sys.argv[3]) as f: tasks = json.load(f)
tasks.append({
    'id': str(int(time.time() * 1000)),
    'title': title,
    'description': desc,
    'status': 'pending',
    'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
})
with open(sys.argv[3], 'w') as f: json.dump(tasks, f, indent=2)
" "$TITLE" "$DESC" "$TASK_FILE"
    ) 7>"$TASK_FILE.lock"
    ;;
  complete)
    TITLE="${1:?usage: task-cli.sh <work_dir> complete <title>}"
    (
      flock 7
      python3 -c "
import json, sys
title = sys.argv[1]
with open(sys.argv[2]) as f: tasks = json.load(f)
for t in tasks:
    if t['title'] == title and t['status'] != 'completed':
        t['status'] = 'completed'
        break
with open(sys.argv[2], 'w') as f: json.dump(tasks, f, indent=2)
" "$TITLE" "$TASK_FILE"
    ) 7>"$TASK_FILE.lock"
    ;;
  list)
    (
      flock -s 7
      cat "$TASK_FILE"
    ) 7>"$TASK_FILE.lock"
    ;;
  *)
    echo "unknown command: $CMD" >&2
    exit 1
    ;;
esac
