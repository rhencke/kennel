#!/usr/bin/env bash
# sync-tasks.sh — sync tasks.json → PR body work queue
# Triggered by: PostToolUse hook, kennel webhook, cron
# Protected by flock to prevent concurrent runs
# If tasks.json changes during a sync, re-syncs automatically
set -euo pipefail

WORK_DIR="${1:-$PWD}"
cd "$WORK_DIR"

mkdir -p "$HOME/log"
log() {
  local msg
  msg="$(printf '[%s] sync: %s' "$(date '+%H:%M:%S')" "$*")"
  printf '%s\n' "$msg"
  printf '%s\n' "$msg" >> "$HOME/log/sync-tasks.log"
}

# ── Lock ──────────────────────────────────────────────────────────────────
SYNC_LOCK="$(git rev-parse --absolute-git-dir)/fido/sync.lock"
mkdir -p "$(dirname "$SYNC_LOCK")"
exec 8>"$SYNC_LOCK"
flock -n 8 || { log "another sync running — skipping"; exit 0; }

# ── Find current PR ──────────────────────────────────────────────────────
REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
STATE_FILE="$(git rev-parse --git-dir)/fido/state.json"

if [[ ! -f "$STATE_FILE" ]]; then
  log "no state file — nothing to sync"
  exit 0
fi

CURRENT_ISSUE=$(jq -r '.issue // empty' "$STATE_FILE")
if [[ -z "$CURRENT_ISSUE" ]]; then
  log "no current issue — nothing to sync"
  exit 0
fi

GH_USER=$(gh api user --jq .login)
_PR_JSON=$(gh pr list --repo "$REPO" --state open --json number,headRefName,author \
  --search "#$CURRENT_ISSUE in:body" 2>/dev/null \
  | jq -r --arg user "$GH_USER" '[.[] | select(.author.login == $user)] | .[0] // empty')
PR=$(printf '%s' "$_PR_JSON" | jq -r '.number // empty')

if [[ -z "$PR" ]]; then
  log "no open PR for issue #$CURRENT_ISSUE — nothing to sync"
  exit 0
fi

TASK_FILE="$(git rev-parse --git-dir)/fido/tasks.json"
if [[ ! -f "$TASK_FILE" ]]; then
  log "no tasks.json — nothing to sync"
  exit 0
fi

# ── Auto-complete ASK tasks with resolved threads ─────────────────────────
_OWNER="${REPO%/*}"
_REPO_NAME="${REPO#*/}"
_ASK_TASKS=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f: tasks = json.load(f)
for t in tasks:
    if t.get('status') == 'pending' and t.get('title','').startswith('ASK:') and t.get('thread'):
        th = t['thread']
        print(f\"{t['id']}\t{th.get('comment_id','')}\")
" "$TASK_FILE" 2>/dev/null || true)

if [[ -n "$_ASK_TASKS" ]]; then
  # Fetch all resolved thread first-comment IDs in one call
  _RESOLVED_IDS=$(gh api graphql \
    -F owner="$_OWNER" -F repo="$_REPO_NAME" -F pr="$PR" \
    -f query='query($owner:String!,$repo:String!,$pr:Int!){
      repository(owner:$owner,name:$repo){
        pullRequest(number:$pr){
          reviewThreads(first:100){
            nodes{
              isResolved
              comments(first:1){ nodes{ databaseId } }
            }
          }
        }
      }
    }' \
    --jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved) | .comments.nodes[0].databaseId] | .[]' \
    2>/dev/null || true)

  while IFS=$'\t' read -r task_id comment_id; do
    [[ -z "$comment_id" ]] && continue
    if echo "$_RESOLVED_IDS" | grep -qx "$comment_id"; then
      log "ASK task $task_id: thread resolved — marking complete"
      bash "$(dirname "$0")/task-cli.sh" "$WORK_DIR" complete "$(python3 -c "
import json, sys
with open(sys.argv[1]) as f: tasks = json.load(f)
for t in tasks:
    if t['id'] == sys.argv[2]: print(t['title']); break
" "$TASK_FILE" "$task_id")" 2>/dev/null || true
    fi
  done <<< "$_ASK_TASKS"
fi

# ── Sync loop: re-run if tasks.json changed during sync ───────────────────
while true; do
  MTIME_BEFORE=$(stat -c %Y "$TASK_FILE" 2>/dev/null || echo 0)

  # Lock task file for reading
  exec 7<"$TASK_FILE"
  flock -s 7
  TASKS=$(cat "$TASK_FILE")
  exec 7<&-

  if [[ -z "$TASKS" || "$TASKS" == "[]" ]]; then
    log "no tasks — nothing to sync"
    break
  fi

  log "syncing task list → PR #$PR"

  # ── Format as work queue markdown ───────────────────────────────────────
  QUEUE=$(printf '%s' "$TASKS" | python3 -c "
import json, sys

tasks = json.load(sys.stdin)
ci = []
pr_comment = []
other = []
completed = []

def fmt(t):
    title = t.get('title', '')
    url = (t.get('thread') or {}).get('url', '')
    return f'[{title}]({url})' if url else title

for t in tasks:
    status = t.get('status', 'pending')
    title = t.get('title', '')
    if status == 'completed':
        completed.append(fmt(t))
    elif status in ('pending', 'in_progress'):
        if title.startswith('CI failure:'):
            ci.append(fmt(t))
        elif t.get('thread'):
            pr_comment.append(fmt(t))
        else:
            other.append(fmt(t))

pending = ci + pr_comment + other

lines = []
for i, display in enumerate(pending):
    suffix = ' **→ next**' if i == 0 else ''
    lines.append(f'- [ ] {display}{suffix}')

if completed:
    lines.append('')
    lines.append(f'<details><summary>Completed ({len(completed)})</summary>')
    lines.append('')
    for display in completed:
        lines.append(f'- [x] {display}')
    lines.append('</details>')

print('\n'.join(lines))
")

  # ── Update PR body ──────────────────────────────────────────────────────
  CURRENT_BODY=$(gh pr view "$PR" --repo "$REPO" --json body --jq .body)

  if ! echo "$CURRENT_BODY" | grep -q "WORK_QUEUE_START"; then
    log "PR #$PR has no work queue markers — skipping"
    break
  fi

  NEW_BODY=$(python3 -c "
import sys
body = sys.stdin.read()
queue = sys.argv[1]
start_marker = '<!-- WORK_QUEUE_START -->'
end_marker = '<!-- WORK_QUEUE_END -->'
start = body.find(start_marker)
end = body.find(end_marker)
if start == -1 or end == -1:
    print(body, end='')
    sys.exit()
start += len(start_marker)
print(body[:start] + '\n' + queue + '\n' + body[end:], end='')
" "$QUEUE" <<< "$CURRENT_BODY")

  gh pr edit "$PR" --repo "$REPO" --body "$NEW_BODY"
  log "PR #$PR work queue synced"

  # Check if tasks.json changed during sync — if so, loop
  MTIME_AFTER=$(stat -c %Y "$TASK_FILE" 2>/dev/null || echo 0)
  if [[ "$MTIME_AFTER" != "$MTIME_BEFORE" ]]; then
    log "tasks.json changed during sync — re-syncing"
    continue
  fi
  break
done
