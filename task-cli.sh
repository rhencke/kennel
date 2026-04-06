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

REPO=$(cd "$WORK_DIR" && gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)
GH_USER=$(gh api user --jq .login 2>/dev/null || true)

case "$CMD" in
  add)
    TITLE="${1:?usage: task-cli.sh <work_dir> add <title> [description]}"
    DESC="${2:-}"
    (
      flock 7
      python3 -c "
import json, sys, time
title, desc = sys.argv[1], sys.argv[2]
# Reject garbage titles
skip_prefixes = ('PR #', 'New issue #', 'Review on PR', 'CI failure:')
if any(title.startswith(p) for p in skip_prefixes):
    sys.exit(0)
if len(title.strip()) < 5:
    sys.exit(0)
with open(sys.argv[3]) as f: tasks = json.load(f)
if any(t['title'] == title and t['status'] != 'completed' for t in tasks):
    sys.exit(0)  # already exists
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
    # Mark completed and extract thread info
    THREAD_JSON=$(
      flock 7
      python3 -c "
import json, sys
title = sys.argv[1]
with open(sys.argv[2]) as f: tasks = json.load(f)
thread = None
for t in tasks:
    if t['title'] == title and t['status'] != 'completed':
        t['status'] = 'completed'
        thread = t.get('thread')
        break
with open(sys.argv[2], 'w') as f: json.dump(tasks, f, indent=2)
print(json.dumps(thread) if thread else '')
" "$TITLE" "$TASK_FILE"
    ) 7>"$TASK_FILE.lock"

    # If task had a thread, check for new replies and resolve if clean
    if [[ -n "$THREAD_JSON" && "$THREAD_JSON" != "null" ]]; then
      _REPO=$(printf '%s' "$THREAD_JSON" | jq -r '.repo // empty')
      _PR=$(printf '%s' "$THREAD_JSON" | jq -r '.pr // empty')
      _COMMENT_ID=$(printf '%s' "$THREAD_JSON" | jq -r '.comment_id // empty')

      if [[ -n "$_REPO" && -n "$_PR" && -n "$_COMMENT_ID" ]]; then
        # Get the thread: find all replies to this comment
        _LAST_AUTHOR=$(gh api "repos/$_REPO/pulls/$_PR/comments" \
          --jq "[.[] | select(.in_reply_to_id == $_COMMENT_ID or .id == $_COMMENT_ID)] | sort_by(.created_at) | last | .user.login" \
          2>/dev/null || true)

        # Only resolve if last reply is from us (fido)
        if [[ "$_LAST_AUTHOR" == "$GH_USER" ]]; then
          # Find the thread node_id via GraphQL
          _OWNER="${_REPO%/*}"
          _REPO_NAME="${_REPO#*/}"
          _THREAD_ID=$(gh api graphql \
            -F owner="$_OWNER" -F repo="$_REPO_NAME" -F pr="$_PR" \
            -f query='query($owner:String!,$repo:String!,$pr:Int!){
              repository(owner:$owner,name:$repo){
                pullRequest(number:$pr){
                  reviewThreads(first:100){
                    nodes{
                      id isResolved
                      comments(first:1){ nodes{ databaseId } }
                    }
                  }
                }
              }
            }' \
            --jq ".data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | select(.comments.nodes[0].databaseId == $_COMMENT_ID) | .id" \
            2>/dev/null || true)

          if [[ -n "$_THREAD_ID" ]]; then
            gh api graphql -F threadId="$_THREAD_ID" \
              -f query='mutation($threadId:ID!){resolveReviewThread(input:{threadId:$threadId}){thread{id}}}' \
              2>/dev/null || true
            echo "thread resolved: $_THREAD_ID"
          fi
        else
          echo "thread has new replies from $_LAST_AUTHOR — not resolving"
        fi
      fi
    fi
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
