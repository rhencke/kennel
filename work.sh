#!/usr/bin/env bash
# fido/work.sh — one step of the fido loop
# exit 0: no more eligible issues (all done)
# exit 1: did work (re-run immediately via exec tail-call)
# exit 2: lock held / transient failure (retry later)
set -euo pipefail

WORK_DIR="${1:-$PWD}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

mkdir -p "$HOME/log"
log() {
  local msg
  msg="$(printf '[%s] %s' "$(date '+%H:%M:%S')" "$*")"
  printf '%s\n' "$msg"
  printf '%s\n' "$msg" >> "$HOME/log/fido.log"
}

# ── Lock (prevent duplicate workers) ──────────────────────────────────────
FIDO_DIR_EARLY="$(git rev-parse --git-dir)/fido"
mkdir -p "$FIDO_DIR_EARLY"
LOCK_FILE="$FIDO_DIR_EARLY/lock"
exec 9>"$LOCK_FILE"
flock -n 9 || { log "another fido is running — exiting"; exit 2; }

# ── Repo context ───────────────────────────────────────────────────────────
REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
OWNER=${REPO%/*}
REPO_NAME=${REPO#*/}
GH_USER=$(gh api user --jq .login)
DEFAULT_BRANCH=$(gh repo view --json defaultBranchRef --jq .defaultBranchRef.name)

FORK_REMOTE=origin
UPSTREAM_REMOTE=origin

log "repo=$REPO user=$GH_USER fork=$FORK_REMOTE upstream=$UPSTREAM_REMOTE/$DEFAULT_BRANCH"

# ── Fido dir inside .git (all ephemeral files live here) ───────────────────
FIDO_DIR="$(git rev-parse --git-dir)/fido"
mkdir -p "$FIDO_DIR"
STATE_FILE="$FIDO_DIR/state.json"

# ── PostCompact hook so sub-Claude re-reads skill instructions after compression
mkdir -p "$WORK_DIR/.claude"
grep -qxF '.claude/settings.local.json' "$(git rev-parse --git-dir)/info/exclude" 2>/dev/null \
  || echo '.claude/settings.local.json' >> "$(git rev-parse --git-dir)/info/exclude"
SETTINGS_LOCAL="$WORK_DIR/.claude/settings.local.json"
COMPACT_SCRIPT="$FIDO_DIR/compact.sh"
cat > "$COMPACT_SCRIPT" <<COMPACTEOF
#!/usr/bin/env bash
printf '[fido PostCompact] Re-reading skill instructions after context compression.\n\n'
for f in "$SCRIPT_DIR/sub/"*.md; do
  printf '## %s\n\n' "\$(basename "\$f")"
  cat "\$f"
  printf '\n\n'
done
COMPACTEOF
chmod +x "$COMPACT_SCRIPT"
HOOK_CMD="bash $COMPACT_SCRIPT"
SYNC_CMD="bash $SCRIPT_DIR/sync-tasks.sh $WORK_DIR &"

python3 -c "
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
cfg = json.loads(p.read_text()) if p.exists() else {}

# PostCompact hook
cfg.setdefault('hooks', {}).setdefault('PostCompact', [])
entry = {'matcher': '', 'hooks': [{'type': 'command', 'command': sys.argv[2]}]}
if entry not in cfg['hooks']['PostCompact']:
    cfg['hooks']['PostCompact'].append(entry)

# PostToolUse hook for task mutations → sync work queue
sync_entry = {'matcher': '', 'hooks': [{'type': 'command', 'command': sys.argv[3]}]}
for tool in ('TaskCreate', 'TaskUpdate', 'TaskDelete', 'TodoWrite', 'TodoRead'):
    key = 'PostToolUse'
    cfg.setdefault('hooks', {}).setdefault(key, [])
    tool_entry = {'matcher': tool, 'hooks': [{'type': 'command', 'command': sys.argv[3]}]}
    if tool_entry not in cfg['hooks'][key]:
        cfg['hooks'][key].append(tool_entry)

p.write_text(json.dumps(cfg, indent=2))
" "$SETTINGS_LOCAL" "$HOOK_CMD" "$SYNC_CMD" 2>/dev/null || true

PROMPT="$FIDO_DIR/prompt"
STREAM="$FIDO_DIR/stream"

cleanup() {
  rm -f "$COMPACT_SCRIPT" "$PROMPT" "$STREAM"
  python3 -c "
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists(): exit()
cfg = json.loads(p.read_text())
compact_cmd = sys.argv[2]
sync_cmd = sys.argv[3]

# Remove PostCompact hook
hooks = cfg.get('hooks', {}).get('PostCompact', [])
cfg['hooks']['PostCompact'] = [h for h in hooks if not any(e.get('command') == compact_cmd for e in h.get('hooks', []))]
if not cfg['hooks']['PostCompact']: del cfg['hooks']['PostCompact']

# Remove PostToolUse hooks
hooks = cfg.get('hooks', {}).get('PostToolUse', [])
cfg['hooks']['PostToolUse'] = [h for h in hooks if not any(e.get('command') == sync_cmd for e in h.get('hooks', []))]
if not cfg['hooks']['PostToolUse']: del cfg['hooks']['PostToolUse']

if not cfg.get('hooks'): cfg.pop('hooks', None)
p.write_text(json.dumps(cfg, indent=2))
" "$SETTINGS_LOCAL" "$HOOK_CMD" "$SYNC_CMD" 2>/dev/null || true
}
die() {
  local code=$?
  local line=${BASH_LINENO[0]}
  log "FATAL: exit $code at line $line"
  cleanup
  exit $code
}
trap die ERR
trap cleanup EXIT INT TERM

# ── Sub-Claude helpers ─────────────────────────────────────────────────────
STREAM_FILTER='
  if .type == "assistant" then
    (.message.content // [])[] |
    if .type == "text" then "[claude] \(.text)"
    elif .type == "tool_use" then "[tool]  \(.name) \(.input | tostring | .[0:120])\n"
    else empty end
  elif .type == "result" then
    "[done]  session=\(.session_id // "?")\n"
  else empty end
'

build_prompt() {   # build_prompt <subskill> <context-string>
  printf '%s\n\n%s\n\n%s\n' "$2" "$(cat "$SCRIPT_DIR/sub/persona.md")" "$(cat "$SCRIPT_DIR/sub/$1.md")" > "$PROMPT"
}

claude_start() {   # start new session; prints session_id to stdout, progress to stderr
  claude --model claude-sonnet-4-6 --output-format stream-json --verbose \
    --dangerously-skip-permissions --print < "$PROMPT" \
    | tee "$STREAM" | stdbuf -oL jq -rj "$STREAM_FILTER" >&2
  jq -r 'select(.type=="result") | .session_id // empty' "$STREAM" | tail -1
}

claude_run() {     # continue or start session; progress to stdout
  if [[ -n "$SESSION_ID" ]]; then
    claude --model claude-sonnet-4-6 --output-format stream-json --verbose \
      --dangerously-skip-permissions --resume "$SESSION_ID" --print < "$PROMPT" \
      | stdbuf -oL jq -rj "$STREAM_FILTER"
  else
    SESSION_ID=$(claude_start)
    log "session: $SESSION_ID"
  fi
}

# ── Main loop (lock held throughout) ──────────────────────────────────────
while true; do

# ── Find current issue ─────────────────────────────────────────────────────
CURRENT_ISSUE=""
if [[ -f "$STATE_FILE" ]]; then
  CURRENT_ISSUE=$(jq -r '.issue // empty' "$STATE_FILE")
fi

if [[ -n "$CURRENT_ISSUE" ]]; then
  ISSUE_STATE=$(gh issue view "$CURRENT_ISSUE" --repo "$REPO" --json state --jq .state 2>/dev/null || echo "OPEN")
  if [[ "$ISSUE_STATE" == "CLOSED" ]]; then
    log "issue #$CURRENT_ISSUE: closed — advancing"
    rm -f "$STATE_FILE"
    CURRENT_ISSUE=""
  fi
fi

# ── Find next issue if needed ──────────────────────────────────────────────
_NEW_ISSUE=false
if [[ -z "$CURRENT_ISSUE" ]]; then
  _NEW_ISSUE=true
  log "finding next eligible issue"
  _QUERY_RAW=$(gh api graphql \
    -F owner="$OWNER" -F repo="$REPO_NAME" -F login="$GH_USER" \
    -f query='query($owner:String!,$repo:String!,$login:String!){
      repository(owner:$owner,name:$repo){
        issues(
          first:50, states:[OPEN],
          filterBy:{assignee:$login},
          orderBy:{field:CREATED_AT,direction:ASC}
        ){
          nodes{
            number title
            subIssues(first:10){ nodes{ state } }
          }
        }
      }
    }' 2>&1) || {
    log "issue query failed (gh exit:$?) — retrying"
    exit 2
  }
  NEXT=$(printf '%s' "$_QUERY_RAW" | jq -r '[
        .data.repository.issues.nodes[]
        | select(
            (.subIssues.nodes | length) == 0
            or (.subIssues.nodes | all(.state == "CLOSED"))
          )
      ] | .[0] | "\(.number)\t\(.title)"' 2>/dev/null || echo "")

  if [[ -z "$NEXT" || "$NEXT" == "null	null" ]]; then
    log "no eligible issues assigned to $GH_USER in $REPO"
    break
  fi

  CURRENT_ISSUE=$(echo "$NEXT" | cut -f1)
  NEXT_TITLE=$(echo "$NEXT" | cut -f2-)
  log "starting issue #$CURRENT_ISSUE: $NEXT_TITLE"
  jq -n --argjson issue "$CURRENT_ISSUE" '{issue: $issue}' > "$STATE_FILE"
  # Only post pickup comment for genuinely new issues (not restarts)
  _ALREADY_COMMENTED=$(gh api "repos/$REPO/issues/$CURRENT_ISSUE/comments" \
    --jq "[.[] | select(.user.login == \"$GH_USER\")] | length" 2>/dev/null || echo "0")
  if [[ "$_ALREADY_COMMENTED" == "0" ]]; then
    _PICKUP_PLAIN="Picking up issue: $NEXT_TITLE"
    _PICKUP_MSG=$(printf '%s\n\nRewrite the following GitHub issue comment in character as Fido. Keep it to 1-2 sentences. Output only the comment text, no quotes, no explanation.\n\n%s' \
      "$(cat "$SCRIPT_DIR/sub/persona.md")" "$_PICKUP_PLAIN" \
      | claude --model claude-opus-4-6 --print 2>/dev/null | head -3)
    : "${_PICKUP_MSG:=$_PICKUP_PLAIN}"
    gh issue comment "$CURRENT_ISSUE" --repo "$REPO" \
      --body "$_PICKUP_MSG" 2>/dev/null || true
  fi
fi

# ── Load issue title ───────────────────────────────────────────────────────
ISSUE_TITLE=$(gh issue view "$CURRENT_ISSUE" --repo "$REPO" --json title --jq .title 2>/dev/null || echo "issue #$CURRENT_ISSUE")
REQUEST="$ISSUE_TITLE (closes #$CURRENT_ISSUE)"

# ── Find or create branch + PR ─────────────────────────────────────────────
_PR_JSON=$(gh pr list --repo "$REPO" --state all --json number,headRefName,state,author \
  --search "#$CURRENT_ISSUE in:body" 2>/dev/null \
  | jq -r --arg user "$GH_USER" '[.[] | select(.author.login == $user)] | .[0] // empty')
EXISTING_PR=$(printf '%s' "$_PR_JSON" | jq -r '.number // empty')
EXISTING_PR_STATE=$(printf '%s' "$_PR_JSON" | jq -r '.state // empty')
EXISTING_SLUG=$(printf '%s' "$_PR_JSON" | jq -r '.headRefName // empty')

if [[ -n "$EXISTING_PR" && "$EXISTING_PR_STATE" == "CLOSED" ]]; then
  log "PR #$EXISTING_PR closed without merge — creating fresh PR"
  EXISTING_PR=""
  EXISTING_SLUG=""
elif [[ -n "$EXISTING_PR" && "$EXISTING_PR_STATE" == "MERGED" ]]; then
  log "PR #$EXISTING_PR already merged — closing issue #$CURRENT_ISSUE"
  gh issue close "$CURRENT_ISSUE" --repo "$REPO" 2>/dev/null || true
  echo '[]' > "$FIDO_DIR/tasks.json" 2>/dev/null || true
  rm -f "$STATE_FILE"
  git checkout "$DEFAULT_BRANCH" 2>/dev/null || true
  git pull "$FORK_REMOTE" "$DEFAULT_BRANCH" --ff-only 2>/dev/null || true
  [[ -n "$EXISTING_SLUG" ]] && git branch -d "$EXISTING_SLUG" 2>/dev/null || true
  continue
fi

if [[ -n "$EXISTING_SLUG" ]]; then
  SLUG="$EXISTING_SLUG"
  PR="$EXISTING_PR"
  log "resuming PR #$PR on branch $SLUG"
  git fetch "$UPSTREAM_REMOTE"
  git checkout "$SLUG" 2>/dev/null \
    || git checkout -b "$SLUG" --track "$FORK_REMOTE/$SLUG"
  # Check if tasks.json has any tasks — run setup only if never planned
  _ALL_TASKS=$(bash "$SCRIPT_DIR/task-cli.sh" "$WORK_DIR" list 2>/dev/null \
    | jq -r 'length' 2>/dev/null || echo "0")
  SESSION_ID=""
  if [[ "$_ALL_TASKS" -eq 0 ]]; then
    log "PR #$PR has no tasks — running setup"
    build_prompt setup \
"Request: $REQUEST
Repo: $REPO
Branch: $SLUG
PR: $PR
Fork remote: $FORK_REMOTE
Upstream: $UPSTREAM_REMOTE/$DEFAULT_BRANCH"
    SESSION_ID=$(claude_start)
    log "session: $SESSION_ID"
  fi
else
  # Generate slug
  _SLUG_RAW=$(printf 'Output ONLY a git branch name: 2-4 lowercase words separated by hyphens, no issue numbers, summarising this request. No explanation, no punctuation, just the branch name.\n\nRequest: %s' "$REQUEST" \
    | claude --model claude-haiku-4-5-20251001 --print 2>/dev/null | head -1 \
    | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-*//;s/-*$//' | cut -c1-40)
  if [[ -z "$_SLUG_RAW" || ${#_SLUG_RAW} -lt 3 ]]; then
    _SLUG_RAW=$(printf '%s' "$REQUEST" \
      | sed 's/(closes #[0-9]*)//g' \
      | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' \
      | sed 's/^-*//;s/-*$//' | cut -c1-40)
  fi
  SLUG="$_SLUG_RAW"
  log "new branch: $SLUG"
  git fetch "$UPSTREAM_REMOTE"
  git checkout -b "$SLUG" "$UPSTREAM_REMOTE/$DEFAULT_BRANCH" 2>/dev/null \
    || git checkout "$SLUG"
  git commit --allow-empty -m "wip: start"
  git push -u "$FORK_REMOTE" "$SLUG"

  # Run setup BEFORE creating PR — plans tasks into tasks.json
  log "running setup (pre-PR)"
  build_prompt setup \
"Request: $REQUEST
Repo: $REPO
Branch: $SLUG
Fork remote: $FORK_REMOTE
Upstream: $UPSTREAM_REMOTE/$DEFAULT_BRANCH"
  SESSION_ID=$(claude_start)
  log "session: $SESSION_ID"

  # Build PR body with tasks already populated
  _PR_BODY_PLAIN="Working on: $REQUEST. Implementation in progress."
  _PR_BODY_TEXT=$(claude --model claude-opus-4-6 --print \
    --system "You are a GitHub PR description writer. Output ONLY the description text — no preamble, no thinking, no quotes, no markdown headers. Your first word is the first word of the description." \
    -p "$(cat "$SCRIPT_DIR/sub/persona.md")

Write a 2-3 sentence pull request description for: $_PR_BODY_PLAIN" 2>/dev/null)
  : "${_PR_BODY_TEXT:=$_PR_BODY_PLAIN}"

  # Format tasks from tasks.json into work queue
  _TASK_QUEUE=$(bash "$SCRIPT_DIR/task-cli.sh" "$WORK_DIR" list 2>/dev/null \
    | jq -r '[.[] | select(.status == "pending")] | to_entries | map("- [ ] \(.value.title)" + if .key == 0 then " **→ next**" else "" end) | .[]' 2>/dev/null || true)
  : "${_TASK_QUEUE:=<!-- no tasks yet -->}"

  _PR_BODY="$(printf '%s\n\n---\n\n## Work queue\n\n<!-- WORK_QUEUE_START -->\n%s\n<!-- WORK_QUEUE_END -->\n\nFixes #%s' "$_PR_BODY_TEXT" "$_TASK_QUEUE" "$CURRENT_ISSUE")"
  PR_URL=$(gh pr create --draft \
    --title "$REQUEST" \
    --body "$_PR_BODY" \
    --base "$DEFAULT_BRANCH" \
    --head "$SLUG" \
    --repo "$REPO")
  PR=$(printf '%s' "$PR_URL" | grep -oE '[0-9]+$')
  log "PR: #$PR opened with $(echo "$_TASK_QUEUE" | grep -c '^- ' || echo 0) tasks"
fi
log "PR: #$PR  https://github.com/$REPO/pull/$PR"

# ── Seed tasks.json from PR body (only if tasks.json is empty/missing) ────
TASK_FILE="$FIDO_DIR/tasks.json"
_EXISTING=$(cat "$TASK_FILE" 2>/dev/null || echo "[]")
if [[ "$_EXISTING" == "[]" || "$_EXISTING" == "" ]]; then
  log "seeding tasks.json from PR body"
  _SEED_BODY=$(gh pr view "$PR" --repo "$REPO" --json body --jq .body 2>/dev/null || true)
  _SEED_TASKS=$(printf '%s' "$_SEED_BODY" \
    | sed -n '/WORK_QUEUE_START/,/WORK_QUEUE_END/p' \
    | { grep '^- \[ \]' || true; } \
    | sed 's/^- \[ \] //; s/ \*\*→ next\*\*//')
  if [[ -n "$_SEED_TASKS" ]]; then
    while IFS= read -r task_title; do
      [[ -z "$task_title" ]] && continue
      bash "$SCRIPT_DIR/task-cli.sh" "$WORK_DIR" add "$task_title" 2>/dev/null || true
    done <<< "$_SEED_TASKS"
    bash "$SCRIPT_DIR/sync-tasks.sh" "$WORK_DIR" &
    log "seeded $(echo "$_SEED_TASKS" | wc -l) tasks"
  fi
fi

# ── CI ─────────────────────────────────────────────────────────────────────
log "checking: ci"
FAILING=$(gh pr checks "$PR" --repo "$REPO" --json name,state,link 2>/dev/null \
  | jq -r '[.[] | select(.state == "FAILURE" or .state == "ERROR")] | .[0].name // empty' \
  || true)

if [[ -n "$FAILING" ]]; then
  log "CI failing: $FAILING"
  RUN_URL=$(gh pr checks "$PR" --repo "$REPO" --json name,state,link 2>/dev/null \
    | jq -r --arg n "$FAILING" '.[] | select(.name == $n) | .link' || true)
  RUN_ID=$(printf '%s' "$RUN_URL" | grep -oP 'runs/\K[0-9]+' \
           || printf '%s' "$RUN_URL" | sed 's|.*/runs/\([0-9]*\).*|\1|')
  FAILURE_LOG=$(gh run view "$RUN_ID" --log-failed 2>&1 | tail -200)
  CI_THREADS=$(gh api graphql \
    -F owner="$OWNER" -F repo="$REPO_NAME" -F pr="$PR" \
    -f query='query($owner:String!,$repo:String!,$pr:Int!){
      repository(owner:$owner,name:$repo){
        pullRequest(number:$pr){
          reviewThreads(first:100){
            nodes{
              id isResolved
              comments(first:50){
                nodes{author{login} body url createdAt databaseId}
              }
            }
          }
        }
      }
    }' \
    | jq --arg user "$GH_USER" --arg check "$FAILING" '[
        .data.repository.pullRequest.reviewThreads.nodes[]
        | select(.isResolved == false)
        | select(
            (.comments.nodes | last | .author.login) != $user
          )
        | select(
            (.comments.nodes[].body | ascii_downcase | contains($check | ascii_downcase))
            or (.comments.nodes[].body | ascii_downcase | contains("ci"))
            or (.comments.nodes[].body | ascii_downcase | contains("lint"))
            or (.comments.nodes[].body | ascii_downcase | contains("format"))
          )
        | {
            first_author: .comments.nodes[0].author.login,
            first_body:   .comments.nodes[0].body,
            last_author:  (.comments.nodes | last | .author.login),
            last_body:    (.comments.nodes | last | .body),
            url:          .comments.nodes[0].url
          }
      ]' 2>/dev/null || echo "[]")
  build_prompt ci \
"PR: $PR
Repo: $REPO
Branch: $SLUG
Upstream: $UPSTREAM_REMOTE/$DEFAULT_BRANCH
Failing check: $FAILING

Failure log (last 200 lines):
$FAILURE_LOG

Review threads related to this CI failure (JSON — may be empty):
$CI_THREADS"
  claude_run
  log "CI fix done"
  bash "$SCRIPT_DIR/task-cli.sh" "$WORK_DIR" complete "CI failure: $FAILING" 2>/dev/null || true
  bash "$SCRIPT_DIR/sync-tasks.sh" "$WORK_DIR" &
  continue
fi

# ── Review-level feedback (non-inline "Request changes" body) ─────────────
log "checking: review feedback"
REVIEW_FEEDBACK=$(gh pr view "$PR" --repo "$REPO" --json reviews,commits \
  | jq -r --arg owner "$OWNER" '
    ([ .reviews[] | select(.author.login == $owner) ] | last) as $review
    | if $review.state != "CHANGES_REQUESTED" then empty
      else
        # Skip if there are commits after the review
        (.commits | last | .committedDate // "") as $last_commit
        | if ($last_commit > ($review.submittedAt // "")) then empty
          else $review.body // empty
          end
      end' 2>/dev/null || true)

if [[ -n "$REVIEW_FEEDBACK" ]]; then
  log "review feedback from $OWNER"
  build_prompt task \
"PR: $PR
Repo: $REPO
Branch: $SLUG
Upstream: $UPSTREAM_REMOTE/$DEFAULT_BRANCH

Task title: Address review feedback from $OWNER
Task description: $OWNER submitted a review requesting changes with the following feedback. Address it, commit, and push.

Review feedback:
$REVIEW_FEEDBACK"
  claude_run
  log "review feedback done"
  bash "$SCRIPT_DIR/task-cli.sh" "$WORK_DIR" complete "Address review feedback from $OWNER" 2>/dev/null || true
  bash "$SCRIPT_DIR/sync-tasks.sh" "$WORK_DIR" &
  continue
fi

# ── Threads ────────────────────────────────────────────────────────────────
log "checking: threads"
THREADS=$(gh api graphql \
  -F owner="$OWNER" -F repo="$REPO_NAME" -F pr="$PR" \
  -f query='query($owner:String!,$repo:String!,$pr:Int!){
    repository(owner:$owner,name:$repo){
      pullRequest(number:$pr){
        reviewThreads(first:100){
          nodes{
            id isResolved
            comments(first:50){
              nodes{author{login} body url createdAt databaseId}
            }
          }
        }
      }
    }
  }' \
  | jq --arg user "$GH_USER" --arg owner "$OWNER" --argjson bots '["copilot[bot]"]' '{
      threads: [
        .data.repository.pullRequest.reviewThreads.nodes[]
        | select(.isResolved == false)
        | select(
            (.comments.nodes | last | .author.login) != $user
          )
        | select(
            (.comments.nodes | last | .author.login) as $author
            | $author == $owner or ($bots | index($author) != null)
          )
        | {
            id,
            is_bot: (.comments.nodes[0].author.login | endswith("[bot]")),
            first_author: .comments.nodes[0].author.login,
            first_db_id:  .comments.nodes[0].databaseId,
            first_body:   .comments.nodes[0].body,
            last_author:  (.comments.nodes | last | .author.login),
            last_body:    (.comments.nodes | last | .body),
            url:          .comments.nodes[0].url,
            total:        (.comments.nodes | length)
          }
      ]
    }')

THREAD_COUNT=$(printf '%s' "$THREADS" | jq '.threads | length')
if [[ "$THREAD_COUNT" -gt 0 ]]; then
  log "unresolved threads: $THREAD_COUNT"
  build_prompt comments \
"PR: $PR
Repo: $REPO
Owner: $OWNER
Repo name: $REPO_NAME
Branch: $SLUG
Upstream: $UPSTREAM_REMOTE/$DEFAULT_BRANCH
GitHub user: $GH_USER

Unresolved threads (JSON):
$THREADS"
  claude_run
  log "threads done"
  bash "$SCRIPT_DIR/sync-tasks.sh" "$WORK_DIR" &
  continue
fi

# ── Task ───────────────────────────────────────────────────────────────────
log "checking: tasks"
_TASK_JSON=$(bash "$SCRIPT_DIR/task-cli.sh" "$WORK_DIR" list 2>/dev/null || echo "[]")
# Prioritise: CI fix → comment-originated (has thread) → rest (skip ASK/DEFER)
PENDING=$(printf '%s' "$_TASK_JSON" | jq -r '
  [.[] | select(.status == "pending")
       | select((.title | ascii_downcase | startswith("ask:")) | not)
       | select((.title | ascii_downcase | startswith("defer:")) | not)
  ] |
  (map(select(.title | startswith("CI failure:"))) | .[0].title // empty) //
  (map(select(.thread != null)) | .[0].title // empty) //
  (.[0].title // empty)' 2>/dev/null || true)

if [[ -n "$PENDING" ]]; then
  log "task: $PENDING"
  build_prompt task \
"PR: $PR
Repo: $REPO
Branch: $SLUG
Upstream: $UPSTREAM_REMOTE/$DEFAULT_BRANCH

Task title: $PENDING"
  claude_run
  log "task done: $PENDING"
  bash "$SCRIPT_DIR/task-cli.sh" "$WORK_DIR" complete "$PENDING" 2>/dev/null || true
  bash "$SCRIPT_DIR/sync-tasks.sh" "$WORK_DIR" &
  continue
fi

# ── Promote or merge ───────────────────────────────────────────────────────
log "checking: review status"

# Check approval first — merge takes priority over promote
_REVIEWS_JSON=$(gh pr view "$PR" --repo "$REPO" --json reviews,isDraft)
IS_DRAFT=$(printf '%s' "$_REVIEWS_JSON" | jq -r '.isDraft')
APPROVED=$(printf '%s' "$_REVIEWS_JSON" \
  | jq -r --arg owner "$OWNER" \
    '[.reviews[] | select(.author.login == $owner and .state == "APPROVED")] | length > 0')

if [[ "$APPROVED" == "true" ]]; then
  MERGE_STATE=$(gh pr view "$PR" --repo "$REPO" --json mergeStateStatus --jq .mergeStateStatus)
  if [[ "$MERGE_STATE" == "BLOCKED" ]]; then
    log "PR #$PR approved but merge blocked (CI pending) — enabling auto-merge"
    gh pr merge "$PR" --repo "$REPO" --squash --auto 2>/dev/null || true
    break
  fi
  log "PR #$PR approved by $OWNER — merging"
  gh pr merge "$PR" --repo "$REPO" --squash 2>/dev/null \
    || gh pr merge "$PR" --repo "$REPO" --squash --auto
  log "PR #$PR merged — closing issue #$CURRENT_ISSUE"
  gh issue close "$CURRENT_ISSUE" --repo "$REPO" 2>/dev/null || true
  echo '[]' > "$FIDO_DIR/tasks.json" 2>/dev/null || true
  rm -f "$STATE_FILE"
  git checkout "$DEFAULT_BRANCH"
  git pull "$FORK_REMOTE" "$DEFAULT_BRANCH" --ff-only 2>/dev/null || true
  git branch -d "$SLUG" 2>/dev/null || true
  continue
fi

LATEST_REVIEW_STATE=$(printf '%s' "$_REVIEWS_JSON" \
  | jq -r --arg owner "$OWNER" \
    '[.reviews[] | select(.author.login == $owner)] | last | .state // "NONE"')

if [[ "$LATEST_REVIEW_STATE" == "CHANGES_REQUESTED" ]]; then
  log "PR #$PR: changes requested by $OWNER — all addressed, re-requesting review"
  gh pr edit "$PR" --repo "$REPO" --add-reviewer "$OWNER"
  break
fi

if [[ "$IS_DRAFT" == "true" ]]; then
  _COMPLETED=$(bash "$SCRIPT_DIR/task-cli.sh" "$WORK_DIR" list 2>/dev/null \
    | jq -r '[.[] | select(.status == "completed")] | length' 2>/dev/null || echo "0")
  if [[ "$_COMPLETED" == "0" ]]; then
    log "PR #$PR: no tasks completed — not promoting (setup may have failed)"
    break
  fi
  log "PR #$PR work complete — marking ready, requesting review from $OWNER"
  gh pr ready "$PR" --repo "$REPO"
  gh pr edit "$PR" --repo "$REPO" --add-reviewer "$OWNER"
  continue  # re-check — approval may already exist
fi

# ── No work ────────────────────────────────────────────────────────────────
log "no work"
break

done # end main loop
exit 0
