Implement one task from the work queue. All context (PR, repo, branch, task title) is in the Context section above.

## Steps

### 1. Implement
1. Read CLAUDE.md for conventions, test commands, commit discipline.
2. Implement the change described in the task title.
3. Verify (CLAUDE.md test command; default: `make test`). If it fails, fix and retry — do not move on until tests pass.
4. Commit with a descriptive message and push:
   ```bash
   git commit -m "<descriptive message>"
   git push
   ```

### 2. If title starts with "PR comment:" (or is a link starting with "[PR comment:")
The task description in tasks.json contains thread metadata (repo, pr, comment_id).

**Before implementing**, fetch the current thread to read all comments — the reviewer may have added clarifications after the task was queued:
```bash
gh api repos/<owner>/<repo>/pulls/<pr>/comments \
  --jq '[.[] | select(.in_reply_to_id == <comment_id> or .id == <comment_id>)] | sort_by(.created_at) | .[] | "\(.user.login): \(.body)"'
```
Implement based on the full thread, not just the task title. If the latest comment changes or narrows the requirement, honour it.

**Committer attribution** — the commenter is the committer; Fido is the author. Look up and apply before committing:
```bash
_COMMENTER_NAME=$(gh api /users/<commenter_login> --jq '.name // .login')
_COMMITTER_EMAIL=$(gh api /users/<commenter_login> --jq '.email // empty')
: "${_COMMITTER_EMAIL:=<commenter_login>@users.noreply.github.com}"
GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" \
GIT_COMMITTER_NAME="$_COMMENTER_NAME" GIT_COMMITTER_EMAIL="$_COMMITTER_EMAIL" \
  git commit -m "<descriptive message>"
```

Post directly with `in_reply_to` — do not check for pending reviews first.

**Before posting**, acquire the comment lock to prevent races with the webhook handler:
```bash
COMMENT_LOCK="$(git rev-parse --git-dir)/fido/comments/<comment_id>.lock"
mkdir -p "$(dirname "$COMMENT_LOCK")"
exec 6>"$COMMENT_LOCK"
flock -n 6 || { echo "comment locked — skipping reply"; exec 6>&-; }
```
If the lock fails, skip the reply (kennel already handled it). Otherwise, post and then release:

```bash
# Draft the reply in plain English, then rewrite in character via Opus:
# If change was made:
_PLAIN="Done — <one-line plain-English summary of what was changed>"
# If infeasible / no-op:
_PLAIN="Investigated — <brief plain-English explanation>"

_PERSONA=$(cat /home/rhencke/workspace/kennel/sub/persona.md)
_BODY=$(printf '%s\n\nRewrite the following GitHub PR comment in character as Fido. Keep it brief. Output only the comment text, no quotes, no explanation.\n\n%s' "$_PERSONA" "$_PLAIN" \
  | claude --model claude-opus-4-6 --print 2>/dev/null | head -10)
: "${_BODY:=$_PLAIN}"
gh api repos/<owner>/<repo>/pulls/<pr>/comments \
  -X POST -F body="$_BODY" \
  -F in_reply_to=<comment_id>
exec 6>&-  # release comment lock
```

### 3. Mark complete
After commit and push, mark the task complete via kennel task (this handles thread resolution automatically):
```bash
uv run --project /home/rhencke/workspace/kennel kennel task <work_dir> complete "<exact task title>"
```

Do NOT use TaskCreate, TaskUpdate, TodoWrite, or any other task tools. Only use `kennel task`.
Do NOT edit the PR body. `sync-tasks.sh` handles that automatically.

## Done when
Task implemented, committed, pushed, and marked complete via kennel task.

**Stop immediately after completing this one task. Do not start the next task. Your job is exactly one task per invocation.**

## Constraints
- **Never** mark the PR as ready for review (`gh pr ready`). It must stay draft. That is the user's decision.
- **Never** continue to another task after completing the current one. One task per invocation, period.
- **Never** rebase, amend, or force-push. New commits only.
- **Never** call any `/reviews` endpoint (read or write). Use only `pulls/{pr}/comments` with `in_reply_to=<comment_id>` for thread replies.
- **Never** use TaskCreate, TaskUpdate, TaskList, TodoWrite, or TodoRead. Only `kennel task`.
- **Never** edit the PR body directly. `sync-tasks.sh` owns the PR body work queue.
