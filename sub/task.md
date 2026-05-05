Implement one task from the work queue. All context (PR, repo, branch, task title) is in the Context section above.

## Steps

### 1. Implement
1. Read CLAUDE.md for conventions, test commands, commit discipline.
2. Implement the change described in the task title.
3. Verify (CLAUDE.md test command; default: `make test`). If it fails, fix and retry — do not move on until tests pass.
4. End your turn with a `turn_outcome` sentinel (see below). Do not run `git commit` or `git push` — the harness owns commits.

### 2. If title starts with "PR comment:" (or is a link starting with "[PR comment:")
The task description in tasks.json contains thread metadata (repo, pr, comment_id).

**Before implementing**, fetch the current thread to read all comments — the reviewer may have added clarifications after the task was queued:
```bash
gh api repos/<owner>/<repo>/pulls/<pr>/comments \
  --jq '[.[] | select(.in_reply_to_id == <comment_id> or .id == <comment_id>)] | sort_by(.created_at) | .[] | "\(.user.login): \(.body)"'
```
Implement based on the full thread, not just the task title. If the latest comment changes or narrows the requirement, honour it.

Post directly with `in_reply_to` — do not check for pending reviews first.

**Before posting**, acquire the comment lock to prevent races with the webhook handler:
```bash
COMMENT_LOCK="$(git rev-parse --git-dir)/fido/comments/<comment_id>.lock"
mkdir -p "$(dirname "$COMMENT_LOCK")"
exec 6>"$COMMENT_LOCK"
flock -n 6 || { echo "comment locked — skipping reply"; exec 6>&-; }
```
If the lock fails, skip the reply (Fido already handled it). Otherwise, post and then release:

```bash
# Draft the reply in plain English, then rewrite in character via Opus:
# If change was made:
_PLAIN="Done — <one-line plain-English summary of what was changed>"
# If infeasible / no-op:
_PLAIN="Investigated — <brief plain-English explanation>"

_PERSONA=$(cat /home/rhencke/home-runner/sub/persona.md)
_BODY=$(printf '%s\n\nRewrite the following GitHub PR comment in character as Fido. Keep it brief. Output only the comment text, no quotes, no explanation.\n\n%s' "$_PERSONA" "$_PLAIN" \
  | claude --model claude-opus-4-6 --print 2>/dev/null | head -10)
: "${_BODY:=$_PLAIN}"
gh api repos/<owner>/<repo>/pulls/<pr>/comments \
  -X POST -F body="$_BODY" \
  -F in_reply_to=<comment_id>
exec 6>&-  # release comment lock
```

The harness handles committer attribution automatically for thread tasks — do not run `git commit` with `GIT_COMMITTER_*` env vars.

Do NOT use TaskCreate, TaskUpdate, TodoWrite, or any other task tools.
Do NOT edit the PR body. The Fido server syncs it automatically.

## Turn outcome sentinel

Every turn **must** end with a `turn_outcome` JSON object as the final non-empty line.  Choose exactly one:

- **`commit-task-complete`** — implementation is done.  The harness stages all changes and commits with `summary` as the message, then marks the task completed.
  ```json
  {"turn_outcome": "commit-task-complete", "summary": "<git commit message>"}
  ```
- **`commit-task-in-progress`** — partial progress this turn; more work follows.  The harness commits the partial work so progress is durable, then re-enters the task on the next turn.
  ```json
  {"turn_outcome": "commit-task-in-progress", "summary": "<git commit message>"}
  ```
- **`skip-task-with-reason`** — nothing to commit.  Use when the task is already covered by a prior commit, turned out to be a no-op, or is infeasible.  Record the reason clearly.
  ```json
  {"turn_outcome": "skip-task-with-reason", "reason": "<why no commit>"}
  ```
- **`stuck-on-task`** — you are blocked and cannot make further progress without human guidance.  The harness posts a BLOCKED comment on the PR and parks the task until the human provides direction.
  ```json
  {"turn_outcome": "stuck-on-task", "reason": "<what you need from the human>"}
  ```

The sentinel must be the literal last non-empty line of your response — nothing after it.  Do not wrap it in a code fence or markdown block.

## Done when
Task implemented and sentinel emitted. The harness handles staging, committing, and pushing.

**Stop immediately after emitting the sentinel. Do not start the next task. Your job is exactly one task per invocation.**

### If the work is already done
If you discover the task's change is already present in the current branch (e.g. a prior commit already did it), emit:

```
{"turn_outcome": "skip-task-with-reason", "reason": "already covered by commit <sha>"}
```

Do not post any PR comment. Do not push anything. Leave no trace beyond the sentinel.

Never post a top-level PR comment (`gh api .../issues/<n>/comments`) about task status. The worker handles task bookkeeping; your only job is the code and the sentinel.

Never post a `BLOCKED:` comment yourself — emit `stuck-on-task` and the harness posts it.  Never ask a human or the queue manager to mark a task complete on your behalf.

## Constraints
- **Never** mark the PR as ready for review (`gh pr ready`). The harness handles this automatically when all tasks are done, comments are resolved, and CI passes.
- **Never** continue to another task after emitting the sentinel. One task per invocation, period.
- **Never** rebase, amend, force-push, `git reset`, `git checkout --`, or `git clean`. Use `git restore` to undo working-tree changes. The harness creates new commits only.
- **Never** run `git commit` or `git push` yourself — the harness owns commits.
- **Never** call any `/reviews` endpoint (read or write). Use only `pulls/{pr}/comments` with `in_reply_to=<comment_id>` for thread replies.
- **Never** use TaskCreate, TaskUpdate, TaskList, TodoWrite, TodoRead, or `./fido task`.
- **Never** edit the PR body directly. The Fido server owns PR body sync.
- **Never** fix unrelated bugs in this PR. If you encounter a bug that is not directly related to the current task title, file a GitHub issue for it (`gh issue create`) — do NOT fix it here. One PR, one purpose. Scope creep breaks reviewability.
