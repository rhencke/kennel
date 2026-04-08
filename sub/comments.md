Unresolved PR review threads need responses. All context (PR, repo, thread JSON) is in the Context section above.

The thread JSON contains only threads that are unresolved and don't already have a final reply — no further filtering needed.

## First: follow up on open ASK tasks

Check tasks.json for ASK tasks:
```bash
uv run --project /home/rhencke/workspace/kennel kennel task <work_dir> list | jq '[.[] | select(.status == "pending" and (.title | startswith("ASK:")))]'
```

For each ASK task with a thread, check if the human has replied:
- **Still unclear** → post another follow-up, leave the task open.
- **Clear enough to act** → create a new task via kennel task, mark the ASK task complete:
  ```bash
  uv run --project /home/rhencke/workspace/kennel kennel task <work_dir> add "<new task title>"
  uv run --project /home/rhencke/workspace/kennel kennel task <work_dir> complete "ASK: <original title>"
  ```
- **Human has NOT replied since last message** → skip.

## Then: process remaining unresolved threads

For each thread in the JSON that does not already have a task:

**Bot threads** (`is_bot: true`):
- **DO** — worth implementing:
  ```bash
  uv run --project /home/rhencke/workspace/kennel kennel task <work_dir> add "PR comment: <short summary>" --comment-id <first_db_id> --repo <repo> --pr <pr>
  ```
  Post acknowledgement reply.
- **DEFER** — useful but out of scope:
  ```bash
  gh issue create --repo <repo> --title "<suggestion>" --body "<context + thread URL>"
  ```
  Post reply noting deferral to the new issue, then resolve the thread.
- **DUMP** — not applicable:
  Post a polite decline reply, then resolve the thread.

**Human threads** (`is_bot: false`):
- **ACT** — you know what to do:
  ```bash
  uv run --project /home/rhencke/workspace/kennel kennel task <work_dir> add "PR comment: <short summary>" --comment-id <first_db_id> --repo <repo> --pr <pr>
  ```
  Post acknowledgement reply.
  **Do NOT implement here** — only queue the work.
- **ASK** — unclear what to do:
  Post a focused clarifying question.
  ```bash
  uv run --project /home/rhencke/workspace/kennel kennel task <work_dir> add "ASK: <short summary>" --comment-id <first_db_id> --repo <repo> --pr <pr>
  ```
- **ANSWER** — a question, not a code change request:
  Post a direct answer. Do NOT resolve. Do NOT create a task.

## Finally: trigger sync
```bash
bash /home/rhencke/workspace/kennel/sync-tasks.sh <work_dir>
```

Do NOT use TaskCreate, TaskUpdate, TodoWrite, or any other task tools. Only `kennel task`.
Do NOT edit the PR body directly. `sync-tasks.sh` owns the PR body work queue.

## Done when
Every thread in the JSON has been responded to or has an open task.

## Constraints
- **Never** mark the PR as ready for review (`gh pr ready`). It must stay draft.
- **Never** call any `/reviews` endpoint (read or write). Use only `pulls/{pr}/comments` with `in_reply_to=<first_db_id>` for thread replies.
- **Never** use TaskCreate, TaskUpdate, TaskList, TodoWrite, or TodoRead. Only `kennel task`.
- **Never** edit the PR body directly.
