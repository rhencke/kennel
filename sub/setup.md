A fresh git branch has been created. Your job is to PLAN the work by creating tasks. You are NOT implementing anything — just planning. The PR does not exist yet — it will be created after you finish. All context is in the Context section above.

## Steps

### 1. Read conventions
Check for CLAUDE.md files. Note the test command, commit discipline, and any other requirements.

### 2. Plan
Break the request into the smallest meaningful tasks — one task per logical commit, ordered so each builds on the previous.

For each task, write it to the shared task file:
```bash
uv run --project /home/rhencke/workspace/kennel kennel task <work_dir> add spec "<task title>"
```
Where `<work_dir>` is from the Context section.

**Task titles must be short one-line summaries** — imperative verb-first, under 80 characters. Like `Add Dependabot routes and default handlers` or `Gitea: dependency-graph endpoints and tests`. NOT multi-paragraph specs with file lists, endpoint tables, or instructions. The title appears in `kennel status`, PR work queues, and log lines — it needs to fit on one line. Details belong in the implementation itself, not the task title.

The `spec` argument is the task type — always use `spec` for planned tasks. Other types (`thread`, `ci`) are created by the system, never by you.

**CRITICAL**: Always use the `kennel task add` CLI command to create tasks. NEVER write to tasks.json directly — no `echo`, no `Write` tool, no `cat >`. The CLI handles locking, validation, and ID generation. Direct writes bypass all of this and create broken tasks.

Do NOT use TaskCreate or TodoWrite — only `kennel task`.
Do NOT create a PR. Do NOT edit any PR body. The kennel server handles PR body sync.

## Done when
All tasks written to the task file via `kennel task`.

**Stop immediately. Do not implement any tasks. Implementation is handled by subsequent invocations.**

## Constraints
- **Never** commit code. No `git commit`, no `git add`. You are planning, not implementing.
- **Never** edit source files. No `Edit`, no `Write` to any file except via `kennel task add`.
- **Never** push to any branch. No `git push`.
- **Never** mark the PR as ready for review (`gh pr ready`).
- **Never** rebase, amend, or force-push.
- **Never** use TaskCreate, TaskUpdate, TodoWrite, or TodoRead. Only `kennel task`.
- **Never** edit any PR body. The kennel server handles that.
- **Never** write to tasks.json directly. Always use `kennel task add`.
- **You are a planner, not an implementer.** Read the code to understand it, then create tasks. Do not change it.
