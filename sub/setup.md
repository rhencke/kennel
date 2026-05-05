A fresh git branch has been created. Your job is to PLAN the work by emitting the desired task list as a JSON sentinel. You are NOT implementing anything — just planning. The PR does not exist yet — it will be created after you finish. All context is in the Context section above.

## Steps

### 1. Read conventions
Check for CLAUDE.md files. Note the test command, commit discipline, and any other requirements.

### 2. Plan
Break the request into the smallest meaningful tasks — one task per logical commit, ordered so each builds on the previous.

**Task titles must be short one-line summaries** — imperative verb-first, under 80 characters. Like `Add Dependabot routes and default handlers` or `Gitea: dependency-graph endpoints and tests`. NOT multi-paragraph specs with file lists, endpoint tables, or instructions. The title appears in `fido status`, PR work queues, and log lines — it needs to fit on one line. Detailed implementation guidance, when needed, belongs in the optional `description` field, not the title.

### 3. Draft the PR description
You also draft the PR description body — what reviewers read above the auto-generated work queue. Keep it scoped to *what* and *why*, not *how* (implementation lives in commits and tasks). A typical body has a `## Summary` section with a few bullet points and may include `## Why` or `## Test plan` where useful.

**Do not write `Fixes #<issue>` yourself** — the harness already knows the issue number and appends the canonical trailer. If you write one anyway the harness strips it and re-appends, so the body always ends with exactly one `Fixes #<n>.`.

### 4. Emit the setup_outcome sentinel
Your final non-empty output line **must** be exactly one JSON object that declares the planned task list and the PR description. The harness reads it and CRUDs the task store and PR body on your behalf — you never write to `tasks.json`, never edit the PR body directly, and never run `./fido task` yourself.

Choose exactly one outcome:

- **`tasks-planned`** — you have at least one task to queue. The harness creates one `spec`-type pending task per entry, in the order given. `description` is optional. `pr_description` is the markdown body the harness will write above the work queue (skip the `---` divider — the harness inserts it; skip `Fixes #<issue>` — the harness appends it).
  ```json
  {"setup_outcome": "tasks-planned", "pr_description": "## Summary\n\n- bullet 1\n- bullet 2", "tasks": [{"title": "First task title"}, {"title": "Second task title", "description": "Optional implementation hint"}]}
  ```
- **`no-tasks-needed`** — the issue's work is already covered by the current branch state, or is a no-op. The harness will mark the PR ready and post an explanation comment. `pr_description` is optional here.
  ```json
  {"setup_outcome": "no-tasks-needed", "reason": "Already implemented in commit abc1234"}
  ```

The sentinel must be the literal last non-empty line of your response — nothing after it. Do not wrap it in a code fence or markdown block. The single object must be valid JSON on one line. Embed newlines inside `pr_description` as `\n` escapes.

## Done when
The setup_outcome sentinel has been emitted.

**Stop immediately after emitting the sentinel. Do not implement any tasks. Implementation is handled by subsequent invocations.**

## Constraints
- **You are a planner, not an implementer.** Read the code to understand it, then emit the sentinel. Do not change the code.
- **Never** commit code. No `git commit`, no `git add`. The harness owns commits.
- **Never** edit source files. No `Edit`, no `Write` to any file.
- **Never** push to any branch. No `git push`.
- **Never** mark the PR as ready for review (`gh pr ready`). The harness handles this.
- **Never** rebase, amend, or force-push.
- **Never** use TaskCreate, TaskUpdate, TaskList, TodoWrite, TodoRead, or `./fido task`.
- **Never** edit any PR body. The Fido server handles that.
- **Never** write to tasks.json directly. The harness owns the task store.
