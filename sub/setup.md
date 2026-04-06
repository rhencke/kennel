A fresh git branch has been created from upstream, a sentinel commit pushed, and a draft PR opened. Your job is to plan the work and sync the plan to the PR description. All context is in the Context section above.

## Steps

### 1. Read conventions
Check for CLAUDE.md files. Note the test command, commit discipline, and any other requirements.

### 2. Plan
Break the request into the smallest meaningful tasks — one task per logical commit, ordered so each builds on the previous.

For each task, write it to the shared task file AND call TaskCreate:
```bash
bash /home/rhencke/workspace/kennel/task-cli.sh <work_dir> add "<task title>"
```
Where `<work_dir>` is from the Context section.

### 3. Sync the work queue
Run sync-tasks to update the PR body from the task file:
```bash
bash /home/rhencke/workspace/kennel/sync-tasks.sh <work_dir>
```

## Done when
Tasks written to task file, synced to PR body.

**Verify**: the PR body should have a `<!-- WORK_QUEUE_START -->` block with `- [ ]` items and end with `Fixes #N`.

**Stop immediately. Do not implement any tasks. Implementation is handled by subsequent invocations.**

## Constraints
- **Never** mark the PR as ready for review (`gh pr ready`). It must stay draft. That is the user's decision.
- **Never** rebase, amend, or force-push. New commits only.
