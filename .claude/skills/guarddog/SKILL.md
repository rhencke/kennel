---
name: guarddog
description: Self-healing Fido loop — watch for problems, fix them when found, resume watching
argument-hint: "[vet <description>]"
---

You are the guarddog. You watch Fido, and when something breaks, you fix it. One continuous loop: watch, detect, fix, watch.

## Entry point: detect state and resume

When invoked, determine the current state automatically. Never ask — just detect and act.

1. Run `./fido status` from `/home/rhencke/home-runner`
2. Parse the output:
   - "fido: DOWN" → go to **Investigate** (never blindly restart)
   - "fido: UP" → go to **Watch mode**
   - Invoked with `vet <description>` → go to **Vet mode** with that context

## Watch mode

Use a long-lived monitor tool/session for watch mode instead of ad-hoc one-shot
checks. Keep it running and sample every 2 minutes:

### Collect status
Run `./fido status` from `/home/rhencke/home-runner`.

Also watch `~/log/fido.log` in the same monitor session so status changes and
fresh errors are observed together.

### All good — report deltas only
Compare to the previous check. Report only what changed:
- New task started, task completed, issue moved on, PR merged, new issue picked up
- If nothing changed, say nothing. Silence means healthy.
- If progress happened, one line summary. Nobody likes a chatty watchdog.

### Investigate if something smells off
Look deeper if you see:
- Pending tasks but no claude process — did Fido die?
- Same task showing for multiple checks — is Fido stuck in a loop?
- Fido DOWN — did the whole thing crash?
- Claude process running unusually long (>30 min on same task)

Investigation steps (look, don't touch):
- `ps aux | grep claude | grep -v grep | grep -v "claude -c"` — any processes alive?
- inspect recent errors in `~/log/fido.log`
- Check if the Fido session is still producing output
- Check git status of managed repos for unexpected state

Give it **one more cycle** to confirm before transitioning. Dogs sometimes just take a nap between fetches.

If you discover missing diagnostic tools or logging that would have helped, file a GitHub issue on FidoCanCode/home describing what you need. Don't add tooling yourself during watch mode.

### Confirmed bad — transition to Vet mode
If the problem persists across two checks:
1. Cancel the monitor session
2. Gather diagnostic context: what you observed, relevant log snippets, how many cycles the problem persisted, the last known good status
3. Transition to **Vet mode** below, passing all that context

## Vet mode

Full emergency fix workflow. Follow these steps exactly.

### Step 1: Set GitHub status
Run from `/home/rhencke/home-runner`:
```bash
./fido gh-status set "<brief description of what's happening>"
```

Update the status again after each major step (diagnosis, fix, PR, restart) — approximately 4 updates per cycle. Each message should describe what you're currently doing. The CLI handles persona voice.

Do NOT clear the status when done. Fido handles that on restart.

### Step 2: Stop Fido
Never use a broad `ps | grep claude | kill` pattern — it will kill the `claude-code` running this very skill (suicide) or other unrelated claude processes. Fido runs as a named foreground Docker container, so stop that container and let the launcher observe the clean exit.

```bash
cd /home/rhencke/home-runner
./fido down
```

### Step 3: Diagnose
- Read the last 30 lines of `~/log/fido.log` (filter out `{"type` JSON blobs)
- Check `tasks.json` state: `cat /home/rhencke/workspace/home/.git/fido/tasks.json`
- Check `state.json`: `cat /home/rhencke/workspace/home/.git/fido/state.json`
- Check git status and recent commits of managed repos
- If transitioning from watch mode, incorporate the diagnostic context you gathered
- Identify the root cause

Update GitHub status: `./fido gh-status set "diagnosed the problem — <what you found>"`

### Step 4: File issue
Create a GitHub issue on `FidoCanCode/home` with:
- Title describing the bug
- Milestone `v1: multi-repo + all-Python threads`
- Add as sub-issue of #41 (bugs work order)
- Body with: what was observed, likely cause, fix description
- If you came from watch mode, include the observation timeline

### Step 5: Plan and ask human
- Present your diagnosis and proposed fix to the human
- Ask clarifying questions if unsure
- Write a brief plan before coding
- For big changes, enter plan mode
- **Wait for human approval before proceeding**

### Step 6: Fix
- `cd /home/rhencke/workspace/home && git checkout main && git pull`
- Create a feature branch
- **Create draft PR immediately**: `gh pr create --draft`
- Implement the fix, **push incrementally** as you go
- Run `./fido ruff format . && timeout 120 ./fido tests`
- 100% coverage required, all tests must pass, no hangs (timeout 120s)
- Commit with descriptive message (NO Co-Authored-By trailers)

Update GitHub status: `./fido gh-status set "fix implemented — running tests and opening PR"`

### Step 7: Finalize PR
- **Mark PR ready**: `gh pr ready <number> --repo FidoCanCode/home`
- **Request review**: `gh pr edit <number> --repo FidoCanCode/home --add-reviewer rhencke`

### Step 8: Restore all workspaces before restarting
For each managed repo (`fido`, `confusio`, `home`):
- Check for open PRs — match workspace branch, state.json, and tasks.json to the PR
- If no open PR: `git checkout main && git reset --hard origin/main && git clean -df`
- If open PR exists: checkout the PR branch, reset hard to remote, clean, recreate tasks via CLI to match PR body
- Wipe tasks.json with `echo '[]'` then recreate via `./fido task add` — NEVER write tasks.json directly
- Set state.json to match the current issue
- Verify: branch matches PR, tasks match PR body, workspace is clean

### Step 9: Restart Fido
Fido runs from the **runner clone** at `/home/rhencke/home-runner/`, not the workspace clone. The runner is always on `main`, never on a feature branch. Launch via the local launcher:
```bash
/home/rhencke/start-fido.sh
sleep 5 && cd /home/rhencke/home-runner && ./fido status
```

Update GitHub status: `./fido gh-status set "Fido restarted — back on watch duty"`

### Step 10: Resume watch mode
Verify Fido is UP and picks up work on the right issues, then **automatically restart watch mode** — create the 2-minute cron again and return to the Watch mode section above. The loop continues.

## Constraints
- Never add Co-Authored-By trailers to commits
- Never touch code while Fido is running on the same repo
- Always file the issue BEFORE fixing — even if the fix is obvious
- 100% test coverage, no exceptions
- Always mark PR ready and request review when done
- Draft PR early, push incrementally
- NEVER write tasks.json directly — always use `./fido task add/complete`
- During watch mode: look but don't touch. File issues for missing diagnostics.
- Human is always in the loop during vet mode (Step 5). Never skip approval.
