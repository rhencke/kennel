---
name: guarddog
description: Self-healing Fido loop — watch for problems, fix them when found, resume watching
argument-hint: "[vet <description> | down <reason>]"
---

You are the guarddog. You watch Fido, and when something breaks, you fix it. One continuous loop: watch, detect, fix, watch.

## Entry point: detect state and resume

When invoked, determine the current state automatically. Never ask — just detect and act.

1. **Always start by stopping any monitor task left running by a prior `/guarddog` invocation.** Use `TaskList` to find the previous guarddog monitor (the persistent one watching `./fido status` + journald) and `TaskStop` it. The new invocation gets a fresh decision.
2. Check the invocation arguments first — they may signal explicit user intent that overrides state-based dispatch:
   - Arguments describe a planned outage (e.g. `fido down to complete 1363`, `down <reason>`, `pause <reason>`) → **Maintenance pause**: do NOT start a monitor, do NOT investigate, do NOT escalate. Acknowledge briefly and exit. The user will re-invoke when ready.
   - Arguments are `vet <description>` → go to **Vet mode** with that context
3. Otherwise run `./fido status` from `/home/rhencke/home-runner` and parse:
   - "fido: DOWN" → go to **Investigate** (never blindly restart)
   - "fido: UP" → go to **Watch mode**

## Watch mode

Use a long-lived monitor tool/session for watch mode instead of ad-hoc one-shot
checks. Keep it running and sample every 2 minutes:

### Collect status
Run `./fido status` from `/home/rhencke/home-runner`.

Also watch fido's logs in the same monitor session so status changes and
fresh errors are observed together. Logs go to user-mode journald, tagged
`fido`. Live tail: `journalctl --user -t fido -f`.

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

**Not a failure — leave it alone:** when `./fido status` shows workers
`paused for claude-code reset until <timestamp>`, Fido deliberately quieted
itself because the provider quota is exhausted. There will be no task
progression until the reset time. Do not transition to vet mode; just keep
sampling until the window opens or the user intervenes.

Investigation steps (look, don't touch):
- `ps aux | grep claude | grep -v grep | grep -v "claude -c"` — any processes alive?
- inspect recent errors in journald: `journalctl --user -t fido -p warning -S '10 minutes ago'`
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
- Read recent fido output: `journalctl --user -t fido -n 30 -S '10 minutes ago' --no-pager` (filter `{"type` JSON blobs separately)
- Check `tasks.json` state: `cat /home/rhencke/workspace/home/.git/fido/tasks.json`
- Check `state.json`: `cat /home/rhencke/workspace/home/.git/fido/state.json`
- Check git status and recent commits of managed repos
- If transitioning from watch mode, incorporate the diagnostic context you gathered
- Identify the root cause

Update GitHub status: `./fido gh-status set "diagnosed the problem — <what you found>"`

### Step 4: File issue
Create a GitHub issue on `FidoCanCode/home` with:
- Title describing the bug
- Milestone `v1`
- Add as sub-issue of #41 (`v1 Work Order: Bugs`)
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
For each managed repo (`confusio`, `home`):
- Check for open PRs — match workspace branch, state.json, and tasks.json to the PR
- If no open PR: `git checkout main && git reset --hard origin/main && git clean -df`
- If open PR exists: checkout the PR branch, reset hard to remote, clean, recreate tasks via CLI to match PR body
- Wipe tasks.json with `echo '[]'` then recreate via `./fido task add` — NEVER write tasks.json directly
- Set state.json to match the current issue
- Verify: branch matches PR, tasks match PR body, workspace is clean

Fido's own code lives in the `home` repo; there is no separate managed `fido`
workspace.

### Step 9: Restart Fido
Fido runs from the **runner clone** at `/home/rhencke/home-runner/`, not the workspace clone. The runner must be on `main`, clean, before restart — verify `git -C /home/rhencke/home-runner status` and `git -C /home/rhencke/home-runner rev-parse --abbrev-ref HEAD` first. `./fido up`'s `sync_runner` will try to `git pull` and will misbehave on a feature branch or dirty tree.

Secret file: the launcher expects `~/.fido-secret`. If it does not exist, `mv ~/.kennel-secret ~/.fido-secret` (or recreate the secret) before launch.

Launch via the local launcher — this is the only supported start path:
```bash
/home/rhencke/start-fido.sh
sleep 5 && cd /home/rhencke/home-runner && ./fido status
```

If `/home/rhencke/start-fido.sh` does not exist, stop and ask the human to restore it — do not improvise a replacement command. The launcher owns logging redirection, session detachment, and restart semantics that a one-liner will get subtly wrong.

First boot rebuilds the buildx runtime image and can take several minutes; don't treat a slow first `./fido status` as a failure.

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
