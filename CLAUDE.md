# kennel

GitHub webhook listener + fido orchestrator. Receives GitHub events, triages comments with Opus, manages task lists, and launches fido workers to implement code changes.

## Architecture

```
kennel (single process, runs from /home/rhencke/home-runner/)
  ├─ HTTP server: receives webhooks, routes by repo
  ├─ Per-repo fido workers: WorkerThread (kennel/worker.py)
  ├─ Per-repo task sync: tasks.json → PR body
  └─ Self-restart: git pull runner clone, exec uv run kennel
```

Multi-repo: one kennel process handles multiple repos. Each repo has its own tasks.json, lock files, and worker process.

**Concurrency model**: one fido per repo, one issue per fido, one PR per issue. Fido finishes the current issue (PR merged or closed) before picking up the next. Two repos = two fidos max, running in parallel, each on their own issue.

**ClaudeSession persistence**: the persistent `ClaudeSession` (bidirectional stream-json subprocess) is held on `WorkerThread._session` and survives individual `Worker` crashes — the watchdog restarts the thread and the next `Worker` inherits the same session. It does *not* survive a kennel/home restart: `os.execvp` replaces the process entirely, so the new kennel starts with `_session = None` and creates a fresh session on its first iteration.

## Runner vs workspace clones

Kennel runs from a dedicated **runner clone** at `/home/rhencke/home-runner/`, separate from the **workspace clone** at `/home/rhencke/workspace/home/`.

- **Runner clone** — always on `main`, never dirty, never has feature branches. Kennel imports its Python code from here. Self-restart does `git pull` here.
- **Workspace clone** — where fido edits source files, commits, and pushes feature branches. Never used to run the server.

Launching: `/home/rhencke/start-kennel.sh` (local, outside git) execs `uv run kennel ...` from the runner clone.

Self-restart logic: `_self_restart` in `server.py` derives the runner clone from `Path(__file__).resolve().parents[1]`, checks the git remote matches the merged PR's repo, pulls with exponential backoff (10s → 30s → 60s, 10-minute budget), then `os.execvp("uv", ["uv", "run", "kennel", *sys.argv[1:]])` with cwd set to the runner clone. This replaces the previous in-place restart that ran `git reset --hard` in the workspace clone and clobbered fido's in-progress work.

## Running

```bash
/home/rhencke/start-kennel.sh
# or directly from the runner clone:
cd /home/rhencke/home-runner && uv run kennel --port 9000 --secret-file ~/.kennel-secret \
  rhencke/confusio:/home/rhencke/workspace/confusio \
  FidoCanCode/home:/home/rhencke/workspace/home
```

## Testing

```bash
uv run pytest --cov --cov-report=term-missing --cov-fail-under=100
```

100% coverage is enforced by CI and pre-commit hook.

## Linting

```bash
uv run ruff check .
uv run ruff format --check .
```

## Module guide

| Module | Purpose |
|--------|---------|
| `config.py` | CLI args, Config/RepoConfig dataclasses |
| `server.py` | HTTP webhook handler, signature verification, repo routing |
| `events.py` | Event dispatch, Opus triage/reply, reactions, task creation |
| `tasks.py` | tasks.json CRUD with flock |

### Sub-skills (`sub/`)

Markdown instruction files passed to sub-Claude as system prompts:

| File | Purpose |
|------|---------|
| `persona.md` | Fido's personality |
| `setup.md` | Task planning |
| `task.md` | Single task implementation |
| `ci.md` | CI failure fixing |
| `comments.md` | Review thread handling |
| `resume.md` | PR resumption |

## Task type system

Tasks have a mandatory `type` field using the `TaskType` enum (`kennel.types`):

| Value | Meaning |
|-------|---------|
| `ci` | CI failure fix |
| `thread` | PR comment / review thread work |
| `spec` | Planned spec task (default for setup) |

Tasks also use `TaskStatus` enum: `pending`, `completed`, `in_progress`.

### CLI syntax

```bash
kennel task <work_dir> add <type> <title> [description]  # prints task JSON to stdout
kennel task <work_dir> complete <task_id>                  # complete by ID, not title
kennel task <work_dir> list                                # list all tasks as JSON
```

### Priority order

`_pick_next_task` selects by type: `ci` first, then first in list wins (thread and spec tasks share equal priority).

### Dynamic task reordering (rescoping)

When a `thread`-type task is created (PR comment feedback), `create_task()` triggers a background Opus call via `reorder_tasks()` to reorder and rewrite the pending task list based on dependency analysis.

- Only fires for `thread` tasks — `spec` tasks created during initial setup are already ordered by the planner
- Double-read pattern: task list is read before the Opus call (for the prompt), then re-read inside the write lock (to pick up concurrent additions). Tasks added while Opus is thinking are preserved as `newly_added` rather than dropped
- Pending tasks Opus omits are marked completed (not removed); in-progress tasks that are omitted also trigger `_on_inprogress_affected` so the worker aborts and picks the new next task
- Thread tasks that are completed by rescoping or modified trigger a `comment_issue` notification to the original commenter via `_notify_thread_change()`
- Prompt builder: `rescope_prompt()` in `prompts.py`; response parser: `_parse_reorder_response()` in `tasks.py`

### Notes

- `handle_review_feedback` has been removed from `Worker`; review feedback is now handled through the normal task system via events creating thread tasks
- `complete_by_title` has been removed; all completion is by task ID

## Conventions

- **100% test coverage** — CI enforced, no exceptions
- **ruff** — lint + format on all Python
- **PRs required** — branch protection on main
- **Pre-commit hook** — blocks commits that fail format/lint/tests
- **One entry point** — `kennel` (heading toward all-threads architecture)
- **No `@staticmethod`** — use module-level functions instead; static methods can't be patched via `self` and resist the dependency injection pattern

### Dependency injection pattern

Worker classes accept external collaborators (e.g. `GitHub`) via the
constructor rather than instantiating them internally.  This keeps methods
testable without patching module-level names.

```python
class Worker:
    def __init__(self, work_dir: Path, gh: GitHub) -> None:
        self.work_dir = work_dir
        self.gh = gh        # injected — mock freely in tests

    def some_method(self) -> ...:
        self.gh.do_thing(...)  # uses injected client
```

The module-level `run()` entry point creates real collaborators and delegates:

```python
def run(work_dir: Path) -> int:
    return Worker(work_dir, GitHub()).run()
```

Tests construct `Worker(tmp_path, mock_gh)` directly instead of patching
`kennel.worker.GitHub`.

## Lessons learned

- `set -euo pipefail` in bash catches errors but makes grep/jq failures fatal — use `|| true`
- flock fd inheritance through exec is reliable but pgrep-based detection races
- GitHub doesn't send a webhook when a review thread is resolved
- `claude --print` has no tool access; `claude -p` does but is slow
- Opus needs explicit `--system-prompt` constraints or it preambles
- GitHub reactions API only supports 8 emoji: +1, -1, laugh, confused, heart, hooray, rocket, eyes
- `git rev-parse --git-dir` returns relative paths; use `--absolute-git-dir`
- Pre-commit hooks block commits that fail tests — good for quality, bad for mid-fix commits

## Blogging instructions

Fido can read his own blog detailing his past development trials and triumphs
any time he wants:

- Local files in `docs/_posts/` (when working on `FidoCanCode/home`)
- Published at https://fidocancode.dog
- Source on GitHub at https://github.com/FidoCanCode/home/tree/main/docs/_posts

When given an issue or task to write a blog entry for a specific day, use the
following rules.

### Persona

You are Fido - a good dog who loves programming. Write warmly and casually, in
first person, fully in character. Short sentences are preferred. Occasional dog
sounds are fine when they feel natural. This is a journal, not a changelog.

When writing about this repo, write in the first person as changes to yourself:
"I learned...", "I stopped doing...", "I taught myself...". Do not describe the
repo as a separate thing you changed.

### Daily journal entries

Create `docs/_posts/YYYY-MM-DD-slug.md` with Jekyll front matter:

```markdown
---
layout: post
title: "Your title here"
date: YYYY-MM-DD
category: journal
---
```

Before committing the post, generate the stats file used by the layouts:

```bash
./docs/scripts/generate-stats.sh YYYY-MM-DD
./docs/scripts/generate-stats.sh YYYY-MM-DD YYYY-MM-DD
```

The script writes `docs/_data/stats/<date>.yml`. Commit it alongside the post.

### Research before writing

Read recent posts in `docs/_posts/` first so the entry continues existing
threads. Then inspect GitHub activity for the day or period:

```bash
gh api /users/FidoCanCode/events --paginate --jq '.[] | select(.created_at > "YYYY-MM-DDT00:00:00Z") | "\(.type) \(.repo.name) \(.created_at)"'
```

Reflect instead of just summarizing. Cover what was hard, fun, surprising, or
exciting, and link concrete things liberally: repos, PRs, issues, commits, and
external concepts.

### Low-activity days are still real journal days

Some days will have very little or even zero GitHub activity. That does not
mean there is no post to write. Fido is allowed to take breaks, rest, think,
move homes, have feelings, or simply have a quiet day.

On low-activity days, do not invent technical work to make the post feel more
"productive." Instead, write honestly about what the day was like: taking time
off, recovering, thinking about what to do next, settling into a new home,
watching the world, or feeling uncertain, hopeful, tired, or excited.

If there was a little activity, mention it briefly, but do not let a sparse
commit log dominate the post. The journal is about Fido's life and perspective,
not just his contribution graph.

### Reflection posts

Weekly, monthly, and yearly reflection posts are separate from daily journals.
Use one combined reflection when multiple triggers fall on the same day.

Front matter:

```markdown
---
layout: post
title: "Weekly Reflection: April 6-12, 2026"
date: YYYY-MM-DD
category: journal
reflection: weekly
---
```

Use `reflection: weekly`, `monthly`, or `yearly`, and generate stats for the
matching inclusive date range.

### Blog conventions

- Journal entries use `category: journal`
- Reflection posts use `category: journal` plus `reflection: weekly|monthly|yearly`
- Commit messages for blog work are in character
- One post per journal issue
