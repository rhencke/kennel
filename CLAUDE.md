# kennel

GitHub webhook listener + fido orchestrator. Receives GitHub events, triages comments with Opus, manages task lists, and launches fido workers to implement code changes.

## Architecture

```
kennel (single process, runs from /home/rhencke/kennel-runner/)
  ├─ HTTP server: receives webhooks, routes by repo
  ├─ Per-repo fido workers: bash work.sh (temporary, becoming Python threads)
  ├─ Per-repo task sync: tasks.json → PR body
  └─ Self-restart: git pull runner clone, exec uv run kennel
```

Multi-repo: one kennel process handles multiple repos. Each repo has its own tasks.json, lock files, and worker process.

**Concurrency model**: one fido per repo, one issue per fido, one PR per issue. Fido finishes the current issue (PR merged or closed) before picking up the next. Two repos = two fidos max, running in parallel, each on their own issue.

## Runner vs workspace clones

Kennel runs from a dedicated **runner clone** at `/home/rhencke/kennel-runner/`, separate from the **workspace clone** at `/home/rhencke/workspace/kennel/`.

- **Runner clone** — always on `main`, never dirty, never has feature branches. Kennel imports its Python code from here. Self-restart does `git pull` here.
- **Workspace clone** — where fido edits source files, commits, and pushes feature branches. Never used to run the server.

Launching: `/home/rhencke/start-kennel.sh` (local, outside git) execs `uv run kennel ...` from the runner clone. The `start.sh` files inside both clones are for reference/local dev only.

Self-restart logic: `_self_restart` in `server.py` derives the runner clone from `Path(__file__).resolve().parents[1]`, checks the git remote matches the merged PR's repo, pulls with exponential backoff (10s → 30s → 60s, 10-minute budget), then `os.execvp("uv", ["uv", "run", "kennel", *sys.argv[1:]])` with cwd set to the runner clone. This replaces the previous in-place restart that ran `git reset --hard` in the workspace clone and clobbered fido's in-progress work.

## Running

```bash
/home/rhencke/start-kennel.sh
# or directly from the runner clone:
cd /home/rhencke/kennel-runner && uv run kennel --port 9000 --secret-file ~/.kennel-secret \
  rhencke/confusio:/home/rhencke/workspace/confusio \
  rhencke/kennel:/home/rhencke/workspace/kennel
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

### Bash scripts (temporary, being rewritten to Python)

| Script | Purpose |
|--------|---------|
| `work.sh` | Main fido worker loop |
| `sync-tasks.sh` | Sync tasks.json → PR body |
| `task-cli.sh` | Bash task CLI (add/complete/list) — superseded by `kennel task`; obsolete once shell scripts are removed |
| `watchdog.sh` | Kill stale workers, restart |
| `start.sh` | Env setup + exec kennel |

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

`_pick_next_task` selects by type: `ci` first, `thread` second, `spec` last.

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
