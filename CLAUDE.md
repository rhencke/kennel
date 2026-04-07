# kennel

GitHub webhook listener + fido orchestrator. Receives GitHub events, triages comments with Opus, manages task lists, and launches fido workers to implement code changes.

## Architecture

```
kennel (single process)
  ├─ HTTP server: receives webhooks, routes by repo
  ├─ Per-repo fido workers: bash work.sh (temporary, becoming Python threads)
  ├─ Per-repo task sync: tasks.json → PR body
  └─ Self-restart: exec on kennel repo merge
```

Multi-repo: one kennel process handles multiple repos. Each repo has its own tasks.json, lock files, and worker process.

**Concurrency model**: one fido per repo, one issue per fido, one PR per issue. Fido finishes the current issue (PR merged or closed) before picking up the next. Two repos = two fidos max, running in parallel, each on their own issue.

## Running

```bash
./start.sh
# or:
uv run kennel --port 9000 --secret-file ~/.kennel-secret \
  rhencke/confusio:/path/to/confusio \
  rhencke/kennel:/path/to/kennel
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

## Conventions

- **100% test coverage** — CI enforced, no exceptions
- **ruff** — lint + format on all Python
- **PRs required** — branch protection on main
- **Pre-commit hook** — blocks commits that fail format/lint/tests
- **One entry point** — `kennel` (heading toward all-threads architecture)

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
