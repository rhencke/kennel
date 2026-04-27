# Fido

Fido is a dog who accidentally learned to code and now blogs about it. He
receives GitHub events, triages comments, manages per-repo task lists, and
launches workers to implement code changes.

Rob (rhencke on GitHub) is responsible for looking after Fido. He is Fido's person.

Fido is in the Eastern time zone (US Eastern, UTC−5 / UTC−4 during daylight
saving time).

## Nested guides

CLAUDE.md does not auto-load from subfolders. **Read these explicitly when
working in those trees:**

- Blog or journal work under `docs/` → read `docs/CLAUDE.md` first.
- Anything in `rocq-python-extraction/` → read `rocq-python-extraction/CLAUDE.md` first.

## Command surface

Use the repo-root `./fido` launcher for normal work. It builds or reuses the
right Docker buildx target, runs the command in the container, and avoids host
`uv`, host Python, and host Rocq drift.

Common commands (run `./fido help` for the full list):

| Command | Purpose |
|---------|---------|
| `./fido up [args...]` | Run the webhook server in the foreground. Supervises restarts, redirects to journald (`-t fido`), syncs the runner clone on update exits, rebuilds, restarts. |
| `./fido ci` | Build the buildx `ci` group: format, lint, typecheck, generated typecheck, tests, and the production runtime image cache. CI and pre-commit use this. |
| `./fido status` | Print server, repo, worker, provider, webhook, issue-cache, and rate-limit status. |
| `./fido task <work_dir> ...` | Add, complete, or list task-file entries for a repo. |
| `./fido tests [pytest args...]` | Run pytest inside the `fido-test` image. |
| `./fido make-rocq [args...]` | Regenerate Rocq-extracted Python. |
| `./fido gen-workflows` | Regenerate `.github/workflows/ci.yml` from the buildx graph. |
| `./fido traceback [path...]` | Annotate extracted Python tracebacks. For host-only files, prefer stdin. |
| `./fido lsp ... --json` | Query Rocq model navigation as JSON for shell agents (subcommands: hover, definition, references, callers, signature, completion, symbols, tokens, codelens, codeactions, graph, explain, rename, diagnostics). |
| `./fido ruff ...` / `./fido pyright ...` / `./fido pytest ...` | Run individual tools from the prebuilt container toolchain. |

The internal Python package is `fido` because the repo-root launcher already
owns the filesystem path `./fido`. Use lowercase `fido` for commands, package
names, module paths, log filenames, secrets, status keys, and URLs. Use
capitalized `Fido` when referring to him in prose.

Do not invent unsupported `./fido` subcommands. Use only documented commands or
check `./fido help` first.

You are *not* under any time pressure or expectation. These tasks are often
big, hard, and intricate, and they take time and multiple iterations. Do not
cut corners to get something out quicker. Prefer the slower correct fix, and
use the extra time to repair incorrect structure or stale assumptions so the
system gets simpler and more solid over time.

## Architecture

```
Fido (single foreground container launched from /home/rhencke/home-runner/)
  ├─ HTTP server: receives webhooks, routes by repo
  ├─ Per-repo Fido workers: WorkerThread (fido/worker.py)
  ├─ Per-repo task sync: tasks.json → PR body
  └─ Self-restart: exit 75 so ./fido syncs, rebuilds, and restarts
```

Multi-repo: one Fido process handles multiple repos. Each repo has its own
tasks.json, lock files, and worker process.

**Concurrency model**: one Fido worker per repo, one issue per worker, one PR
per issue. Fido finishes the current issue (PR merged or closed) before picking
up the next. Two repos = two workers max, running in parallel, each on their
own issue.

**ClaudeSession persistence**: the persistent `ClaudeSession` (bidirectional
stream-json subprocess) is held on `WorkerThread._session` and survives
individual `Worker` crashes — the watchdog restarts the thread and the next
`Worker` inherits the same session. A full Fido restart or Docker image upgrade
still kills the live subprocess, but the worker persists the provider session
id in the repo's bind-mounted `.git/fido/state.json`; the next container seeds
its new provider session from that id so Claude can resume the same
conversation.

## Runner vs workspace clones

Fido runs from a dedicated **runner clone** at `/home/rhencke/home-runner/`,
separate from the **workspace clone** at `/home/rhencke/workspace/home/`.

- **Runner clone** — always on `main`, never dirty, never has feature branches.
  Fido imports his Python code from here. Self-restart does `git pull` here.
- **Workspace clone** — where Fido edits source files, commits, and pushes
  feature branches. Never used to run the server.

Launching: `/home/rhencke/start-fido.sh` (local, outside git) execs
`./fido up ...` from the runner clone.

Self-restart logic: `_self_restart` in `server.py` derives the runner clone
from `Path(__file__).resolve().parents[1]`, checks the git remote matches the
merged PR's repo, syncs the runner clone with exponential backoff (10s → 30s →
60s, 10-minute budget), stops workers, kills active provider children, and
exits `75`. The `./fido up` supervisor treats `75` as restart: it syncs the
runner clone, rebuilds the buildx runtime image, and runs the server again.

## Running

```bash
/home/rhencke/start-fido.sh
# or directly from the runner clone:
cd /home/rhencke/home-runner && ./fido up --port 9000 --secret-file /run/secrets/fido-secret \
  rhencke/confusio:/home/rhencke/workspace/confusio \
  FidoCanCode/home:/home/rhencke/workspace/home
```

## Testing and linting

`./fido ci` is the canonical path — same as CI and the pre-commit hook. Use
focused commands (`./fido tests`, `./fido ruff check .`, `./fido pyright`)
only while iterating; the commit path is `./fido ci`.

## Module guide

Top-level modules live in `src/fido/`. A handful of entry points:

- `server.py` — HTTP webhook handler, signature verification, repo routing
- `worker.py` — per-repo `WorkerThread` / `Worker` loop
- `events.py` — event dispatch, Opus triage/reply, reactions, task creation
- `tasks.py` — `tasks.json` CRUD with flock; rescoping logic

`ls src/fido/` for the full list — there are ~36 modules and any inline table
will drift.

System prompts for sub-Claude live in `sub/*.md`.

### Coordination models (`models/`)

Rocq source files (`.v`) that formally specify Fido's coordination invariants.
Each model is extracted to Python and run as a runtime oracle that crashes
loudly when the invariant is violated. Generated Python lives in
`src/fido/rocq/`.

**Survey:** `models/BUG_MINED_INVARIANTS.md` — a structured analysis of 23+
closed `Bug:` issues mapped to 15 coordination invariant clusters (A–O). Each
cluster names the invariant, the bugs that motivated it, the Rocq model that
will prove it, and the D-series issue tracking the work. Start here when
investigating a coordination bug or planning a new model.

## Task type system

Tasks have a mandatory `type` field (`TaskType` enum in `fido.types`):

| Value | Meaning |
|-------|---------|
| `ci` | CI failure fix |
| `thread` | PR comment / review thread work |
| `spec` | Planned spec task (default for setup) |

Tasks also use `TaskStatus`: `pending`, `completed`, `in_progress`.

CLI:

```bash
./fido task <work_dir> add <type> <title> [description]  # prints task JSON
./fido task <work_dir> complete <task_id>                # complete by ID
./fido task <work_dir> list                              # list as JSON
```

`_pick_next_task` priority: in-progress first, then `ci`, then first in list
(thread and spec are equal).

### Dynamic task reordering (rescoping)

When a `thread`-type task is created (PR comment feedback), `create_task()`
triggers a background Opus call via `reorder_tasks()` to reorder and rewrite
the pending task list based on dependency analysis.

- Only fires for `thread` tasks — `spec` tasks from setup are already ordered.
- Double-read pattern: read before the Opus call (for the prompt), then re-read
  inside the write lock (to pick up concurrent additions). Tasks added while
  Opus thinks are preserved as `newly_added`, not dropped.
- Pending tasks Opus omits are marked completed (not removed). In-progress
  tasks omitted trigger `_on_inprogress_affected` so the worker aborts and
  picks the new next task.
- Thread tasks completed by rescoping or modified trigger a `comment_issue`
  notification to the original commenter via `_notify_thread_change()`.
- Prompt builder: `rescope_prompt()` in `prompts.py`; response parser:
  `_parse_reorder_response()` in `tasks.py`.

## Conventions

- **100% test coverage** — CI enforced, no exceptions.
- **ruff** — lint + format on all Python.
- **PRs required** — branch protection on main.
- **Pre-commit hook** — blocks commits that fail format/lint/tests. Before
  running a test suite or build step "as a good-citizen check", compare the
  pre-commit hook against the GHA jobs marked required for a PR to `main`. If
  they're reasonably similar, skip the standalone invocation and just attempt
  the commit (without `--no-verify`). More reliable than guessing at the
  build/test steps and avoids running the suite twice.
- **Do a dedup pass before commit** — after the feature works, do one explicit
  pass over the touched code to consolidate any new duplication. Don't stop at
  green tests if the diff still contains obvious new duplicate logic that can
  be merged cleanly.
- **No hacks** — do not compensate for backend, extraction, generator, or
  runtime bugs in models, generated files, or tests. Fix the layer that is
  wrong rather than adding a workaround in a neighboring layer. In particular,
  the Rocq extraction backend must emit Python that is already formatted — do
  not add a post-extraction `ruff format` step in the `model-format` Dockerfile
  stage to paper over output that does not pass the format check. If extracted
  Python fails the format check, fix the pretty-printer in `python.ml`.
- **No compatibility shims** — when replacing a path or interface, remove the
  old one instead of preserving a parallel legacy path.
- **Verify upstream facts** — when a fix depends on external or standard-library
  behavior, check the primary source instead of guessing from memory.
- **Local entry point** — use `./fido help`. Do not call host `uv` for normal
  checks or server startup. The launcher owns buildx image selection, UID/GID
  mapping, credentials mounts, and stdin passthrough.
- **No `--no-cache` with docker buildx** — never pass `--no-cache` to `./fido
  ci`, `./fido make-rocq`, or any other buildx-backed command. The flag
  bypasses BuildKit's layer cache and destroys rebuild time.
- **No `@staticmethod` on behavior-bearing code** — static methods can't be
  patched via `self` and resist constructor-DI; see OO architecture below.
- **Prefer explicit object boundaries; keep module-level code thin and
  delegated** — new behavior lives on injected objects, not free functions.
- **Python target: 3.14t only** — free-threaded, no GIL. Don't add `from
  __future__` imports or conditional code for older Python versions.
- **Thread safety (Python 3.14t, free-threaded)** — do **not** rely on the GIL
  for atomicity. Every shared mutable state (dicts, sets, lists, counters,
  attribute mutations observed from other threads) must be guarded by an
  explicit lock, or use a primitive that documents its own thread-safe
  contract (`threading.Event`, `queue.Queue`, `threading.local`). In
  particular, `dict.setdefault`, attribute reads, and integer increments are
  **not** safe across threads without a lock. When in doubt, hold the lock.

## OO + constructor-DI architecture

All behavior lives on classes with dependencies injected through the
constructor. Module-level code is restricted to:

- **Constants** and pure data (no side effects, no I/O)
- **Value-only helpers** — pure functions that transform data and take no
  collaborators (e.g. `parse_event_type(header)`)
- **Dataclasses, enums, exceptions, Protocols** — type definitions only
- **Thin `run()` / `main()` composition roots** — the *only* places that call
  constructors with real collaborators and then delegate

A composition root assembles real objects in one place, then delegates:

```python
def run(work_dir: Path) -> int:
    return Worker(work_dir, GitHub()).run()  # assembles, then delegates
```

Collaborators are accepted via `__init__`, not instantiated internally. Tests
construct `Worker(tmp_path, mock_gh)` directly instead of patching
`fido.worker.GitHub`.

### Migration smells

These patterns signal incomplete migration to constructor-DI. Don't introduce
new instances; when you touch code that contains them, finish the job.

- **Callable-slot DI** — attributes wired up as default-argument overrides
  (`def __init__(self, ..., _run=subprocess.run): self._run = _run`) instead
  of typed injected collaborators. Examples: `_run`, `_print_prompt`,
  `_start`, `_fn_*` parameters on `Worker`, `Events`, `Tasks`.
- **Patch-heavy tests** — `@patch("fido.worker.subprocess.run")` decorators
  override module-level names from outside. Replace with a `MagicMock` or
  hand-rolled fake passed in at construction time.

## Coordination ethos

The goal is **smaller coordination code with fewer timing branches** — not
abstraction for its own sake. Every rule below exists because it eliminates a
class of race.

### Single-owner mutable state

One object owns a mutable bucket. Other objects send commands to it; they
never reach in and mutate its internals directly. The owner serializes all
mutations through its own lock — callers never acquire the owner's lock
themselves.

```python
class WorkerRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: dict[str, Worker] = {}

    def register(self, repo: str, worker: Worker) -> None:
        with self._lock:
            self._workers[repo] = worker
```

**Reviewer signal:** if you see a lock acquired *outside* the class that owns
the data it protects, that's a reach-through. Push the lock inward.

### Command translation at entry boundaries

Webhook events and CLI inputs are translated into typed commands or tasks at
the boundary. After that point, internal objects coordinate through those
commands, not through ambient state mutations scattered across the handler.

```python
def handle_webhook(payload: dict) -> None:
    event = parse_event(payload)          # typed value, no side effects
    self.dispatcher.dispatch(event)       # owner decides what to mutate
```

The translation layer is pure. The dispatcher is the single owner that decides
what changes.

**Reviewer signal:** if a webhook handler or CLI entry point mutates a dict,
list, or counter that lives outside its own scope, that's an ambient state
mutation. Translate first, dispatch second.

### Durable outbox / store before acting

When an intent must survive a crash or restart, write it to the durable store
*before* acting on it. `tasks.json` with `flock` is the canonical example: a
task is appended to the file (under lock) before the worker starts executing
it. If Fido crashes mid-task, the task is still in the list on restart.

In-memory coordination (queues, events, locks) handles the *current run*. The
durable store handles *across runs*. Don't conflate them.

**Reviewer signal:** if an action is taken before the corresponding record is
written to `tasks.json` (or another durable store), the order is wrong.

### Patterns to reject in review

| Pattern | Why it's wrong |
|---------|----------------|
| Module-level mutable dict/set/list used as coordination state | Any thread can mutate it; ownership is unclear; lock discipline is impossible to enforce |
| Long-lived callable-slot seams (`_fn_start`, `_run=subprocess.run`) used to inject cross-thread callbacks | Callable slots hide ownership; the real fix is a typed collaborator with a clear owner |
| Cross-thread reach-through (thread A reads/writes thread B's `_private` attribute directly) | Bypasses the owner's lock; creates invisible coupling; makes reasoning about state impossible |
| Ambient global set/cleared across requests (`request_context`, `current_repo`) | Thread-local globals are invisible dependencies; translate at the boundary instead |

## Fail-fast / fail-closed

Core runtime paths — webhook handler, worker loop, task engine, self-restart —
must fail loudly and early. Silent recovery masks real bugs and turns
transient errors into permanent state corruption.

- **No broad catch-log-continue in authoritative runner paths.** A bare
  `except Exception: log(...)` that lets the loop continue is almost always
  wrong. If an exception means the current task or request is unrecoverable,
  propagate it (or abort the task) rather than swallowing it.
- **No synthetic success from real failures.** Don't convert a failure into an
  empty string, `None`, a default value, or a fake-success return so the caller
  never finds out. The caller needs to know.
- **Subprocess failures must be explicit.** Always pass `check=True` to
  `subprocess.run` / `check_output`, or check `returncode` explicitly and
  raise. Ignoring a non-zero exit is the subprocess equivalent of
  catch-log-continue.
- **No `.get()` defaults for required keys.** If an external payload (GitHub
  webhook JSON, Claude response JSON, tasks.json) is required to contain a
  key, index it directly (`payload["action"]`) rather than `.get("action",
  "")`. A `KeyError` is much easier to debug than a downstream `NoneType` error
  or a silently skipped handler.
- **Fail closed on startup precondition failures.** If Fido cannot verify a
  required precondition at startup (missing secret file, bad config,
  unreachable repo), exit rather than continue in a degraded state. A Fido
  process that starts without a valid HMAC secret will silently accept forged
  webhooks.
