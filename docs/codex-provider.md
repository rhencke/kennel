# Codex Provider Notes

Codex provider work is pinned to the stable upstream release
[`rust-v0.125.0`](https://github.com/openai/codex/releases/tag/rust-v0.125.0)
and npm package `@openai/codex@0.125.0`.

## One-Shot CLI

The non-persistent harness uses the stable CLI surface:

```bash
codex exec --json --model <model> --sandbox danger-full-access \
  --ask-for-approval never --skip-git-repo-check -C <cwd> -
```

The prompt is passed on stdin. CI tests use fixtures and must not call live
Codex.

The JSONL event shape is defined by upstream
`codex-rs/exec/src/exec_events.rs`:

- `thread.started` carries `thread_id`, which is the resume handle for later
  work.
- `item.completed` with `item.type == "agent_message"` carries assistant text.
- `error` and `turn.failed` carry provider/auth/quota failure messages.

## Persistent App-Server Transport

Persistent transport uses the pinned app-server protocol over stdio:

```bash
codex app-server --listen stdio://
```

The wire format is newline-delimited JSON without a `jsonrpc` version field.
Fido sends `initialize`, then the `initialized` notification, then uses:

- `account/rateLimits/read` for quota/status snapshots.
- `thread/start` and `thread/resume` for persistent Codex thread ids.
- `turn/start` for prompt turns.
- `turn/interrupt` for worker preemption/cancel.

Fido always sends `approvalPolicy: "never"` and a `danger-full-access`
sandbox policy because container and branch isolation happen outside Codex.
The live worker-provider factory is still intentionally gated until the final
Codex epic issue wires Codex into `CodexClient`/provider selection.

For bootstrap validation against the local Codex install, run:

```bash
PYTHONPATH=src ./pyproject .venv/bin/python tools/codex_appserver_smoke.py \
  --turn --resume --interrupt --preempt \
  --prompt 'Run `sleep 20` in the shell, then reply done.'
docker run --rm --interactive --user "$(id -u):$(id -g)" \
  --env "HOME=$HOME" --env "PYTHONPATH=/workspace/src" --network host \
  --volume "$PWD:/workspace" --volume "$HOME/workspace:$HOME/workspace" \
  --volume "$HOME/.codex:$HOME/.codex" --workdir /workspace \
  --entrypoint /opt/fido-venv/bin/python3 fido-test:local \
  tools/codex_appserver_smoke.py --turn --resume --interrupt --preempt \
  --prompt 'Run `sleep 20` in the shell, then reply done.'
```

The smoke driver is intentionally outside CI. It is for reconciling Fido's
fixtures with live app-server behavior before codifying that behavior in unit
tests. The Docker form validates the same image, Codex binary, and `~/.codex`
mount shape Fido uses at runtime.

Pinned app-server schemas:

- `codex-rs/app-server-protocol/schema/json/v2/ThreadStartParams.json`
- `codex-rs/app-server-protocol/schema/json/v2/TurnStartParams.json`
- `codex-rs/app-server-protocol/schema/json/v2/TurnInterruptParams.json`
- `codex-rs/app-server-protocol/schema/json/v2/GetAccountRateLimitsResponse.json`
