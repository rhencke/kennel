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

## Persistent-Session Fallback Refs

Persistent transport is not implemented in the one-shot harness. If stable CLI
resume is insufficient for later issues, use these pinned app-server schemas as
the fallback source of truth:

- `codex-rs/app-server-protocol/schema/json/v2/ThreadStartParams.json`
- `codex-rs/app-server-protocol/schema/json/v2/TurnStartParams.json`
- `codex-rs/app-server-protocol/schema/json/v2/TurnInterruptParams.json`
- `codex-rs/app-server-protocol/schema/json/v2/GetAccountRateLimitsResponse.json`
