#!/usr/bin/env python3
"""Generate the CI workflow from the buildx bake target graph."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
FIDO = ROOT / "fido"


@dataclass
class CopyInstruction:
    sources: list[str]
    from_name: str | None = None


@dataclass
class RunInstruction:
    uses_cache_mount: bool


Instruction = CopyInstruction | RunInstruction


@dataclass
class Stage:
    name: str
    base: str
    instructions: list[Instruction] = field(default_factory=list)


def bake_plan() -> dict[str, object]:
    output = subprocess.check_output(
        ["docker", "buildx", "bake", "--print", "warm"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1:
        sys.exit(f"docker buildx bake --print did not emit JSON:\n{output}")
    return json.loads(output[start : end + 1])


def rocq_cache_paths() -> dict[str, str]:
    script = FIDO.read_text()
    paths: dict[str, str] = {}
    for variable, key in (
        ("image_cache", "image"),
        ("build_cache", "buildx"),
        ("cache_context", "context"),
    ):
        match = re.search(
            rf'^\s*local {variable}="\$repo_root/([^"]+)"$',
            script,
            flags=re.MULTILINE,
        )
        if match is None:
            sys.exit(f"could not find {variable} in {FIDO}")
        paths[key] = match.group(1)
    return paths


def cache_entries(plan: dict[str, object]) -> list[tuple[str, str, str]]:
    _ = cache_scopes(plan)
    entries = []
    for key, path in rocq_cache_paths().items():
        target_key = "rocq-image" if key == "image" else "rocq-models"
        entries.append((f"rocq-model-{key}", path, target_key))
    return entries


def cache_scopes(plan: dict[str, object]) -> list[str]:
    targets = plan.get("target")
    if not isinstance(targets, dict):
        sys.exit("buildx bake plan did not include target map")
    return sorted(targets)


def joined_dockerfile_lines(path: Path) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        lines.append(current + line)
        current = ""
    if current:
        lines.append(current)
    return lines


def parse_dockerfile(path: Path) -> list[Stage]:
    stages: list[Stage] = []
    for line in joined_dockerfile_lines(path):
        parts = shlex.split(line, comments=False)
        if not parts:
            continue
        keyword = parts[0].upper()
        if keyword == "FROM":
            base = parts[1]
            lowered = [part.lower() for part in parts]
            if "as" in lowered:
                name = parts[lowered.index("as") + 1]
            else:
                name = str(len(stages))
            stages.append(Stage(name=name, base=base))
            continue
        if not stages:
            continue
        if keyword == "COPY":
            from_name: str | None = None
            args: list[str] = []
            for part in parts[1:]:
                if part.startswith("--from="):
                    from_name = part.split("=", maxsplit=1)[1]
                elif part.startswith("--"):
                    continue
                else:
                    args.append(part)
            if len(args) >= 2:
                stages[-1].instructions.append(
                    CopyInstruction(sources=args[:-1], from_name=from_name)
                )
        elif keyword == "RUN":
            stages[-1].instructions.append(
                RunInstruction(uses_cache_mount="--mount=type=cache" in line)
            )
    return stages


def stage_map(stages: list[Stage]) -> dict[str, Stage]:
    return {stage.name: stage for stage in stages}


def normalize_context_source(context: str, source: str) -> str:
    if source == ".":
        return context
    if context == ".":
        return source
    return f"{context.rstrip('/')}/{source}"


def named_target(plan: dict[str, object], bake_target: str, name: str) -> str | None:
    targets = plan.get("target")
    if not isinstance(targets, dict):
        return None
    target = targets.get(bake_target)
    if not isinstance(target, dict):
        return None
    contexts = target.get("contexts", {})
    if not isinstance(contexts, dict):
        return None
    value = contexts.get(name)
    if isinstance(value, str) and value.startswith("target:"):
        return value.removeprefix("target:")
    return None


def arg_named_target(plan: dict[str, object], bake_target: str, arg: str) -> str | None:
    targets = plan.get("target")
    if not isinstance(targets, dict):
        return None
    target = targets.get(bake_target)
    if not isinstance(target, dict):
        return None
    args = target.get("args", {})
    if not isinstance(args, dict):
        return None
    value = args.get(arg)
    if isinstance(value, str):
        return named_target(plan, bake_target, value)
    return None


def dockerfile_for_bake_target(
    plan: dict[str, object], bake_target: str
) -> tuple[str, Path, str]:
    targets = plan.get("target")
    if not isinstance(targets, dict):
        sys.exit("buildx bake plan did not include target map")
    target = targets.get(bake_target)
    if not isinstance(target, dict):
        sys.exit(f"buildx bake plan did not include target {bake_target!r}")
    context = str(target.get("context", "."))
    dockerfile = ROOT / context / str(target.get("dockerfile", "Dockerfile"))
    target_stage = str(target.get("target", ""))
    return context, dockerfile, target_stage


def target_inputs(
    plan: dict[str, object],
    bake_target: str,
    *,
    target_stage_override: str | None = None,
    include_cache_mount_stages: bool = False,
) -> set[str]:
    context, dockerfile, target_stage = dockerfile_for_bake_target(plan, bake_target)
    stages = parse_dockerfile(dockerfile)
    stages_by_name = stage_map(stages)
    if not target_stage:
        target_stage = stages[-1].name
    if target_stage_override is not None:
        target_stage = target_stage_override

    inputs = {str(dockerfile.relative_to(ROOT))}
    seen_stages: set[str] = set()
    seen_bake_targets = {bake_target}

    def add_bake_target(name: str) -> None:
        if name in seen_bake_targets:
            return
        seen_bake_targets.add(name)
        inputs.update(target_inputs(plan, name))

    def add_stage(stage_name: str, *, stop_after_cache_mounts: bool = False) -> None:
        if stage_name in seen_stages:
            return
        seen_stages.add(stage_name)
        stage = stages_by_name[stage_name]
        if stage.base in stages_by_name:
            add_stage(stage.base, stop_after_cache_mounts=stop_after_cache_mounts)
        elif stage.base.startswith("${") and stage.base.endswith("}"):
            if target := arg_named_target(plan, bake_target, stage.base[2:-1]):
                add_bake_target(target)
        for instruction in stage.instructions:
            if isinstance(instruction, CopyInstruction):
                if instruction.from_name is None:
                    inputs.update(
                        normalize_context_source(context, source)
                        for source in instruction.sources
                    )
                elif instruction.from_name in stages_by_name:
                    add_stage(
                        instruction.from_name,
                        stop_after_cache_mounts=stop_after_cache_mounts,
                    )
                elif target := named_target(plan, bake_target, instruction.from_name):
                    add_bake_target(target)
            elif (
                stop_after_cache_mounts
                and isinstance(instruction, RunInstruction)
                and instruction.uses_cache_mount
            ):
                return

    if include_cache_mount_stages:
        for stage in stages:
            if any(
                isinstance(instruction, RunInstruction) and instruction.uses_cache_mount
                for instruction in stage.instructions
            ):
                add_stage(stage.name, stop_after_cache_mounts=True)
    else:
        add_stage(target_stage)
    return inputs


def cache_key_inputs(plan: dict[str, object]) -> dict[str, list[str]]:
    warm_targets = cache_scopes(plan)
    mount_inputs: set[str] = {"docker-bake.hcl"}
    for target in warm_targets:
        mount_inputs.update(
            target_inputs(plan, target, include_cache_mount_stages=True)
        )
    return {
        "rocq-image": sorted(target_inputs(plan, "rocq-image")),
        "rocq-models": sorted(
            target_inputs(plan, "format", target_stage_override="extract")
        ),
        "buildkit-mounts": sorted(mount_inputs),
    }


def hashfiles_pattern(path: str) -> str:
    if any(char in path for char in "*?["):
        return path
    full_path = ROOT / path
    if full_path.is_dir():
        return f"{path.rstrip('/')}/**"
    return path


def hashfiles_expression(paths: list[str]) -> str:
    patterns = (hashfiles_pattern(path).replace("'", "''") for path in paths)
    quoted = ", ".join(f"'{pattern}'" for pattern in patterns)
    return f"${{{{ hashFiles({quoted}) }}}}"


def restore_step(name: str, path: str, key_expression: str) -> str:
    step_id = "restore-" + name.replace("-", "_")
    return f"""\
      - name: Restore {name} build cache
        id: {step_id}
        uses: actions/cache/restore@v4
        with:
          path: {path}
          key: buildx-{name}-${{{{ runner.os }}}}-{key_expression}
          restore-keys: |
            buildx-{name}-${{{{ runner.os }}}}-
"""


def save_step(name: str, path: str, key_expression: str) -> str:
    restore_id = "restore-" + name.replace("-", "_")
    return f"""\
      - name: Save {name} build cache
        if: success() && steps.{restore_id}.outputs.cache-hit != 'true'
        uses: actions/cache/save@v4
        with:
          path: {path}
          key: buildx-{name}-${{{{ runner.os }}}}-{key_expression}
"""


def render(plan: dict[str, object]) -> str:
    keys = cache_key_inputs(plan)
    key_expressions = {key: hashfiles_expression(paths) for key, paths in keys.items()}
    restores = "".join(
        restore_step(name, path, key_expressions[key])
        for name, path, key in cache_entries(plan)
    )
    saves = "".join(
        save_step(name, path, key_expressions[key])
        for name, path, key in cache_entries(plan)
    )
    scopes = ", ".join(cache_scopes(plan))
    return f"""\
# Generated by ./fido gen-workflows; do not edit by hand.
# Refresh with: ./fido gen-workflows
# Buildx gha cache scopes discovered from bake: {scopes}
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:
  schedule:
    - cron: '17 3 */6 * *'

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    env:
      FIDO_BUILDX_CACHE_BACKEND: gha
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
        id: setup-buildx
{restores}      - name: Restore buildkit cache mounts
        id: restore-buildkit-mounts
        uses: actions/cache@v4
        with:
          path: .cache/buildkit-mounts
          key: buildkit-mounts-${{{{ runner.os }}}}-{key_expressions["buildkit-mounts"]}
          restore-keys: |
            buildkit-mounts-${{{{ runner.os }}}}-
      - name: Inject buildkit cache mounts
        uses: reproducible-containers/buildkit-cache-dance@v3.3.2
        with:
          builder: ${{{{ steps.setup-buildx.outputs.name }}}}
          cache-dir: .cache/buildkit-mounts
          dockerfile: models/Dockerfile
      - run: ./.githooks/pre-commit
      - name: Check committed model mirrors
        run: |
          ./fido make-rocq
          git diff --exit-code -- src/fido/rocq/
{saves}"""


def main() -> None:
    WORKFLOW.write_text(render(bake_plan()))


if __name__ == "__main__":
    main()
