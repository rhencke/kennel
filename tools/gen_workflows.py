#!/usr/bin/env python3
"""Generate the CI workflow from the buildx bake target graph."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
BUILD_GRAPH = ROOT / "tools" / "build_graph.sh"


@dataclass
class CopyInstruction:
    sources: list[str]
    from_name: str | None = None


@dataclass
class RunInstruction:
    pass


Instruction = CopyInstruction | RunInstruction


@dataclass
class Stage:
    name: str
    base: str
    instructions: list[Instruction] = field(default_factory=list)


def load_bake_plan(output: str) -> dict[str, object]:
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1:
        sys.exit(f"docker buildx bake --print did not emit JSON:\n{output}")
    return json.loads(output[start : end + 1])


def bake_plan() -> dict[str, object]:
    output = subprocess.check_output(
        ["docker", "buildx", "bake", "--print", "ci", "fido-test", "make-rocq"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    return load_bake_plan(output)


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
            stages[-1].instructions.append(RunInstruction())
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

    def add_stage(stage_name: str) -> None:
        if stage_name in seen_stages:
            return
        seen_stages.add(stage_name)
        stage = stages_by_name[stage_name]
        if stage.base in stages_by_name:
            add_stage(stage.base)
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
                    add_stage(instruction.from_name)
                elif target := named_target(plan, bake_target, instruction.from_name):
                    add_bake_target(target)

    add_stage(target_stage)
    return inputs


def expanded_input_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if any(char in path for char in "*?["):
            files.extend(sorted(ROOT.glob(path)))
            continue
        full_path = ROOT / path
        if full_path.is_dir():
            files.extend(
                sorted(item for item in full_path.rglob("*") if item.is_file())
            )
        elif full_path.is_file():
            files.append(full_path)
    return sorted(set(files))


def content_hash(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in expanded_input_files(paths):
        rel = path.relative_to(ROOT).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def target_hashes(plan: dict[str, object], group: str) -> dict[str, str]:
    targets = plan.get("target")
    groups = plan.get("group")
    if not isinstance(targets, dict) or not isinstance(groups, dict):
        sys.exit("buildx bake plan did not include target and group maps")
    selected = groups.get(group, {})
    if not isinstance(selected, dict):
        sys.exit(f"buildx bake plan did not include group {group!r}")
    group_targets = selected.get("targets", [])
    if not isinstance(group_targets, list):
        sys.exit(f"buildx bake group {group!r} did not include targets")
    names = sorted(set(targets) | {str(target) for target in group_targets})
    if group == "ci":
        names = [name for name in names if name != "fido-test"]
    return {name: content_hash(sorted(target_inputs(plan, name))) for name in names}


def render_build_graph(plan: dict[str, object]) -> str:
    groups = plan.get("group")
    if not isinstance(groups, dict):
        sys.exit("buildx bake plan did not include group map")
    graph_targets = {
        target: sorted(target_inputs(plan, target)) for target in cache_scopes(plan)
    }
    graph_targets["make-rocq"] = sorted(target_inputs(plan, "make-rocq"))

    lines = [
        "# Generated by ./fido gen-workflows; do not edit by hand.",
        "# Refresh with: ./fido gen-workflows",
        "",
        "fido_build_targets_for_group() {",
        '  case "$1" in',
    ]
    for group_name, group_data in sorted(groups.items()):
        if not isinstance(group_data, dict):
            continue
        raw_targets = group_data.get("targets", [])
        if not isinstance(raw_targets, list):
            continue
        if group_name == "ci":
            targets = [target for target in cache_scopes(plan) if target != "fido-test"]
        else:
            targets = sorted(raw_targets)
        quoted_targets = " ".join(shlex.quote(str(target)) for target in targets)
        lines.extend(
            [
                f"    {shlex.quote(group_name)})",
                f"      printf '%s\\n' {quoted_targets}",
                "      ;;",
            ]
        )
    lines.extend(
        [
            "    *)",
            "      return 2",
            "      ;;",
            "  esac",
            "}",
            "",
            "fido_build_inputs_for_target() {",
            '  case "$1" in',
        ]
    )
    for target, inputs in sorted(graph_targets.items()):
        lines.extend(
            [
                f"    {shlex.quote(target)})",
                "      cat <<'EOF'",
                *inputs,
                "EOF",
                "      ;;",
            ]
        )
    lines.extend(
        [
            "    *)",
            "      return 2",
            "      ;;",
            "  esac",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def render(plan: dict[str, object]) -> str:
    scopes = ", ".join(cache_scopes(plan))
    return f"""\
# Generated by ./fido gen-workflows; do not edit by hand.
# Refresh with: ./fido gen-workflows
# Buildx local cache scopes discovered from bake: {scopes}
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  pre-commit:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - run: ./.githooks/pre-commit
      - name: Check committed model mirrors
        run: |
          ./fido make-rocq
          git diff --exit-code -- src/fido/rocq/
      - name: Check generated workflow files
        run: |
          ./fido gen-workflows
          git diff --exit-code -- .github/workflows/ci.yml tools/build_graph.sh
"""


def main() -> None:
    plan_path: Path | None = None
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--plan":
        plan_path = ROOT / args[1]
        args = args[2:]
    plan = load_bake_plan(plan_path.read_text()) if plan_path is not None else None

    if len(args) == 2 and args[0] == "--target-hashes":
        for name, digest in target_hashes(plan or bake_plan(), args[1]).items():
            print(f"{name} {digest}")
        return
    if len(args) == 2 and args[0] == "--target-hash":
        hashes = target_hashes(plan or bake_plan(), "ci")
        try:
            print(hashes[args[1]])
        except KeyError:
            sys.exit(f"buildx bake plan did not include target {args[1]!r}")
        return
    if args:
        sys.exit(
            "usage: gen_workflows.py [--plan PATH] [--target-hashes GROUP|--target-hash TARGET]"
        )
    plan = plan or bake_plan()
    WORKFLOW.write_text(render(plan))
    BUILD_GRAPH.write_text(render_build_graph(plan))


if __name__ == "__main__":
    main()
