#!/usr/bin/env python3
"""Generate the CI workflow from the buildx bake target graph."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
BUILD_GRAPH = ROOT / "tools" / "build_graph.sh"
FIDO = ROOT / "fido"


@dataclass
class CopyInstruction:
    sources: list[str]
    from_name: str | None = None


@dataclass
class RunInstruction:
    cache_mounts: list["CacheMount"] = field(default_factory=list)

    @property
    def uses_cache_mount(self) -> bool:
        return bool(self.cache_mounts)


@dataclass(frozen=True)
class CacheMount:
    id: str
    target: str


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
        ["docker", "buildx", "bake", "--print", "warm", "fido-test"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    return load_bake_plan(output)


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
            stages[-1].instructions.append(RunInstruction(parse_cache_mounts(parts)))
    return stages


def parse_cache_mounts(parts: list[str]) -> list[CacheMount]:
    mounts: list[CacheMount] = []
    for part in parts:
        if not part.startswith("--mount="):
            continue
        options: dict[str, str] = {}
        for option in part.removeprefix("--mount=").split(","):
            key, separator, value = option.partition("=")
            if separator:
                options[key] = value
        if options.get("type") != "cache":
            continue
        target = options.get("target")
        if target is None:
            continue
        mount_id = options.get("id", target.strip("/").replace("/", "-"))
        mounts.append(CacheMount(id=mount_id, target=target))
    return mounts


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
    keys = {
        "rocq-image": sorted(target_inputs(plan, "rocq-image")),
        "rocq-models": sorted(
            target_inputs(plan, "format", target_stage_override="extract")
        ),
    }
    for mount in cache_mounts_for_targets(plan, warm_targets):
        keys[f"buildkit-mount-{mount.id}"] = cache_mount_key_inputs(mount)
    return keys


def cache_mounts_for_targets(
    plan: dict[str, object], bake_targets: list[str]
) -> list[CacheMount]:
    mounts: dict[str, CacheMount] = {}
    visited: set[tuple[str, str]] = set()

    def visit_stage(
        context: str,
        stages_by_name: dict[str, Stage],
        bake_target: str,
        stage_name: str,
    ) -> None:
        key = (bake_target, stage_name)
        if key in visited:
            return
        visited.add(key)
        stage = stages_by_name[stage_name]
        if stage.base in stages_by_name:
            visit_stage(context, stages_by_name, bake_target, stage.base)
        elif stage.base.startswith("${") and stage.base.endswith("}"):
            if target := arg_named_target(plan, bake_target, stage.base[2:-1]):
                visit_target(target)
        for instruction in stage.instructions:
            if isinstance(instruction, CopyInstruction):
                if instruction.from_name in stages_by_name:
                    visit_stage(
                        context,
                        stages_by_name,
                        bake_target,
                        instruction.from_name,
                    )
                elif instruction.from_name is not None and (
                    target := named_target(plan, bake_target, instruction.from_name)
                ):
                    visit_target(target)
            elif isinstance(instruction, RunInstruction):
                for mount in instruction.cache_mounts:
                    existing = mounts.get(mount.id)
                    if existing is not None and existing.target != mount.target:
                        sys.exit(
                            f"cache mount id {mount.id!r} has conflicting targets: "
                            f"{existing.target!r} and {mount.target!r}"
                        )
                    mounts[mount.id] = mount

    def visit_target(name: str) -> None:
        context, dockerfile, target_stage = dockerfile_for_bake_target(plan, name)
        stages = parse_dockerfile(dockerfile)
        stages_by_name = stage_map(stages)
        if not target_stage:
            target_stage = stages[-1].name
        visit_stage(context, stages_by_name, name, target_stage)

    for target in bake_targets:
        visit_target(target)
    return [mounts[name] for name in sorted(mounts)]


def cache_mount_key_inputs(mount: CacheMount) -> list[str]:
    base = {"models/Dockerfile"}
    if mount.id.startswith("fido-uv-"):
        return sorted(base | {".python-version", "pyproject.toml", "uv.lock"})
    if mount.id == "fido-npm":
        return sorted(base | {"package.json", "package-lock.json"})
    if mount.id == "fido-pyright":
        return sorted(
            base
            | {".python-version", "pyproject.toml", "uv.lock", "pyrightconfig.json"}
        )
    return sorted(base)


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
    if group == "warm":
        names = [name for name in names if name != "fido-test"]
    return {name: content_hash(sorted(target_inputs(plan, name))) for name in names}


def render_build_graph(plan: dict[str, object]) -> str:
    groups = plan.get("group")
    if not isinstance(groups, dict):
        sys.exit("buildx bake plan did not include group map")
    graph_targets = {
        target: sorted(target_inputs(plan, target)) for target in cache_scopes(plan)
    }
    graph_targets["make-rocq"] = sorted(
        target_inputs(plan, "format", target_stage_override="export")
    )

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
        if group_name == "warm":
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


def cache_step_id(name: str) -> str:
    return "cache-" + name.replace("-", "_")


def buildkit_mount_cache_step(mount: CacheMount, key_expression: str) -> str:
    name = f"buildkit-mount-{mount.id}"
    step_id = cache_step_id(name)
    path = f".cache/buildkit-mounts/{mount.id}"
    cache_map = json.dumps(
        {path: {"target": mount.target, "id": mount.id}},
        indent=2,
        sort_keys=True,
    )
    indented_cache_map = "\n".join(
        f"            {line}" for line in cache_map.splitlines()
    )
    return f"""\
      - name: Cache {mount.id} BuildKit mount
        id: {step_id}
        uses: actions/cache@v4
        with:
          path: {path}
          key: buildkit-mount-{mount.id}-${{{{ runner.os }}}}-{key_expression}
          restore-keys: |
            buildkit-mount-{mount.id}-${{{{ runner.os }}}}-
      - name: Inject {mount.id} BuildKit mount
        uses: reproducible-containers/buildkit-cache-dance@v3.3.2
        with:
          builder: ${{{{ steps.setup-buildx.outputs.name }}}}
          cache-map: |
{indented_cache_map}
          scratch-dir: .cache/buildkit-mounts-scratch/{mount.id}
          skip-extraction: ${{{{ steps.{step_id}.outputs.cache-hit }}}}
"""


def render(plan: dict[str, object]) -> str:
    keys = cache_key_inputs(plan)
    key_expressions = {key: hashfiles_expression(paths) for key, paths in keys.items()}
    buildkit_mounts = cache_mounts_for_targets(plan, cache_scopes(plan))
    buildkit_mount_steps = "".join(
        buildkit_mount_cache_step(
            mount,
            key_expressions[f"buildkit-mount-{mount.id}"],
        )
        for mount in buildkit_mounts
    )
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
{restores}{buildkit_mount_steps}      - run: ./.githooks/pre-commit
      - name: Check committed model mirrors
        run: |
          ./fido make-rocq
          git diff --exit-code -- src/fido/rocq/
      - name: Check generated workflow files
        run: |
          ./fido gen-workflows
          git diff --exit-code -- .github/workflows/ci.yml tools/build_graph.sh
{saves}"""


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
        hashes = target_hashes(plan or bake_plan(), "warm")
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
