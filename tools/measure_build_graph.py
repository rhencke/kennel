#!/usr/bin/env python3
"""Run warm build-graph scenarios and record timing/cache summaries."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "perf" / "build-graph" / "measurements.json"
STEP_RE = re.compile(r"^#(?P<id>\d+) \[(?P<label>[^\]]+)\]")


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    path: str | None = None
    append: str | None = None
    replace_old: str | None = None
    replace_new: str | None = None


SCENARIOS = [
    Scenario("warm_no_change", "Warm ./fido ci with no source changes."),
    Scenario(
        "warm_prod_python_edit",
        "Warm ./fido ci after a tiny prod Python edit.",
        path="src/fido/fido_help.py",
        append="\n# benchmark: warm_prod_python_edit\n",
    ),
    Scenario(
        "warm_test_edit",
        "Warm ./fido ci after a tiny test edit.",
        path="tests/test_main.py",
        replace_old='"""Tests for fido.main — top-level server entry point."""',
        replace_new='"""Tests for fido.main — top-level server entry point. Benchmark."""',
    ),
    Scenario(
        "warm_launcher_metadata_edit",
        "Warm ./fido ci after a tiny launcher/metadata edit.",
        path="pyproject.tools.toml",
        append="\n# benchmark: warm_launcher_metadata_edit\n",
    ),
    Scenario(
        "warm_model_edit",
        "Warm ./fido ci after a tiny Rocq model edit.",
        path="models/task_queue_rescope.v",
        append="\n(* benchmark: warm_model_edit *)\n",
    ),
    Scenario(
        "warm_rocq_test_edit",
        "Warm ./fido ci after a tiny Rocq extraction test edit.",
        path="rocq-python-extraction/test/python.v",
        append="\n(* benchmark: warm_rocq_test_edit *)\n",
    ),
    Scenario(
        "warm_pytest_focused",
        "Warm focused pytest run through ./fido pytest.",
        path=None,
        append=None,
    ),
    Scenario(
        "warm_ruff_focused",
        "Warm focused ruff run through ./fido ruff.",
        path=None,
        append=None,
    ),
    Scenario(
        "warm_pyright_focused",
        "Warm focused pyright run through ./fido pyright.",
        path=None,
        append=None,
    ),
]


def command_for(scenario: Scenario) -> list[str]:
    if scenario.name == "warm_pytest_focused":
        return ["./fido", "pytest", "tests/test_main.py", "-q"]
    if scenario.name == "warm_ruff_focused":
        return ["./fido", "ruff", "check", "src/fido/fido_help.py"]
    if scenario.name == "warm_pyright_focused":
        return ["./fido", "pyright", "src/fido/main.py"]
    return ["./fido", "ci"]


def apply_edit(worktree: Path, scenario: Scenario) -> None:
    if scenario.path is None:
        return
    path = worktree / scenario.path
    if scenario.append is not None:
        path.write_text(path.read_text() + scenario.append)
        return
    if scenario.replace_old is not None and scenario.replace_new is not None:
        path.write_text(
            path.read_text().replace(scenario.replace_old, scenario.replace_new)
        )
        return
    raise ValueError(f"scenario {scenario.name} has no edit operation")


def parse_step_summary(output: str) -> dict[str, Any]:
    steps: dict[str, dict[str, str]] = {}
    labels: dict[str, str] = {}
    for line in output.splitlines():
        match = STEP_RE.match(line)
        if match is None:
            continue
        step_id = match.group("id")
        labels[step_id] = match.group("label")
        status = None
        if " CACHED" in line:
            status = "cached"
        elif " DONE" in line:
            status = "done"
        elif " ERROR" in line:
            status = "error"
        elif " ..." in line:
            status = "running"
        if status is None:
            continue
        steps[step_id] = {"label": labels[step_id], "status": status}

    for line in output.splitlines():
        if not line.startswith("#"):
            continue
        step_id = line[1:].split(" ", 1)[0]
        if step_id not in labels:
            continue
        status = None
        if line.endswith(" CACHED"):
            status = "cached"
        elif line.endswith(" DONE"):
            status = "done"
        elif line.endswith(" ERROR"):
            status = "error"
        elif line.endswith(" CANCELED"):
            status = "canceled"
        elif line.endswith(" ..."):
            status = "running"
        if status is None:
            continue
        steps[step_id] = {"label": labels[step_id], "status": status}
    counts: dict[str, int] = {}
    for step in steps.values():
        counts[step["status"]] = counts.get(step["status"], 0) + 1
    return {
        "counts": counts,
        "steps": [
            {"id": int(step_id), **step}
            for step_id, step in sorted(steps.items(), key=lambda item: int(item[0]))
        ],
    }


def create_stash() -> str:
    before = subprocess.check_output(
        ["git", "stash", "list", "--format=%H"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    subprocess.run(
        [
            "git",
            "stash",
            "push",
            "--include-untracked",
            "-m",
            "measure-build-graph",
            "--",
            ".",
            ":(exclude).codex",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    after = subprocess.check_output(
        ["git", "stash", "list", "--format=%H"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    new_entries = [entry for entry in after if entry not in before]
    if len(new_entries) != 1:
        raise RuntimeError("expected exactly one new stash entry")
    return new_entries[0]


def restore_root_stash() -> None:
    subprocess.run(
        ["git", "stash", "pop", "--index"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def apply_stash(worktree: Path, stash_rev: str) -> None:
    subprocess.run(
        ["git", "stash", "apply", "--index", stash_rev],
        cwd=worktree,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def create_snapshot_commit(worktree: Path, stash_rev: str) -> str:
    apply_stash(worktree, stash_rev)
    subprocess.run(["git", "add", "-A"], cwd=worktree, check=True)
    cached_diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=worktree,
        check=False,
    )
    if cached_diff.returncode == 0:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            text=True,
        ).strip()
    if cached_diff.returncode != 1:
        raise subprocess.CalledProcessError(cached_diff.returncode, cached_diff.args)
    env = os.environ | {
        "GIT_AUTHOR_NAME": "Fido Benchmark",
        "GIT_AUTHOR_EMAIL": "fido@example.invalid",
        "GIT_COMMITTER_NAME": "Fido Benchmark",
        "GIT_COMMITTER_EMAIL": "fido@example.invalid",
    }
    commit = subprocess.run(
        ["git", "commit", "-m", "benchmark snapshot"],
        cwd=worktree,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        combined = commit.stdout + commit.stderr
        if "nothing to commit" in combined:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=worktree,
                text=True,
            ).strip()
        raise subprocess.CalledProcessError(
            commit.returncode,
            commit.args,
            output=commit.stdout,
            stderr=commit.stderr,
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        text=True,
    ).strip()


def run_scenario(
    worktree: Path, snapshot_rev: str, scenario: Scenario
) -> dict[str, Any]:
    subprocess.run(["git", "reset", "--hard", snapshot_rev], cwd=worktree, check=True)
    subprocess.run(["git", "clean", "-fd"], cwd=worktree, check=True)
    apply_edit(worktree, scenario)
    started = time.monotonic()
    env = os.environ | {"BUILDKIT_PROGRESS": "plain"}
    result = subprocess.run(
        command_for(scenario),
        cwd=worktree,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    return {
        "name": scenario.name,
        "description": scenario.description,
        "command": command_for(scenario),
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "step_summary": parse_step_summary(combined),
        "output_tail": combined.splitlines()[-40:],
    }


def measure(output: Path) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="fido-build-bench-"))
    worktree = temp_root / "worktree"
    stash_rev = create_stash()
    restored_root_stash = False
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
            cwd=ROOT,
            check=True,
        )
        snapshot_rev = create_snapshot_commit(worktree, stash_rev)
        restore_root_stash()
        restored_root_stash = True
        scenarios = [
            run_scenario(worktree, snapshot_rev, scenario) for scenario in SCENARIOS
        ]
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "snapshot_rev": snapshot_rev,
            "scenarios": scenarios,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        if not restored_root_stash:
            subprocess.run(
                ["git", "stash", "pop", "--index"],
                cwd=ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        subprocess.run(
            ["git", "stash", "drop", stash_rev],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=ROOT,
            check=False,
        )
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    measure(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
