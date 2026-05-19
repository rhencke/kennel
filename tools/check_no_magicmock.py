#!/usr/bin/env python3
"""Guard: reject new MagicMock usage outside exemptions.

MagicMock is a generic dynamic mock — it hides ownership and makes
constructor-DI migrations look complete while tests still depend on
untyped behavior.

New tests must use hand-rolled mock classes, fakes, spies, stubs, value
objects, or typed collaborators instead.  See CLAUDE.md "OO + constructor-DI
architecture" and #1773 for the migration epic.
"""

import re
import sys
from pathlib import Path

# Temporary exemptions — existing MagicMock sites pending
# constructor-DI migration under #1773.  Do not add new files here; fix the
# root cause instead.
_EXEMPTIONS: frozenset[str] = frozenset(
    {
        "tests/fakes.py",
        "tests/test_check_no_magicmock.py",  # checker's own test fixture strings
        "tests/test_claude.py",
        "tests/test_claude_hold_for_handler.py",
        "tests/test_claude_stream_fsm_oracle.py",
        "tests/test_cli.py",
        "tests/test_codex.py",
        "tests/test_copilotcli.py",
        "tests/test_coverage_fills.py",
        "tests/test_events.py",
        "tests/test_github.py",
        "tests/test_infra.py",
        "tests/test_live_provider_stats.py",
        "tests/test_provider.py",
        "tests/test_provider_factory.py",
        "tests/test_provider_pressure.py",
        "tests/test_rate_limit.py",
        "tests/test_registry.py",
        "tests/test_server.py",
        "tests/test_session_agent.py",
        "tests/test_status.py",
        "tests/test_sync_tasks_cli.py",
        "tests/test_synthesis_call.py",
        "tests/test_synthesis_executor.py",
        "tests/test_task_queue_rescope.py",
        "tests/test_tasks.py",
        "tests/test_watchdog.py",
        "tests/test_watchdog_fsm_oracle.py",
        "tests/test_worker.py",
        "tests/test_worker_persist_session_id.py",
    }
)

# Tests: any line that mentions MagicMock (import or call-site).
_PATTERN: re.Pattern[str] = re.compile(r"\bMagicMock\b")

# Src: only import statements — avoids false positives from docstrings that
# mention MagicMock as a negative example (e.g. session_lock_watchdog.py).
_SRC_IMPORT_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:import|from)\b.*\bMagicMock\b"
)

_GUIDANCE = (
    "\nMagicMock is a generic dynamic mock — it hides ownership and makes\n"
    "constructor-DI migrations look complete while tests still depend on\n"
    "untyped behavior.\n"
    "\n"
    "Fix: use hand-rolled mock classes, fakes, spies, or typed collaborators\n"
    "instead.  Accept collaborators via __init__ and pass fakes at construction\n"
    "time.  See CLAUDE.md 'OO + constructor-DI architecture' and #1773.\n"
)


def check(
    root: Path,
    exemptions: frozenset[str] | None = None,
) -> int:
    """Scan ``root/tests`` and ``root/src`` for MagicMock outside exemptions.

    Returns 0 if clean, 1 if violations found.  Violations are written to
    stderr along with a pointer to constructor-DI.
    """
    if exemptions is None:
        exemptions = _EXEMPTIONS
    violations: list[str] = []

    tests_dir = root / "tests"
    if tests_dir.is_dir():
        for py_file in sorted(tests_dir.rglob("*.py")):
            rel = py_file.relative_to(root).as_posix()
            if rel in exemptions:
                continue
            for lineno, line in enumerate(py_file.read_text().splitlines(), 1):
                if _PATTERN.search(line):
                    violations.append(f"{rel}:{lineno}: MagicMock usage\n")

    src_dir = root / "src"
    if src_dir.is_dir():
        for py_file in sorted(src_dir.rglob("*.py")):
            rel = py_file.relative_to(root).as_posix()
            if rel in exemptions:
                continue
            for lineno, line in enumerate(py_file.read_text().splitlines(), 1):
                if _SRC_IMPORT_PATTERN.search(line):
                    violations.append(
                        f"{rel}:{lineno}: MagicMock import in src\n"
                    )

    if violations:
        sys.stderr.write("".join(violations))
        sys.stderr.write(_GUIDANCE)
        sys.stderr.write(f"\n{len(violations)} violation(s) found.\n")
        return 1
    return 0


def main() -> int:
    """Entry point: scan from the repo root and return the check result."""
    root = Path(__file__).resolve().parents[1]
    return check(root)


if __name__ == "__main__":
    raise SystemExit(main())
