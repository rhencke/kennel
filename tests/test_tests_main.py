"""Tests for the project test entrypoint."""

from pathlib import Path
from typing import Any

import pytest

from fido.tests_main import (
    ensure_rocq_python_artifacts,
    main,
)


def test_ensure_rocq_python_artifacts_skips_prepared_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FIDO_ROCQ_PYTEST_ARTIFACTS", "prepared")

    calls: list[tuple[Any, Any]] = []
    ensure_rocq_python_artifacts(
        _run=lambda *args, **kwargs: calls.append((args, kwargs))
    )

    assert calls == []


def test_ensure_rocq_python_artifacts_runs_export_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIDO_ROCQ_PYTEST_ARTIFACTS", raising=False)

    run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_run(*args: object, **kwargs: object) -> None:
        run_calls.append((args, kwargs))

    ensure_rocq_python_artifacts(_run=fake_run)

    assert len(run_calls) == 1
    (positional,), kwargs = run_calls[0]
    assert str(positional[0]).endswith(
        "rocq-python-extraction/export_pytest_generated.sh"
    )
    assert kwargs["check"] is True
    # cwd must be the repo root — the directory containing src/fido/.  The
    # test used to hardcode ``cwd.name == "workspace"`` which broke whenever
    # the repo was checked out under a directory named anything else (e.g.
    # the build worktrees under ``home-preempt-bash``).  Anchor on the
    # actual marker of the repo root instead.
    cwd = Path(str(kwargs["cwd"]))
    assert (cwd / "src" / "fido" / "tests_main.py").is_file()


def test_module_executes_main_under_dunder_main() -> None:
    """``python -m fido.tests_main`` actually invokes :func:`main` (closes #1252).

    Regression: the module shipped without an ``if __name__ == "__main__"``
    guard for weeks, so ``./fido tests`` (and the CI pre-commit ``test``
    stage) was a silent no-op — every ``./fido ci`` succeeded regardless
    of real test status, letting 100+ failures pile up on main.  Read the
    file text and assert the guard is present so it can never silently
    rot again.
    """
    source = Path(__file__).resolve().parent.parent / "src" / "fido" / "tests_main.py"
    text = source.read_text()
    assert 'if __name__ == "__main__":' in text, (
        "fido.tests_main must keep the __main__ guard or `./fido tests` "
        "becomes a silent no-op (see #1252)."
    )
    assert "raise SystemExit(main())" in text, (
        "the __main__ guard must propagate main()'s exit code so CI can "
        "actually fail on test failure (see #1252)."
    )


def test_main_delegates_to_pytest_with_repo_defaults() -> None:
    ensure_calls: list[None] = []
    pytest_args_seen: list[list[str]] = []

    def fake_ensure() -> None:
        ensure_calls.append(None)

    def fake_pytest_main(args: list[str]) -> int:
        pytest_args_seen.append(args)
        return 0

    result = main(argv=["-q"], _ensure=fake_ensure, _pytest_main=fake_pytest_main)

    assert result == 0
    assert ensure_calls == [None]
    # ``-n 2`` is injected when the caller doesn't specify parallelism
    # (#1248 + PR #1254): caps xdist at 2 workers so total memory stays
    # under the 4 GiB cgroup cap on the test container.
    assert pytest_args_seen == [
        [
            "--cov=fido",
            "--cov=rocq-python-extraction/test",
            "--cov-report=term-missing",
            "--cov-fail-under=100",
            "-n",
            "2",
            "-q",
        ]
    ]


def test_main_respects_explicit_n_flag() -> None:
    """When the caller passes ``-n``, don't inject the default.

    Lets ``./fido tests -n 1`` (single worker, max safety) and
    ``./fido tests -n 4`` (full parallelism, when the box has headroom)
    both work without fighting the default.
    """
    pytest_args_seen: list[list[str]] = []

    def fake_pytest_main(args: list[str]) -> int:
        pytest_args_seen.append(args)
        return 0

    main(
        argv=["-n", "1", "-q"],
        _ensure=lambda: None,
        _pytest_main=fake_pytest_main,
    )

    assert len(pytest_args_seen) == 1
    args = pytest_args_seen[0]
    # exactly one ``-n`` should be present — ours, not a duplicate
    assert args.count("-n") == 1
    assert "1" in args  # the caller's value, not 2
