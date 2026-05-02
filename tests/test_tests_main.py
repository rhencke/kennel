"""Tests for the project test entrypoint."""

from unittest.mock import patch

import pytest

from fido.tests_main import (
    ensure_rocq_python_artifacts,
    main,
)


def test_ensure_rocq_python_artifacts_skips_prepared_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FIDO_ROCQ_PYTEST_ARTIFACTS", "prepared")

    with patch("subprocess.run") as mock_run:
        ensure_rocq_python_artifacts()

    mock_run.assert_not_called()


def test_ensure_rocq_python_artifacts_runs_export_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    monkeypatch.delenv("FIDO_ROCQ_PYTEST_ARTIFACTS", raising=False)

    with patch("subprocess.run") as mock_run:
        ensure_rocq_python_artifacts()

    ((args,), kwargs) = mock_run.call_args
    assert args[0].endswith("rocq-python-extraction/export_pytest_generated.sh")
    assert kwargs["check"] is True
    # cwd must be the repo root — the directory containing src/fido/.  The
    # test used to hardcode ``cwd.name == "workspace"`` which broke whenever
    # the repo was checked out under a directory named anything else (e.g.
    # the build worktrees under ``home-preempt-bash``).  Anchor on the
    # actual marker of the repo root instead.
    cwd = Path(kwargs["cwd"])
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
    from pathlib import Path

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
    with (
        patch("sys.argv", ["tests", "-q"]),
        patch("fido.tests_main.ensure_rocq_python_artifacts") as mock_ensure,
        patch("pytest.main", return_value=0) as mock_pytest_main,
    ):
        result = main()

    assert result == 0
    mock_ensure.assert_called_once_with()
    mock_pytest_main.assert_called_once_with(
        [
            "--cov=fido",
            "--cov=rocq-python-extraction/test",
            "--cov-report=term-missing",
            "--cov-fail-under=100",
            "-q",
        ]
    )
