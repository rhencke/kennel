"""Tests for the `uv run tests` entrypoint."""

from __future__ import annotations

from unittest.mock import patch

from kennel.tests_main import ensure_rocq_python_artifacts, main


def test_ensure_rocq_python_artifacts_runs_export_helper() -> None:
    with patch("subprocess.run") as mock_run:
        ensure_rocq_python_artifacts()

    ((args,), kwargs) = mock_run.call_args
    assert args[0].endswith("rocq-python-extraction/export_pytest_generated.sh")
    assert kwargs["check"] is True
    assert (kwargs["cwd"] / "pyproject.toml").is_file()


def test_main_delegates_to_pytest_with_repo_defaults() -> None:
    with (
        patch("sys.argv", ["tests", "-q"]),
        patch("kennel.tests_main.ensure_rocq_python_artifacts") as mock_ensure,
        patch("pytest.main", return_value=0) as mock_pytest_main,
    ):
        result = main()

    assert result == 0
    mock_ensure.assert_called_once_with()
    mock_pytest_main.assert_called_once_with(
        [
            "--cov",
            "--cov-report=term-missing",
            "--cov-fail-under=100",
            "-q",
        ]
    )
