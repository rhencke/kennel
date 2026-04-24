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
    monkeypatch.delenv("FIDO_ROCQ_PYTEST_ARTIFACTS", raising=False)

    with patch("subprocess.run") as mock_run:
        ensure_rocq_python_artifacts()

    ((args,), kwargs) = mock_run.call_args
    assert args[0].endswith("rocq-python-extraction/export_pytest_generated.sh")
    assert kwargs["check"] is True
    assert kwargs["cwd"].name == "workspace"


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
