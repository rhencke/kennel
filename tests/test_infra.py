"""Tests for fido.infra — infrastructure port protocols and real implementations."""

import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fido.infra import RealClock, RealFilesystem, RealOsProcess, RealProcessRunner


class TestRealProcessRunner:
    def test_run_delegates_to_subprocess_run(self) -> None:
        mock_result = MagicMock(stdout="v2.43.0\n")
        with patch("fido.infra.subprocess.run", return_value=mock_result) as mock_run:
            runner = RealProcessRunner()
            result = runner.run(
                ["git", "--version"],
                cwd="/tmp",
                capture_output=True,
                text=True,
                check=True,
            )
        mock_run.assert_called_once_with(
            ["git", "--version"],
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=True,
        )
        assert result is mock_result

    def test_run_forwards_arbitrary_kwargs(self) -> None:
        mock_result = MagicMock()
        with patch("fido.infra.subprocess.run", return_value=mock_result) as mock_run:
            RealProcessRunner().run(["true"], env={"FOO": "bar"}, timeout=5)
        mock_run.assert_called_once_with(
            ["true"], check=True, env={"FOO": "bar"}, timeout=5
        )

    def test_run_propagates_called_process_error(self) -> None:
        with (
            patch(
                "fido.infra.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["bad"]),
            ),
            pytest.raises(subprocess.CalledProcessError),
        ):
            RealProcessRunner().run(["bad"], check=True)


class TestRealClock:
    def test_sleep_delegates_to_time_sleep(self) -> None:
        with patch("fido.infra.time.sleep") as mock_sleep:
            RealClock().sleep(2.5)
        mock_sleep.assert_called_once_with(2.5)

    def test_monotonic_delegates_to_time_monotonic(self) -> None:
        with patch("fido.infra.time.monotonic", return_value=123.456):
            result = RealClock().monotonic()
        assert result == 123.456

    def test_monotonic_returns_float(self) -> None:
        result = RealClock().monotonic()
        assert isinstance(result, float)
        assert result > 0


class TestRealFilesystem:
    def test_which_returns_path_when_tool_found(self) -> None:
        with patch("fido.infra.shutil.which", return_value="/usr/bin/git"):
            result = RealFilesystem().which("git")
        assert result == "/usr/bin/git"

    def test_which_returns_none_when_tool_absent(self) -> None:
        with patch("fido.infra.shutil.which", return_value=None):
            result = RealFilesystem().which("no-such-tool")
        assert result is None

    def test_is_dir_true_for_existing_directory(self, tmp_path: Path) -> None:
        assert RealFilesystem().is_dir(tmp_path) is True

    def test_is_dir_false_for_nonexistent_path(self, tmp_path: Path) -> None:
        assert RealFilesystem().is_dir(tmp_path / "does-not-exist") is False

    def test_is_dir_false_for_file(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        assert RealFilesystem().is_dir(f) is False


class TestRealOsProcess:
    def test_execvp_delegates_to_os_execvp(self) -> None:
        with patch("fido.infra.os.execvp") as mock_execvp:
            RealOsProcess().execvp("uv", ["uv", "run", "fido"])
        mock_execvp.assert_called_once_with("uv", ["uv", "run", "fido"])

    def test_exit_delegates_to_os_exit(self) -> None:
        with patch("fido.infra.os._exit") as mock_exit:
            RealOsProcess().exit(75)
        mock_exit.assert_called_once_with(75)

    def test_chdir_delegates_to_os_chdir(self, tmp_path: Path) -> None:
        with patch("fido.infra.os.chdir") as mock_chdir:
            RealOsProcess().chdir(tmp_path)
        mock_chdir.assert_called_once_with(tmp_path)

    def test_chdir_accepts_string_path(self) -> None:
        with patch("fido.infra.os.chdir") as mock_chdir:
            RealOsProcess().chdir("/tmp")
        mock_chdir.assert_called_once_with("/tmp")

    def test_install_signal_delegates_to_signal_signal(self) -> None:
        old_handler = MagicMock()
        handler = MagicMock()
        with patch("fido.infra.signal.signal", return_value=old_handler) as mock_sig:
            result = RealOsProcess().install_signal(signal.SIGTERM, handler)
        mock_sig.assert_called_once_with(signal.SIGTERM, handler)
        assert result is old_handler
