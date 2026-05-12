"""Tests for fido.infra — infrastructure port protocols and real implementations."""

import signal
import subprocess
from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock

import pytest

from fido.infra import RealClock, RealFilesystem, RealOsProcess, RealProcessRunner


class TestRealProcessRunner:
    def test_run_returns_completed_process(self) -> None:
        result = RealProcessRunner().run(["true"])
        assert result.returncode == 0

    def test_run_forwards_kwargs_to_subprocess(self) -> None:
        result = RealProcessRunner().run(["true"], capture_output=True, text=True)
        assert result.returncode == 0
        assert result.stdout == ""

    def test_run_propagates_called_process_error(self) -> None:
        with pytest.raises(subprocess.CalledProcessError):
            RealProcessRunner().run(["false"])


class TestRealClock:
    def test_sleep_accepts_float_seconds(self) -> None:
        # Verify sleep delegates without raising; zero duration keeps tests fast.
        RealClock().sleep(0.0)

    def test_monotonic_returns_increasing_values(self) -> None:
        t1 = RealClock().monotonic()
        t2 = RealClock().monotonic()
        assert t2 >= t1

    def test_monotonic_returns_float(self) -> None:
        result = RealClock().monotonic()
        assert isinstance(result, float)
        assert result > 0


class TestRealFilesystem:
    def test_which_returns_path_when_tool_found(self) -> None:
        result = RealFilesystem().which("true")
        assert result is not None
        assert "true" in result

    def test_which_returns_none_when_tool_absent(self) -> None:
        result = RealFilesystem().which("no-such-tool-xyzzy-42")
        assert result is None

    def test_is_dir_true_for_existing_directory(self, tmp_path: Path) -> None:
        assert RealFilesystem().is_dir(tmp_path) is True

    def test_is_dir_false_for_nonexistent_path(self, tmp_path: Path) -> None:
        assert RealFilesystem().is_dir(tmp_path / "does-not-exist") is False

    def test_is_dir_false_for_file(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        assert RealFilesystem().is_dir(f) is False


def _make_os_proc(**overrides: object) -> RealOsProcess:
    """Construct a RealOsProcess with MagicMock defaults for all four callables."""

    def _never_return(_code: int) -> NoReturn:
        raise AssertionError("unexpected exit call")

    kwargs: dict[str, object] = {
        "_execvp": MagicMock(),
        "_exit": _never_return,
        "_chdir": MagicMock(),
        "_install_signal": MagicMock(),
    }
    kwargs.update(overrides)
    return RealOsProcess(**kwargs)  # type: ignore[arg-type]


class TestRealOsProcess:
    def test_execvp_delegates_to_os_execvp(self) -> None:
        mock_execvp = MagicMock()
        _make_os_proc(_execvp=mock_execvp).execvp("uv", ["uv", "run", "fido"])
        mock_execvp.assert_called_once_with("uv", ["uv", "run", "fido"])

    def test_exit_delegates_to_os_exit(self) -> None:
        calls: list[int] = []

        def _fake_exit(code: int) -> NoReturn:
            calls.append(code)
            raise SystemExit(code)

        with pytest.raises(SystemExit) as exc:
            _make_os_proc(_exit=_fake_exit).exit(75)
        assert exc.value.code == 75
        assert calls == [75]

    def test_chdir_delegates_to_os_chdir(self, tmp_path: Path) -> None:
        mock_chdir = MagicMock()
        _make_os_proc(_chdir=mock_chdir).chdir(tmp_path)
        mock_chdir.assert_called_once_with(tmp_path)

    def test_chdir_accepts_string_path(self) -> None:
        mock_chdir = MagicMock()
        _make_os_proc(_chdir=mock_chdir).chdir("/tmp")
        mock_chdir.assert_called_once_with("/tmp")

    def test_install_signal_returns_previous_handler(self) -> None:
        old_handler = MagicMock()
        mock_install = MagicMock(return_value=old_handler)
        new_handler = MagicMock()
        result = _make_os_proc(_install_signal=mock_install).install_signal(
            signal.SIGTERM, new_handler
        )
        mock_install.assert_called_once_with(signal.SIGTERM, new_handler)
        assert result is old_handler
