"""Tests for fido.infra — infrastructure port protocols and real implementations."""

import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest

from fido.infra import RealClock, RealFilesystem, RealOsProcess, RealProcessRunner


class TestRealProcessRunner:
    def test_run_returns_completed_process(self) -> None:
        """run() delegates to subprocess and returns the CompletedProcess result."""
        runner = RealProcessRunner()
        result = runner.run(
            ["/bin/sh", "-c", "exit 0"],
            capture_output=True,
            text=True,
        )
        assert isinstance(result, subprocess.CompletedProcess)
        assert result.returncode == 0

    def test_run_forwards_kwargs_to_subprocess(self) -> None:
        """env kwarg is forwarded — the subprocess receives the provided environment."""
        runner = RealProcessRunner()
        result = runner.run(
            ["/bin/sh", "-c", "echo $FOO"],
            env={"FOO": "bar", "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
        assert result.stdout == "bar\n"

    def test_run_propagates_called_process_error(self) -> None:
        with pytest.raises(subprocess.CalledProcessError):
            RealProcessRunner().run(["/bin/sh", "-c", "exit 1"], check=True)


class TestRealClock:
    def test_sleep_returns_without_error(self) -> None:
        """sleep(0) completes immediately without blocking."""
        RealClock().sleep(0)

    def test_monotonic_returns_float(self) -> None:
        result = RealClock().monotonic()
        assert isinstance(result, float)
        assert result > 0

    def test_monotonic_returns_current_time(self) -> None:
        """monotonic() returns the actual time.monotonic() value."""
        t1 = time.monotonic()
        result = RealClock().monotonic()
        t2 = time.monotonic()
        assert t1 <= result <= t2


class TestRealFilesystem:
    def test_which_returns_path_when_tool_found(self) -> None:
        result = RealFilesystem().which("sh")
        assert result is not None
        assert result.endswith("sh")

    def test_which_returns_none_when_tool_absent(self) -> None:
        result = RealFilesystem().which("no-such-tool-xyz-abc-123-fido")
        assert result is None

    def test_is_dir_true_for_existing_directory(self, tmp_path: Path) -> None:
        assert RealFilesystem().is_dir(tmp_path) is True

    def test_is_dir_false_for_nonexistent_path(self, tmp_path: Path) -> None:
        assert RealFilesystem().is_dir(tmp_path / "does-not-exist") is False

    def test_is_dir_false_for_file(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        assert RealFilesystem().is_dir(f) is False


class _FakeOsBackend:
    """Typed fake for _OsBackend that records calls and simulates OS operations.

    exit() raises SystemExit to satisfy the NoReturn contract while still
    allowing tests to verify the call and inspect the exit code.
    """

    def __init__(self) -> None:
        self.execvp_calls: list[tuple[str, list[str]]] = []
        self.exit_calls: list[int] = []
        self.chdir_calls: list[Path | str] = []
        self.signal_calls: list[tuple[int, Callable[..., object]]] = []
        self._signal_return: object = None

    def execvp(self, file: str, args: list[str]) -> None:
        self.execvp_calls.append((file, args))

    def exit(self, code: int) -> NoReturn:
        self.exit_calls.append(code)
        raise SystemExit(code)

    def chdir(self, path: Path | str) -> None:
        self.chdir_calls.append(path)

    def signal(self, signum: int, handler: Callable[..., object]) -> object:
        self.signal_calls.append((signum, handler))
        return self._signal_return


class TestRealOsProcess:
    def test_execvp_delegates_to_os_execvp(self) -> None:
        backend = _FakeOsBackend()
        RealOsProcess(backend).execvp("uv", ["uv", "run", "fido"])
        assert backend.execvp_calls == [("uv", ["uv", "run", "fido"])]

    def test_exit_delegates_to_os_exit(self) -> None:
        backend = _FakeOsBackend()
        with pytest.raises(SystemExit) as exc:
            RealOsProcess(backend).exit(75)
        assert backend.exit_calls == [75]
        assert exc.value.code == 75

    def test_chdir_delegates_to_os_chdir(self, tmp_path: Path) -> None:
        backend = _FakeOsBackend()
        RealOsProcess(backend).chdir(tmp_path)
        assert backend.chdir_calls == [tmp_path]

    def test_chdir_accepts_string_path(self) -> None:
        backend = _FakeOsBackend()
        RealOsProcess(backend).chdir("/tmp")
        assert backend.chdir_calls == ["/tmp"]

    def test_install_signal_delegates_to_signal_signal(self) -> None:
        old_handler_sentinel = object()
        backend = _FakeOsBackend()
        backend._signal_return = old_handler_sentinel

        def my_handler(signum: int, frame: object) -> None:
            pass

        result = RealOsProcess(backend).install_signal(signal.SIGTERM, my_handler)
        assert backend.signal_calls == [(signal.SIGTERM, my_handler)]
        assert result is old_handler_sentinel
