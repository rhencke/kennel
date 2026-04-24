"""Tests for fido.chat — interactive claude session launcher."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fido.chat import main as chat_main
from fido.chat import run


class TestChat:
    def _run(
        self,
        argv: list[str],
        persona_text: str = "You are Fido.",
        *,
        tmp_path: Path,
        extra_kw: dict | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        """Run fido.chat.run with a real persona file and mocked side-effects.

        Returns (chdir_mock, execvp_mock).
        """
        persona_file = tmp_path / "persona.md"
        persona_file.write_text(persona_text)
        runner_clone = tmp_path / "home-runner"

        chdir = MagicMock()
        execvp = MagicMock()
        environ: dict[str, str] = {}

        kw: dict = dict(
            persona_file=persona_file,
            runner_clone=runner_clone,
            chdir=chdir,
            execvp=execvp,
            environ=environ,
        )
        if extra_kw:
            kw.update(extra_kw)

        run(argv, **kw)
        return chdir, execvp

    def test_missing_persona_exits_1(self, tmp_path: Path, capsys) -> None:
        missing = tmp_path / "nope.md"
        with pytest.raises(SystemExit) as exc:
            run(
                [],
                persona_file=missing,
                runner_clone=tmp_path / "home-runner",
                chdir=MagicMock(),
                execvp=MagicMock(),
                environ={},
            )
        assert exc.value.code == 1
        assert "persona file not found" in capsys.readouterr().err

    def test_missing_persona_message_contains_path(
        self, tmp_path: Path, capsys
    ) -> None:
        missing = tmp_path / "sub" / "persona.md"
        with pytest.raises(SystemExit):
            run(
                [],
                persona_file=missing,
                runner_clone=tmp_path / "home-runner",
                chdir=MagicMock(),
                execvp=MagicMock(),
                environ={},
            )
        assert str(missing) in capsys.readouterr().err

    def test_no_args_defaults_to_remote_control(self, tmp_path: Path) -> None:
        _, execvp = self._run([], tmp_path=tmp_path)
        cmd = execvp.call_args[0][1]
        assert "/remote-control" in cmd

    def test_explicit_args_override_default(self, tmp_path: Path) -> None:
        _, execvp = self._run(["/foo", "bar"], tmp_path=tmp_path)
        cmd = execvp.call_args[0][1]
        assert "/foo" in cmd
        assert "bar" in cmd
        assert "/remote-control" not in cmd

    def test_execvp_program_is_nice(self, tmp_path: Path) -> None:
        _, execvp = self._run([], tmp_path=tmp_path)
        assert execvp.call_args[0][0] == "nice"

    def test_execvp_argv0_is_nice(self, tmp_path: Path) -> None:
        _, execvp = self._run([], tmp_path=tmp_path)
        cmd = execvp.call_args[0][1]
        assert cmd[0] == "nice"

    def test_execvp_includes_nice_priority(self, tmp_path: Path) -> None:
        _, execvp = self._run([], tmp_path=tmp_path)
        cmd = execvp.call_args[0][1]
        assert "-n19" in cmd

    def test_execvp_runs_claude(self, tmp_path: Path) -> None:
        _, execvp = self._run([], tmp_path=tmp_path)
        cmd = execvp.call_args[0][1]
        assert "claude" in cmd

    def test_execvp_passes_permission_mode(self, tmp_path: Path) -> None:
        _, execvp = self._run([], tmp_path=tmp_path)
        cmd = execvp.call_args[0][1]
        assert "--permission-mode=bypassPermissions" in cmd

    def test_execvp_passes_continue(self, tmp_path: Path) -> None:
        _, execvp = self._run([], tmp_path=tmp_path)
        cmd = execvp.call_args[0][1]
        assert "--continue" in cmd

    def test_execvp_passes_persona_as_append_system_prompt(
        self, tmp_path: Path
    ) -> None:
        persona_text = "I am a good dog."
        _, execvp = self._run([], persona_text=persona_text, tmp_path=tmp_path)
        cmd = execvp.call_args[0][1]
        idx = cmd.index("--append-system-prompt")
        assert cmd[idx + 1] == persona_text

    def test_chdir_called_with_runner_clone(self, tmp_path: Path) -> None:
        runner_clone = tmp_path / "home-runner"
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("woof")
        chdir = MagicMock()
        run(
            [],
            persona_file=persona_file,
            runner_clone=runner_clone,
            chdir=chdir,
            execvp=MagicMock(),
            environ={},
        )
        chdir.assert_called_once_with(runner_clone)

    def test_environ_no_flicker_set(self, tmp_path: Path) -> None:
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("woof")
        environ: dict[str, str] = {}
        run(
            [],
            persona_file=persona_file,
            runner_clone=tmp_path / "home-runner",
            chdir=MagicMock(),
            execvp=MagicMock(),
            environ=environ,
        )
        assert environ["CLAUDE_CODE_NO_FLICKER"] == "1"

    def test_environ_defaults_to_os_environ(self, tmp_path: Path) -> None:
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("woof")
        with patch("os.environ", {}) as mock_env:
            run(
                [],
                persona_file=persona_file,
                runner_clone=tmp_path / "home-runner",
                chdir=MagicMock(),
                execvp=MagicMock(),
            )
            assert mock_env["CLAUDE_CODE_NO_FLICKER"] == "1"

    def test_main_passes_argv_to_run(self) -> None:
        with patch("fido.chat.run") as mock_run:
            chat_main(["hello"])
        mock_run.assert_called_once_with(["hello"])


class TestChatViaMain:
    def test_chat_subcommand_dispatches(self, tmp_path: Path) -> None:
        """'fido chat' should delegate to fido.chat.run."""
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("woof")

        with patch("fido.chat.main") as mock_main:
            from fido.main import main

            main(["chat", "hello"])

        mock_main.assert_called_once_with(["hello"])

    def test_chat_subcommand_no_extra_args(self, tmp_path: Path) -> None:
        """'fido chat' with no extra args passes empty list."""
        with patch("fido.chat.main") as mock_main:
            from fido.main import main

            main(["chat"])

        mock_main.assert_called_once_with([])
