from __future__ import annotations

from kennel.fido_help import main


def test_help_lists_commands(capsys) -> None:  # type: ignore[no-untyped-def]
    main()

    out = capsys.readouterr().out
    assert "usage: ./fido <command>" in out
    assert "up [--detach]" in out
    assert "down" in out
    assert "status" in out
    assert "task" in out
