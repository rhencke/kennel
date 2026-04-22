from __future__ import annotations

from fido.fido_help import main


def test_help_lists_commands(capsys) -> None:  # type: ignore[no-untyped-def]
    main()

    out = capsys.readouterr().out
    removed_flag = "--detach"
    assert "usage: ./fido <command>" in out
    assert "up" in out
    assert removed_flag not in out
    assert "down" in out
    assert "gen-workflows" in out
    assert "make-rocq" in out
    assert "repl" in out
    assert "status" in out
    assert "task" in out
