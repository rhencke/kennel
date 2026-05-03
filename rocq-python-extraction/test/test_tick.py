from pathlib import Path

from tick import StateT, tick


def test_tick_extracts_to_state_class(build_default: Path) -> None:
    assert isinstance(tick, StateT)
    assert tick.run(41) == 41
    assert tick.state == 42
    assert tick.run_with_state(10) == (10, 11)
    assert tick.state == 11

    # ``StateT`` lives in ``fido.rocq_runtime`` and is re-exported into
    # the generated module via ``from fido.rocq_runtime import *``.  The
    # generated tick.py just imports + uses it — the class definition
    # itself sits in the runtime file.
    source = (build_default / "tick.py").read_text()
    assert "from fido.rocq_runtime import *" in source
    assert ".bind(" in source

    from pathlib import Path as _Path

    runtime = (
        _Path(__file__).resolve().parents[2] / "src" / "fido" / "rocq_runtime.py"
    ).read_text()
    assert "class StateT" in runtime
    assert "def get_state" in runtime
    assert "def put_state" in runtime


def test_state_monad_marker_calls_lower_to_runtime_methods(build_default: Path) -> None:
    source = (build_default / "tick.py").read_text()

    for snippet in (
        "StateT.get_state()",
        "StateT.put_state(n + 1)",
        "StateT.pure(n)",
        ".bind(",
    ):
        assert snippet in source
    assert "__PYMONAD_STATE_" not in source
