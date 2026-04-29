from tick import StateT, tick


def test_tick_extracts_to_state_class(build_default) -> None:
    assert isinstance(tick, StateT)
    assert tick.run(41) == 41
    assert tick.state == 42
    assert tick.run_with_state(10) == (10, 11)
    assert tick.state == 11

    source = (build_default / "tick.py").read_text()
    assert "class StateT" in source
    assert "def get_state" in source
    assert "def put_state" in source
    assert ".bind(" in source


def test_state_monad_marker_calls_lower_to_runtime_methods(build_default) -> None:
    source = (build_default / "tick.py").read_text()

    for snippet in (
        "StateT.get_state()",
        "StateT.put_state(n + 1)",
        "StateT.pure(n)",
        ".bind(",
    ):
        assert snippet in source
    assert "__PYMONAD_STATE_" not in source
