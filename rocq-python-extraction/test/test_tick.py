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
