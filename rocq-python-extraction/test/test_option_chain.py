from pathlib import Path

from option_chain import option_chain
from option_chain_twice import option_chain_twice


def test_option_chain_uses_none_short_circuit(build_default: Path) -> None:
    assert option_chain(None) is None
    assert option_chain(4) == 5

    source = (build_default / "option_chain.py").read_text()
    assert "None if __option_value is None else" in source
    assert "__PYMONAD_OPTION_BIND__" not in source


def test_nested_option_bind_child_preserves_precedence(build_default: Path) -> None:
    assert option_chain_twice(None) is None
    assert option_chain_twice(4) == 6

    source = (build_default / "option_chain_twice.py").read_text()
    assert "None if __option_value is None else" in source
    assert ")(__option_value))((lambda __option_value:" in source
    assert "__PYMONAD_OPTION_BIND__" not in source
