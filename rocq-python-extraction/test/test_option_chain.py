from option_chain import option_chain


def test_option_chain_uses_none_short_circuit(build_default) -> None:
    assert option_chain(None) is None
    assert option_chain(4) == 5

    source = (build_default / "option_chain.py").read_text()
    assert "None if __option_value is None else" in source
