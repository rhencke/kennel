"""Tests for fido.__main__ — the ``python -m fido`` entry point."""

from pathlib import Path


def test_main_module_calls_main() -> None:
    """``python -m fido`` executes ``main()`` from ``fido.main``.

    The guard (``if __name__ == "__main__"``) prevents ``main()`` from
    firing on import, so this test can safely import the module to cover
    line 1 and then verify the structural contract via source-text assertion.
    """
    import fido.__main__  # noqa: F401 — import covers line 1; guard prevents server start
    from fido.main import main

    assert fido.__main__.main is main  # right symbol was imported

    source = Path(__file__).resolve().parent.parent / "src" / "fido" / "__main__.py"
    text = source.read_text()
    assert "from fido.main import main" in text
    assert 'if __name__ == "__main__":' in text
    assert "main()" in text
