from pathlib import Path


def test_main_calls_main() -> None:
    """Verify that fido/__main__.py imports and calls main().

    Source-text assertion so the guard can never silently rot — importing
    the module is a side-effect that is hard to test without patching, but
    verifying the call is present in the source is cheap and reliable.
    """
    source = Path(__file__).resolve().parent.parent / "src" / "fido" / "__main__.py"
    text = source.read_text()
    assert "from fido.main import main" in text, "__main__ must import main"
    assert "main()" in text, "__main__ must call main()"
