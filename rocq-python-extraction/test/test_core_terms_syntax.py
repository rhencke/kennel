import py_compile
from pathlib import Path


def test_core_terms_syntax(build_default: Path) -> None:
    py_compile.compile(str(build_default / "core_terms.py"), doraise=True)


def test_unsupported_dtype_comments_are_catalogued(build_default: Path) -> None:
    source = (build_default / "core_terms.py").read_text()

    assert "Python ExtractionDiagnostic [PYEX003]" in source
    assert "Remediation:" in source


def test_primitive_string_type_is_bytes(build_default: Path) -> None:
    source = (build_default / "core_terms.py").read_text()

    assert 'str_val: bytes = b"hello"' in source
