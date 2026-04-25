import py_compile


def test_core_terms_syntax(build_default) -> None:
    py_compile.compile(str(build_default / "core_terms.py"), doraise=True)


def test_unsupported_dtype_comments_are_catalogued(build_default) -> None:
    source = (build_default / "core_terms.py").read_text()

    assert "Python ExtractionDiagnostic [PYEX003]" in source
    assert "Remediation:" in source


def test_primitive_string_type_is_bytes(build_default) -> None:
    source = (build_default / "core_terms.py").read_text()

    assert 'str_val: bytes = b"hello"' in source
