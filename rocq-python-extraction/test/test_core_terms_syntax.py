import py_compile


def test_core_terms_syntax(build_default) -> None:
    for filename in [
        "nat_add.py",
        "mk_pair_r.py",
        "zeros.py",
        "uint_val.py",
        "float_val.py",
        "str_val.py",
        "todo_val.py",
    ]:
        py_compile.compile(str(build_default / filename), doraise=True)


def test_unsupported_dtype_comments_are_catalogued(build_default) -> None:
    for filename in ["uint_val.py", "float_val.py"]:
        source = (build_default / filename).read_text()

        assert "Python ExtractionDiagnostic [PYEX003]" in source
        assert "Remediation:" in source


def test_primitive_string_type_is_bytes(build_default) -> None:
    source = (build_default / "str_val.py").read_text()

    assert 'str_val: bytes = b"hello"' in source
