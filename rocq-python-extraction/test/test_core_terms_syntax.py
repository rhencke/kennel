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
