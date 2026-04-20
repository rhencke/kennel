# ruff: noqa: E402
import py_compile

from test_support import add_build_default_to_syspath, run_as_script

build_default = add_build_default_to_syspath()


def test_core_terms_syntax() -> None:
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


if __name__ == "__main__":
    run_as_script(
        test_core_terms_syntax,
        "Core-terms extracted .py files are syntactically valid Python.",
    )
