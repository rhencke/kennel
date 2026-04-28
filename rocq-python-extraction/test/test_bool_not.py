from primitives import (
    bool_and,
    bool_and_eq,
    bool_eq,
    bool_eq_and,
    bool_neg,
    bool_neg_and,
    bool_not,
    lambda_call_head,
    list_cons_append,
)


def test_bool_not_round_trip() -> None:
    assert bool_not(True) is False, "bool_not(True): got " + repr(bool_not(True))
    assert bool_not(False) is True, "bool_not(False): got " + repr(bool_not(False))


def test_bool_and_round_trip() -> None:
    assert bool_and(True, True) is True
    assert bool_and(True, False) is False
    assert bool_and(False, True) is False
    assert bool_and(False, False) is False


def test_bool_neg_round_trip() -> None:
    assert bool_neg(True) is False
    assert bool_neg(False) is True


def test_bool_neg_and_round_trip() -> None:
    assert bool_neg_and(True, True) is False
    assert bool_neg_and(True, False) is True
    assert bool_neg_and(False, True) is True
    assert bool_neg_and(False, False) is True


def test_bool_eq_round_trip() -> None:
    assert bool_eq(True, True) is True
    assert bool_eq(True, False) is False
    assert bool_eq(False, True) is False
    assert bool_eq(False, False) is True


def test_bool_eq_and_round_trip() -> None:
    assert bool_eq_and(True, True, True) is True
    assert bool_eq_and(True, True, False) is False
    assert bool_eq_and(True, False, True) is False


def test_bool_and_eq_round_trip() -> None:
    assert bool_and_eq(True, True, True) is True
    assert bool_and_eq(True, False, False) is True
    assert bool_and_eq(False, True, True) is False


def test_list_cons_append_round_trip() -> None:
    assert list_cons_append(1, [2], [3, 4]) == [1, 2, 3, 4]


def test_lambda_call_head_round_trip() -> None:
    assert lambda_call_head(4) == 5


def test_bool_and_lowers_to_native_and(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "return b1 and b2" in source


def test_bool_neg_lowers_to_native_not(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "def negb(" not in source
    assert "return not b" in source


def test_bool_neg_and_preserves_precedence(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "return not (b1 and b2)" in source
    assert "return not b1 and b2" not in source


def test_bool_eq_lowers_to_native_equality(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "def eqb(" not in source
    assert "return b1 == b2" in source


def test_bool_equality_operand_keeps_and_precedence(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "return b1 == b2 and b3" in source


def test_bool_and_operand_is_parenthesized_for_equality(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "return (b1 and b2) == b3" in source
    assert "return b1 and b2 == b3" not in source


def test_list_append_preserves_left_associative_grouping(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "return [h] + left + right" in source


def test_lambda_call_head_is_parenthesized(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "return (lambda f: f(n))(lambda x: x + 1)" in source


def test_remapped_primitives_do_not_emit_unused_typevars(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert '_A = TypeVar("_A")' not in source
    assert '_B = TypeVar("_B")' not in source


def test_remapped_primitives_do_not_emit_section_comments(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "remapped to Python primitive" not in source
