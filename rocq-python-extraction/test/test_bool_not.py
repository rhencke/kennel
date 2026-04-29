from primitives import (
    bool_and,
    bool_and_eq,
    bool_and_or,
    bool_eq,
    bool_eq_and,
    bool_neg,
    bool_neg_and,
    bool_neg_or,
    bool_not,
    bool_or,
    bool_or_and,
    lambda_call_head,
    list_append_left_nested,
    list_append_let_child,
    list_append_match_child,
    list_append_right_nested,
    list_cons_append,
    option_nat_neq,
)


def test_bool_not_round_trip() -> None:
    assert bool_not(True) is False, "bool_not(True): got " + repr(bool_not(True))
    assert bool_not(False) is True, "bool_not(False): got " + repr(bool_not(False))


def test_bool_and_round_trip() -> None:
    assert bool_and(True, True) is True
    assert bool_and(True, False) is False
    assert bool_and(False, True) is False
    assert bool_and(False, False) is False


def test_bool_or_round_trip() -> None:
    assert bool_or(True, True) is True
    assert bool_or(True, False) is True
    assert bool_or(False, True) is True
    assert bool_or(False, False) is False


def test_bool_neg_round_trip() -> None:
    assert bool_neg(True) is False
    assert bool_neg(False) is True


def test_bool_neg_and_round_trip() -> None:
    assert bool_neg_and(True, True) is False
    assert bool_neg_and(True, False) is True
    assert bool_neg_and(False, True) is True
    assert bool_neg_and(False, False) is True


def test_bool_neg_or_round_trip() -> None:
    assert bool_neg_or(True, True) is False
    assert bool_neg_or(True, False) is False
    assert bool_neg_or(False, True) is False
    assert bool_neg_or(False, False) is True


def test_bool_or_and_round_trip() -> None:
    assert bool_or_and(False, True, True) is True
    assert bool_or_and(False, True, False) is False
    assert bool_or_and(True, False, False) is True


def test_bool_and_or_round_trip() -> None:
    assert bool_and_or(True, False, True) is True
    assert bool_and_or(True, False, False) is False
    assert bool_and_or(False, True, True) is False


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


def test_option_nat_neq_round_trip() -> None:
    assert option_nat_neq(None, None) is False
    assert option_nat_neq(None, 1) is True
    assert option_nat_neq(1, None) is True
    assert option_nat_neq(1, 1) is False
    assert option_nat_neq(1, 2) is True


def test_list_cons_append_round_trip() -> None:
    assert list_cons_append(1, [2], [3, 4]) == [1, 2, 3, 4]


def test_nested_list_append_round_trip() -> None:
    assert list_append_left_nested([1], [2], [3]) == [1, 2, 3]
    assert list_append_right_nested([1], [2], [3]) == [1, 2, 3]


def test_low_precedence_list_append_children_round_trip() -> None:
    assert list_append_let_child(1, [2, 3]) == [1, 2, 3]
    assert list_append_match_child(True, [2, 3]) == [0, 2, 3]
    assert list_append_match_child(False, [2, 3]) == [1, 2, 3]


def test_lambda_call_head_round_trip() -> None:
    assert lambda_call_head(4) == 5


def test_bool_and_lowers_to_native_and(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "return b1 and b2" in source


def test_bool_or_lowers_to_native_or(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "def orb(" not in source
    assert "return b1 or b2" in source


def test_bool_neg_lowers_to_native_not(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "def negb(" not in source
    assert "return not b" in source


def test_bool_neg_and_preserves_precedence(
    build_default,
    assert_rendered_source,
) -> None:
    source = (build_default / "primitives.py").read_text()

    assert_rendered_source(
        source,
        "return not (b1 and b2)",
        ("return not b1 and b2",),
    )


def test_bool_neg_or_preserves_precedence(
    build_default,
    assert_rendered_source,
) -> None:
    source = (build_default / "primitives.py").read_text()

    assert_rendered_source(
        source,
        "return not (b1 or b2)",
        ("return not b1 or b2",),
    )


def test_bool_or_and_keeps_python_precedence(
    build_default,
    assert_rendered_source,
) -> None:
    source = (build_default / "primitives.py").read_text()

    assert_rendered_source(
        source,
        "return b1 or b2 and b3",
        ("return b1 or (b2 and b3)",),
    )


def test_bool_and_or_parenthesizes_or_operand(
    build_default,
    assert_rendered_source,
) -> None:
    source = (build_default / "primitives.py").read_text()

    assert_rendered_source(
        source,
        "return b1 and (b2 or b3)",
        ("return b1 and b2 or b3",),
    )


def test_bool_eq_lowers_to_native_equality(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "def eqb(" not in source
    assert "return b1 == b2" in source


def test_bool_lowering_helpers_are_suppressed(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    for helper in ("andb", "orb", "negb", "eqb"):
        assert f"def {helper}(" not in source


def test_bool_equality_operand_keeps_and_precedence(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "return b1 == b2 and b3" in source


def test_bool_and_operand_is_parenthesized_for_equality(
    build_default,
    assert_rendered_source,
) -> None:
    source = (build_default / "primitives.py").read_text()

    assert_rendered_source(
        source,
        "return (b1 and b2) == b3",
        ("return b1 and b2 == b3",),
    )


def test_option_nat_neq_lowers_to_direct_comparison(
    build_default,
    assert_rendered_source,
) -> None:
    source = (build_default / "primitives.py").read_text()

    assert_rendered_source(
        source,
        "return left != right",
        (
            "return not (left == right)",
            "if __option is None",
        ),
    )


def test_list_append_preserves_left_associative_grouping(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "return [h] + left + right" in source


def test_nested_list_append_expressions_stay_flat(
    build_default,
    assert_rendered_source,
) -> None:
    source = (build_default / "primitives.py").read_text()

    assert_rendered_source(
        source,
        "return left + middle + right",
        (
            "return (left + middle) + right",
            "return left + (middle + right)",
        ),
    )


def test_list_append_low_precedence_children_are_parenthesized(
    build_default,
    assert_rendered_source,
) -> None:
    source = (build_default / "primitives.py").read_text()

    assert_rendered_source(
        source,
        "return (lambda prefix: prefix)([h] + []) + right",
    )
    assert_rendered_source(
        source,
        "return ([0] + [] if flag else [0 + 1] + []) + right",
        ("return [0] + [] if flag else [0 + 1] + [] + right",),
    )


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
