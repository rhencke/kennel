from fractions import Fraction
from pathlib import Path

import pytest
from conftest import RenderedSourceAssert
from n_case import n_case
from n_seven import n_seven
from nat_compare_and import nat_compare_and
from nat_compare_bool_eq import nat_compare_bool_eq
from nat_compare_neg import nat_compare_neg
from nat_compare_neg_lt import nat_compare_neg_lt
from nat_compare_or import nat_compare_or
from nat_pred_or_zero import nat_pred_or_zero
from nat_roundtrip import nat_roundtrip
from nat_three import nat_three
from positive_case import positive_case
from positive_eq import positive_eq
from positive_five import positive_five
from q_den import q_den
from q_half import q_half
from q_num import q_num
from z_neg_three import z_neg_three
from z_sign_code import z_sign_code


def test_nat_positive_n_and_z_are_native_ints() -> None:
    assert nat_three == 3
    assert nat_pred_or_zero(0) == 0
    assert nat_pred_or_zero(5) == 4
    assert nat_roundtrip(7) == 7
    assert positive_five == 5
    assert positive_case(1) == 0
    assert positive_case(4) == 2
    assert positive_case(5) == 1
    assert positive_eq(1, 1) is True
    assert positive_eq(1, 5) is False
    assert nat_compare_and(1, 2, 2) is True
    assert nat_compare_and(2, 1, 3) is False
    assert nat_compare_or(3, 3, 2) is True
    assert nat_compare_or(3, 2, 4) is True
    assert nat_compare_or(3, 2, 1) is False
    assert nat_compare_neg(3, 2) is True
    assert nat_compare_neg(2, 3) is False
    assert nat_compare_neg_lt(3, 3) is True
    assert nat_compare_neg_lt(2, 3) is False
    assert nat_compare_bool_eq(1, 2, True) is True
    assert nat_compare_bool_eq(1, 2, False) is False
    assert n_seven == 7
    assert n_case(0) == 0
    assert n_case(9) == 1
    assert z_neg_three == -3
    assert z_sign_code(0) == 0
    assert z_sign_code(8) == 1
    assert z_sign_code(-8) == -1


def test_numeric_domain_errors_reject_invalid_python_inputs() -> None:
    with pytest.raises(ValueError):
        nat_pred_or_zero(-1)
    with pytest.raises(ValueError):
        positive_case(0)
    with pytest.raises(ValueError):
        n_case(-1)


def test_q_extracts_to_fraction_with_normalized_fields() -> None:
    assert q_half == Fraction(1, 2)
    assert q_num(Fraction(2, 4)) == 1
    assert q_den(Fraction(2, 4)) == 2


def test_positive_equality_lowers_without_pos_protocol(
    build_default: Path,
    assert_rendered_source: RenderedSourceAssert,
) -> None:
    source = (build_default / "positive_eq.py").read_text()

    assert_rendered_source(
        source,
        "return left == right",
        (
            "def eqb(",
            "class Pos_Module",
            "Pos:",
        ),
    )


def test_numeric_stdlib_operation_declarations_are_suppressed(
    build_default: Path,
    assert_rendered_source: RenderedSourceAssert,
) -> None:
    compare_and_source = (build_default / "nat_compare_and.py").read_text()
    compare_or_source = (build_default / "nat_compare_or.py").read_text()

    assert_rendered_source(
        compare_and_source,
        "return left < middle and middle <= right",
        (
            "def andb(",
            "def ltb(",
            "def leb(",
        ),
    )
    assert_rendered_source(
        compare_or_source,
        "return left == middle or middle < right",
        (
            "def orb(",
            "def eqb(",
            "def ltb(",
        ),
    )


def test_numeric_constructor_constants_render_as_literals(
    build_default: Path,
    assert_rendered_source: RenderedSourceAssert,
) -> None:
    assert_rendered_source(
        (build_default / "nat_three.py").read_text(),
        "nat_three: int = 3",
    )
    assert_rendered_source(
        (build_default / "positive_five.py").read_text(),
        "positive_five: int = 5",
    )
    assert_rendered_source(
        (build_default / "n_seven.py").read_text(),
        "n_seven: int = 7",
    )
    assert_rendered_source(
        (build_default / "z_neg_three.py").read_text(),
        "z_neg_three: int = -3",
    )
    assert_rendered_source(
        (build_default / "q_half.py").read_text(),
        "q_half: Fraction = Fraction(1, 2)",
    )


def test_primitive_comparisons_compose_with_bool_ops(
    build_default: Path,
    assert_rendered_source: RenderedSourceAssert,
) -> None:
    compare_and = (build_default / "nat_compare_and.py").read_text()
    compare_or = (build_default / "nat_compare_or.py").read_text()
    compare_neg = (build_default / "nat_compare_neg.py").read_text()
    compare_neg_lt = (build_default / "nat_compare_neg_lt.py").read_text()

    assert_rendered_source(
        compare_and,
        "return left < middle and middle <= right",
        ("return (left < middle) and (middle <= right)",),
    )
    assert_rendered_source(
        compare_or,
        "return left == middle or middle < right",
        ("return (left == middle) or (middle < right)",),
    )
    assert_rendered_source(
        compare_neg,
        "return left > right",
        (
            "return not left <= right",
            "return not (left <= right)",
        ),
    )
    assert_rendered_source(
        compare_neg_lt,
        "return left >= right",
        (
            "return not left < right",
            "return not (left < right)",
        ),
    )


def test_primitive_comparison_as_equality_operand_is_parenthesized(
    build_default: Path,
    assert_rendered_source: RenderedSourceAssert,
) -> None:
    source = (build_default / "nat_compare_bool_eq.py").read_text()

    assert_rendered_source(
        source,
        "return (left < right) == expected",
        ("return left < right == expected",),
    )
