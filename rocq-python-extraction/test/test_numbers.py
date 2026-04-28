from fractions import Fraction

import pytest
from n_case import n_case
from n_seven import n_seven
from nat_compare_and import nat_compare_and
from nat_compare_bool_eq import nat_compare_bool_eq
from nat_compare_neg import nat_compare_neg
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


def test_positive_equality_lowers_without_pos_protocol(build_default) -> None:
    source = (build_default / "positive_eq.py").read_text()

    assert "class Pos_Module" not in source
    assert "Pos:" not in source
    assert "return left == right" in source


def test_primitive_comparisons_compose_with_bool_ops(build_default) -> None:
    compare_and = (build_default / "nat_compare_and.py").read_text()
    compare_or = (build_default / "nat_compare_or.py").read_text()
    compare_neg = (build_default / "nat_compare_neg.py").read_text()

    assert "return left < middle and middle <= right" in compare_and
    assert "return (left < middle) and (middle <= right)" not in compare_and
    assert "return left == middle or middle < right" in compare_or
    assert "return (left == middle) or (middle < right)" not in compare_or
    assert "return not (left <= right)" in compare_neg
    assert "return not left <= right" not in compare_neg


def test_primitive_comparison_as_equality_operand_is_parenthesized(
    build_default,
) -> None:
    source = (build_default / "nat_compare_bool_eq.py").read_text()

    assert "return (left < right) == expected" in source
    assert "return left < right == expected" not in source
