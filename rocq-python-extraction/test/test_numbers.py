from fractions import Fraction

import pytest
from n_case import n_case
from n_seven import n_seven
from nat_pred_or_zero import nat_pred_or_zero
from nat_roundtrip import nat_roundtrip
from nat_three import nat_three
from positive_case import positive_case
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
