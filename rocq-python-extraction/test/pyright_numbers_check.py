from fractions import Fraction
from typing import assert_type

from n_case import n_case
from nat_pred_or_zero import nat_pred_or_zero
from positive_case import positive_case
from q_den import q_den
from q_half import q_half
from q_num import q_num
from z_sign_code import z_sign_code

assert_type(nat_pred_or_zero(3), int)
assert_type(positive_case(5), int)
assert_type(n_case(2), int)
assert_type(z_sign_code(-4), int)
assert_type(q_half, Fraction)
assert_type(q_num(Fraction(2, 4)), int)
assert_type(q_den(Fraction(2, 4)), int)
