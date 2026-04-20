# ruff: noqa: E402
from test_support import add_build_default_to_syspath

add_build_default_to_syspath()

from expr_is_num_pair import (
    EAdd,
    ELift,
    ENum,
    Expr,
    Val,
    VNum,
    VPair,
    expr_is_num_pair,
)


def test_expr_is_num_pair_round_trip() -> None:
    assert expr_is_num_pair(ELift(VPair(VNum(1), VNum(2)))) is True, (
        "ELift(VPair(VNum(1), VNum(2))): got "
        + repr(expr_is_num_pair(ELift(VPair(VNum(1), VNum(2)))))
    )
    assert expr_is_num_pair(ENum(42)) is False, "ENum(42): got " + repr(
        expr_is_num_pair(ENum(42))
    )
    assert expr_is_num_pair(ELift(VNum(7))) is False, "ELift(VNum(7)): got " + repr(
        expr_is_num_pair(ELift(VNum(7)))
    )
    assert expr_is_num_pair(ELift(VPair(VPair(VNum(1), VNum(2)), VNum(3)))) is False, (
        "ELift(VPair(VPair(...), VNum(...))): got "
        + repr(expr_is_num_pair(ELift(VPair(VPair(VNum(1), VNum(2)), VNum(3)))))
    )
    assert expr_is_num_pair(EAdd(ENum(1), ENum(2))) is False, (
        "EAdd(ENum(1), ENum(2)): got " + repr(expr_is_num_pair(EAdd(ENum(1), ENum(2))))
    )
    assert isinstance(ENum(0), Expr), "ENum(0) must be instance of Expr"
    assert isinstance(VNum(0), Val), "VNum(0) must be instance of Val"
    assert isinstance(ELift(VNum(0)), Expr), "ELift(VNum(0)) must be instance of Expr"
