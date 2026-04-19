# ruff: noqa: E402
import os
import sys

# The extracted .py files always land in the dune workspace build root (_build/default/).
# Walk up from __file__ to find it — works whether or not dune-workspace is present.
_d = os.path.dirname(os.path.abspath(__file__))
while not (
    os.path.basename(_d) == "default"
    and os.path.basename(os.path.dirname(_d)) == "_build"
):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
del _d

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

# The target pattern: ELift(VPair(VNum(_), VNum(_))) → True
assert expr_is_num_pair(ELift(VPair(VNum(1), VNum(2)))) is True, (
    "ELift(VPair(VNum(1), VNum(2))): got "
    + repr(expr_is_num_pair(ELift(VPair(VNum(1), VNum(2)))))
)

# Plain ENum — not a num pair
assert expr_is_num_pair(ENum(42)) is False, "ENum(42): got " + repr(
    expr_is_num_pair(ENum(42))
)

# ELift of a bare VNum — not a pair
assert expr_is_num_pair(ELift(VNum(7))) is False, "ELift(VNum(7)): got " + repr(
    expr_is_num_pair(ELift(VNum(7)))
)

# ELift of a nested VPair — not flat num pair
assert expr_is_num_pair(ELift(VPair(VPair(VNum(1), VNum(2)), VNum(3)))) is False, (
    "ELift(VPair(VPair(...), VNum(...))): got "
    + repr(expr_is_num_pair(ELift(VPair(VPair(VNum(1), VNum(2)), VNum(3)))))
)

# EAdd — not a num pair
assert expr_is_num_pair(EAdd(ENum(1), ENum(2))) is False, (
    "EAdd(ENum(1), ENum(2)): got " + repr(expr_is_num_pair(EAdd(ENum(1), ENum(2))))
)

# Type sanity
assert isinstance(ENum(0), Expr), "ENum(0) must be instance of Expr"
assert isinstance(VNum(0), Val), "VNum(0) must be instance of Val"
assert isinstance(ELift(VNum(0)), Expr), "ELift(VNum(0)) must be instance of Expr"

print("Phase 5 expr_is_num_pair round-trip: OK")
