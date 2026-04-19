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

from stree_size import DDecl, DEnd, DTree, SLit, SSeq, STree, stree_size

assert stree_size(SLit(42)) == 1, "stree_size(SLit(42)): got " + repr(
    stree_size(SLit(42))
)
assert stree_size(SSeq(DEnd(), SLit(0))) == 1, (
    "stree_size(SSeq(DEnd(), SLit(0))): got " + repr(stree_size(SSeq(DEnd(), SLit(0))))
)
assert stree_size(SSeq(DDecl(SLit(1), DDecl(SLit(2), DEnd())), SLit(3))) == 3, (
    "stree_size nested: got "
    + repr(stree_size(SSeq(DDecl(SLit(1), DDecl(SLit(2), DEnd())), SLit(3))))
)
assert isinstance(SLit(0), STree), "SLit(0) must be instance of STree"
assert isinstance(DEnd(), DTree), "DEnd() must be instance of DTree"
print("Phase 4 STree/DTree round-trip: OK")
