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

from even_depth import Even, EvenO, EvenS, Odd, OddS, even_depth

assert even_depth(EvenO()) == 0, "even_depth(EvenO()): got " + repr(even_depth(EvenO()))
assert even_depth(EvenS(OddS(EvenO()))) == 2, "even_depth depth-2: got " + repr(
    even_depth(EvenS(OddS(EvenO())))
)
assert even_depth(EvenS(OddS(EvenS(OddS(EvenO()))))) == 4, (
    "even_depth depth-4: got " + repr(even_depth(EvenS(OddS(EvenS(OddS(EvenO()))))))
)
assert isinstance(EvenO(), Even), "EvenO() must be instance of Even"
assert isinstance(OddS(EvenO()), Odd), "OddS() must be instance of Odd"
print("Phase 4 Even/Odd round-trip: OK")
