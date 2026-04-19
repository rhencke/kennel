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

from is_even import is_even

assert is_even(0) is True, "is_even(0): got " + repr(is_even(0))
assert is_even(1) is False, "is_even(1): got " + repr(is_even(1))
assert is_even(2) is True, "is_even(2): got " + repr(is_even(2))
assert is_even(3) is False, "is_even(3): got " + repr(is_even(3))
assert is_even(4) is True, "is_even(4): got " + repr(is_even(4))
print("Phase 4 is_even round-trip: OK")
