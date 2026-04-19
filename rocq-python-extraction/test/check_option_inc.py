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

from option_inc import option_inc

assert option_inc(None) is None, "option_inc(None): got " + repr(option_inc(None))
assert option_inc(0) == 1, "option_inc(0): got " + repr(option_inc(0))
assert option_inc(5) == 6, "option_inc(5): got " + repr(option_inc(5))
print("Phase 3 option round-trip: OK")
