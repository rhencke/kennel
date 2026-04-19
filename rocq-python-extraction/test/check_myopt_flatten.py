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

from myopt_flatten import MyNone, MySome, myopt_flatten

r1 = myopt_flatten(MyNone())
assert isinstance(r1, MyNone), "myopt_flatten(MyNone()): got " + repr(r1)
r2 = myopt_flatten(MySome(MyNone()))
assert isinstance(r2, MyNone), "myopt_flatten(MySome(MyNone())): got " + repr(r2)
r3 = myopt_flatten(MySome(MySome(42)))
assert isinstance(r3, MySome) and r3.arg0 == 42, (
    "myopt_flatten(MySome(MySome(42))): got " + repr(r3)
)
print("Phase 4 MyOpt round-trip: OK")
