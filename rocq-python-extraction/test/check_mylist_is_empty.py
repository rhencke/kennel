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

from mylist_is_empty import MCons, MNil, mylist_is_empty

assert mylist_is_empty(MNil()) is True, "mylist_is_empty(MNil()): got " + repr(
    mylist_is_empty(MNil())
)
assert mylist_is_empty(MCons(1, MNil())) is False, (
    "mylist_is_empty(MCons(...)): got " + repr(mylist_is_empty(MCons(1, MNil())))
)
print("Phase 4 MyList round-trip: OK")
