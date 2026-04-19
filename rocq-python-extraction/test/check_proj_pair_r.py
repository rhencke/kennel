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

# Each extracted .py file is a self-contained module with its own copy of the
# MkPairR dataclass.  We import each function from its own module and build
# test instances from the same module so that isinstance checks stay consistent
# within each boundary.

from proj_first import MkPairR as MkPairR_pf
from proj_first import proj_first
from proj_second import MkPairR as MkPairR_ps
from proj_second import proj_second
from swap_pair_r import MkPairR as MkPairR_sw
from swap_pair_r import swap_pair_r

# --- proj_first ---
p1 = MkPairR_pf(pfst_r=3, psnd_r=7)
assert proj_first(p1) == 3, "proj_first(3, 7): got " + repr(proj_first(p1))

p2 = MkPairR_pf(pfst_r=0, psnd_r=99)
assert proj_first(p2) == 0, "proj_first(0, 99): got " + repr(proj_first(p2))

# --- proj_second ---
q1 = MkPairR_ps(pfst_r=3, psnd_r=7)
assert proj_second(q1) == 7, "proj_second(3, 7): got " + repr(proj_second(q1))

q2 = MkPairR_ps(pfst_r=0, psnd_r=99)
assert proj_second(q2) == 99, "proj_second(0, 99): got " + repr(proj_second(q2))

# Field names are accessible as attributes (acceptance criterion: named fields,
# not positional _0/_1/…)
assert p1.pfst_r == 3, "p1.pfst_r != 3"
assert p1.psnd_r == 7, "p1.psnd_r != 7"

# --- swap_pair_r ---
s1 = MkPairR_sw(pfst_r=3, psnd_r=7)
swapped = swap_pair_r(s1)
assert swapped.pfst_r == 7, "swap(3,7).pfst_r: got " + repr(swapped.pfst_r)
assert swapped.psnd_r == 3, "swap(3,7).psnd_r: got " + repr(swapped.psnd_r)

# swap is its own inverse
double_swap = swap_pair_r(swapped)
assert double_swap.pfst_r == s1.pfst_r, "double-swap pfst_r"
assert double_swap.psnd_r == s1.psnd_r, "double-swap psnd_r"

# Type sanity (within each module's own class boundary)
assert isinstance(p1, MkPairR_pf), "p1 must be MkPairR"
assert isinstance(swapped, MkPairR_sw), "swapped must be MkPairR"

print("Phase 6 pair_r projection round-trip: OK")
