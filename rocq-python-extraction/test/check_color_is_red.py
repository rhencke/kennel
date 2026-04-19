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

from color_is_red import Blue, Color, Green, Red, color_is_red

assert color_is_red(Red()) is True, "color_is_red(Red()): got " + repr(
    color_is_red(Red())
)
assert color_is_red(Green()) is False, "color_is_red(Green()): got " + repr(
    color_is_red(Green())
)
assert color_is_red(Blue()) is False, "color_is_red(Blue()): got " + repr(
    color_is_red(Blue())
)
assert isinstance(Red(), Color), (
    "Red() must be instance of Color (capitalized base class)"
)
print("Phase 4 color capitalization: OK")
