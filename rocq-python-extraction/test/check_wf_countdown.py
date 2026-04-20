# ruff: noqa: E402
import inspect
import os
import sys
from pathlib import Path

# The extracted .py files always land in the dune workspace build root (_build/default/).
# Walk up from __file__ to find it — works whether or not dune-workspace is present.
_d = os.path.dirname(os.path.abspath(__file__))
while not (
    os.path.basename(_d) == "default"
    and os.path.basename(os.path.dirname(_d)) == "_build"
):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)

from wf_countdown import wf_countdown

assert wf_countdown(0) == 0, "wf_countdown(0): got " + repr(wf_countdown(0))
assert wf_countdown(1) == 1, "wf_countdown(1): got " + repr(wf_countdown(1))
assert wf_countdown(4) == 4, "wf_countdown(4): got " + repr(wf_countdown(4))

sig = inspect.signature(wf_countdown)
assert list(sig.parameters) == ["x"], "signature: got " + str(sig)

source = (Path(_d) / "wf_countdown.py").read_text()
for forbidden in ("_acc", "_dummy", "Acc", "accessibility", "recproof"):
    assert forbidden not in source, forbidden + " leaked into wf_countdown.py"

del _d
print("Phase 7 wf_countdown Program Fixpoint round-trip: OK")
