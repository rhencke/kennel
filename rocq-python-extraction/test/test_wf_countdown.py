import inspect
from pathlib import Path

from wf_recursion import wf_countdown


def test_wf_countdown_round_trip(build_default: Path) -> None:
    assert wf_countdown(0) == 0, "wf_countdown(0): got " + repr(wf_countdown(0))
    assert wf_countdown(1) == 1, "wf_countdown(1): got " + repr(wf_countdown(1))
    assert wf_countdown(4) == 4, "wf_countdown(4): got " + repr(wf_countdown(4))

    sig = inspect.signature(wf_countdown)
    assert list(sig.parameters) == ["x"], "signature: got " + str(sig)

    source = (build_default / "wf_recursion.py").read_text()
    for forbidden in ("_acc", "_dummy", "Acc", "accessibility", "recproof"):
        assert forbidden not in source, forbidden + " leaked into wf_recursion.py"
