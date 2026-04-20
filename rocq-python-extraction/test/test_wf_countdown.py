# ruff: noqa: E402
import inspect

from test_support import add_build_default_to_syspath, run_as_script

build_default = add_build_default_to_syspath()

from wf_countdown import wf_countdown


def test_wf_countdown_round_trip() -> None:
    assert wf_countdown(0) == 0, "wf_countdown(0): got " + repr(wf_countdown(0))
    assert wf_countdown(1) == 1, "wf_countdown(1): got " + repr(wf_countdown(1))
    assert wf_countdown(4) == 4, "wf_countdown(4): got " + repr(wf_countdown(4))

    sig = inspect.signature(wf_countdown)
    assert list(sig.parameters) == ["x"], "signature: got " + str(sig)

    source = (build_default / "wf_countdown.py").read_text()
    for forbidden in ("_acc", "_dummy", "Acc", "accessibility", "recproof"):
        assert forbidden not in source, forbidden + " leaked into wf_countdown.py"


if __name__ == "__main__":
    run_as_script(
        test_wf_countdown_round_trip,
        "wf_countdown Program Fixpoint round-trip: OK",
    )
