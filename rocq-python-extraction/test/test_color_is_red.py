# ruff: noqa: E402
from test_support import add_build_default_to_syspath, run_as_script

add_build_default_to_syspath()

from color_is_red import Blue, Color, Green, Red, color_is_red


def test_color_is_red_round_trip() -> None:
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


if __name__ == "__main__":
    run_as_script(test_color_is_red_round_trip, "color capitalization: OK")
