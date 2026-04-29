import inspect

from datatypes import Blue, Color, Green, Red, color_is_red


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


def test_color_is_red_lowers_to_direct_type_check() -> None:
    source = inspect.getsource(color_is_red)

    assert "match c:" not in source
    assert "isinstance(c, Red)" in source
