# ruff: noqa: F821
exec(open("color_is_red.py").read())
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
