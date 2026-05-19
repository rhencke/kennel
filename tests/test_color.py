"""Tests for fido.color — ANSI color helper."""

import pytest

from fido.color import (
    _CODES,
    _RESET,
    BOLD,
    CYAN,
    DARK_GRAY,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RED_BOLD,
    YELLOW,
    color,
    color_enabled,
)


class _FakeStdout:
    """Minimal stdout fake for TTY detection tests."""

    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


# ---------------------------------------------------------------------------
# color_enabled
# ---------------------------------------------------------------------------


class TestColorEnabled:
    def test_force_color_enables(self) -> None:
        assert color_enabled({"FORCE_COLOR": "1"}) is True

    def test_force_color_overrides_no_color(self) -> None:
        assert color_enabled({"FORCE_COLOR": "1", "NO_COLOR": ""}) is True

    def test_no_color_disables(self) -> None:
        assert color_enabled({"NO_COLOR": ""}) is False

    def test_no_color_any_value_disables(self) -> None:
        assert color_enabled({"NO_COLOR": "1"}) is False

    def test_tty_enables(self) -> None:
        assert color_enabled({}, stdout=_FakeStdout(tty=True)) is True

    def test_non_tty_disables(self) -> None:
        assert color_enabled({}, stdout=_FakeStdout(tty=False)) is False

    def test_force_color_wrong_value_falls_through_to_tty(self) -> None:
        assert (
            color_enabled({"FORCE_COLOR": "0"}, stdout=_FakeStdout(tty=False)) is False
        )


# ---------------------------------------------------------------------------
# color()
# ---------------------------------------------------------------------------

_ENABLED = {"FORCE_COLOR": "1"}
_DISABLED = {"NO_COLOR": ""}


class TestColor:
    def test_disabled_returns_text_unchanged(self) -> None:
        assert color(BOLD, "hello", env=_DISABLED) == "hello"

    def test_unknown_style_returns_text_unchanged(self) -> None:
        assert color("neon_rainbow", "oops", env=_ENABLED) == "oops"

    @pytest.mark.parametrize(
        "style",
        [BOLD, DIM, RED, RED_BOLD, CYAN, MAGENTA, GREEN, YELLOW, DARK_GRAY],
    )
    def test_each_style_wraps_with_ansi(self, style: str) -> None:
        result = color(style, "text", env=_ENABLED)
        assert result == f"{_CODES[style]}text{_RESET}"

    def testcolor_enabled_bold(self) -> None:
        assert color(BOLD, "hi", env=_ENABLED) == "\033[1mhi\033[0m"

    def testcolor_enabled_dim(self) -> None:
        assert color(DIM, "quiet", env=_ENABLED) == "\033[2mquiet\033[0m"

    def testcolor_enabled_red(self) -> None:
        assert color(RED, "danger", env=_ENABLED) == "\033[31mdanger\033[0m"

    def testcolor_enabled_red_bold(self) -> None:
        assert color(RED_BOLD, "alarm", env=_ENABLED) == "\033[1;31malarm\033[0m"

    def testcolor_enabled_cyan(self) -> None:
        assert color(CYAN, "#42", env=_ENABLED) == "\033[36m#42\033[0m"

    def testcolor_enabled_magenta(self) -> None:
        assert color(MAGENTA, "#99", env=_ENABLED) == "\033[35m#99\033[0m"

    def testcolor_enabled_green(self) -> None:
        assert color(GREEN, "worker", env=_ENABLED) == "\033[32mworker\033[0m"

    def testcolor_enabled_yellow(self) -> None:
        assert color(YELLOW, "webhook", env=_ENABLED) == "\033[33mwebhook\033[0m"

    def testcolor_enabled_dark_gray(self) -> None:
        assert color(DARK_GRAY, "paused", env=_ENABLED) == "\033[90mpaused\033[0m"


# ---------------------------------------------------------------------------
# wrap_raw / rgb_fg / rgb_bg / wrap_bg_line  (provider-color feature)
# ---------------------------------------------------------------------------


class TestRawWrapping:
    def test_rgb_fg_emits_truecolor_escape(self) -> None:
        from fido.color import rgb_fg

        assert rgb_fg(255, 160, 60) == "\033[38;2;255;160;60m"

    def test_rgb_bg_emits_truecolor_escape(self) -> None:
        from fido.color import rgb_bg

        assert rgb_bg(30, 15, 0) == "\033[48;2;30;15;0m"

    def test_wrap_raw_wraps_when_enabled(self) -> None:
        from fido.color import rgb_fg, wrap_raw

        result = wrap_raw(rgb_fg(10, 20, 30), "x", env=_ENABLED)
        assert result == f"\033[38;2;10;20;30mx{_RESET}"

    def test_wrap_raw_returns_text_when_disabled(self) -> None:
        from fido.color import rgb_fg, wrap_raw

        assert wrap_raw(rgb_fg(10, 20, 30), "x", env=_DISABLED) == "x"

    def test_wrap_raw_ignores_empty_escape(self) -> None:
        from fido.color import wrap_raw

        assert wrap_raw("", "x", env=_ENABLED) == "x"

    def test_wrap_bg_line_applies_bg_across_inner_resets(self) -> None:
        """Inner `_RESET`s must not punch holes in the background."""
        from fido.color import rgb_bg, wrap_bg_line

        bg = rgb_bg(30, 15, 0)
        # A pre-styled line: bold "A" then plain "B".
        line = f"\033[1mA{_RESET}B"
        result = wrap_bg_line(bg, line, env=_ENABLED)
        # After every inner reset, bg is re-applied; one final reset closes.
        assert result == f"{bg}\033[1mA{_RESET}{bg}B{_RESET}"

    def test_wrap_bg_line_no_inner_reset(self) -> None:
        from fido.color import rgb_bg, wrap_bg_line

        bg = rgb_bg(30, 15, 0)
        assert wrap_bg_line(bg, "plain", env=_ENABLED) == f"{bg}plain{_RESET}"

    def test_wrap_bg_line_disabled_returns_line_unchanged(self) -> None:
        from fido.color import rgb_bg, wrap_bg_line

        assert wrap_bg_line(rgb_bg(30, 15, 0), "plain", env=_DISABLED) == "plain"

    def test_wrap_bg_line_empty_escape_returns_line_unchanged(self) -> None:
        from fido.color import wrap_bg_line

        assert wrap_bg_line("", "plain", env=_ENABLED) == "plain"
