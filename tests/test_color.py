"""Tests for kennel.color — ANSI color helper."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kennel.color import (
    _CODES,
    _RESET,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RED_BOLD,
    YELLOW,
    _color_enabled,
    color,
)

# ---------------------------------------------------------------------------
# _color_enabled
# ---------------------------------------------------------------------------


class TestColorEnabled:
    def test_force_color_enables(self) -> None:
        with patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=True):
            assert _color_enabled() is True

    def test_force_color_overrides_no_color(self) -> None:
        with patch.dict("os.environ", {"FORCE_COLOR": "1", "NO_COLOR": ""}, clear=True):
            assert _color_enabled() is True

    def test_no_color_disables(self) -> None:
        with patch.dict("os.environ", {"NO_COLOR": ""}, clear=True):
            assert _color_enabled() is False

    def test_no_color_any_value_disables(self) -> None:
        with patch.dict("os.environ", {"NO_COLOR": "1"}, clear=True):
            assert _color_enabled() is False

    def test_tty_enables(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("sys.stdout") as mock_stdout:
                mock_stdout.isatty.return_value = True
                assert _color_enabled() is True

    def test_non_tty_disables(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("sys.stdout") as mock_stdout:
                mock_stdout.isatty.return_value = False
                assert _color_enabled() is False

    def test_force_color_wrong_value_falls_through_to_tty(self) -> None:
        with patch.dict("os.environ", {"FORCE_COLOR": "0"}, clear=True):
            with patch("sys.stdout") as mock_stdout:
                mock_stdout.isatty.return_value = False
                assert _color_enabled() is False


# ---------------------------------------------------------------------------
# color()
# ---------------------------------------------------------------------------


class TestColor:
    def _enabled(self) -> dict[str, object]:
        """Patch dict that forces color on."""
        return {"FORCE_COLOR": "1"}

    def _disabled(self) -> dict[str, object]:
        """Patch dict that forces color off."""
        return {"NO_COLOR": ""}

    def test_disabled_returns_text_unchanged(self) -> None:
        with patch.dict("os.environ", self._disabled(), clear=True):
            assert color(BOLD, "hello") == "hello"

    def test_unknown_style_returns_text_unchanged(self) -> None:
        with patch.dict("os.environ", self._enabled(), clear=True):
            assert color("neon_rainbow", "oops") == "oops"

    @pytest.mark.parametrize(
        "style",
        [BOLD, DIM, RED, RED_BOLD, CYAN, MAGENTA, GREEN, YELLOW],
    )
    def test_each_style_wraps_with_ansi(self, style: str) -> None:
        with patch.dict("os.environ", self._enabled(), clear=True):
            result = color(style, "text")
            assert result == f"{_CODES[style]}text{_RESET}"

    def test_color_enabled_bold(self) -> None:
        with patch.dict("os.environ", self._enabled(), clear=True):
            assert color(BOLD, "hi") == "\033[1mhi\033[0m"

    def test_color_enabled_dim(self) -> None:
        with patch.dict("os.environ", self._enabled(), clear=True):
            assert color(DIM, "quiet") == "\033[2mquiet\033[0m"

    def test_color_enabled_red(self) -> None:
        with patch.dict("os.environ", self._enabled(), clear=True):
            assert color(RED, "danger") == "\033[31mdanger\033[0m"

    def test_color_enabled_red_bold(self) -> None:
        with patch.dict("os.environ", self._enabled(), clear=True):
            assert color(RED_BOLD, "alarm") == "\033[1;31malarm\033[0m"

    def test_color_enabled_cyan(self) -> None:
        with patch.dict("os.environ", self._enabled(), clear=True):
            assert color(CYAN, "#42") == "\033[36m#42\033[0m"

    def test_color_enabled_magenta(self) -> None:
        with patch.dict("os.environ", self._enabled(), clear=True):
            assert color(MAGENTA, "#99") == "\033[35m#99\033[0m"

    def test_color_enabled_green(self) -> None:
        with patch.dict("os.environ", self._enabled(), clear=True):
            assert color(GREEN, "worker") == "\033[32mworker\033[0m"

    def test_color_enabled_yellow(self) -> None:
        with patch.dict("os.environ", self._enabled(), clear=True):
            assert color(YELLOW, "webhook") == "\033[33mwebhook\033[0m"
