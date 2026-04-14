"""ANSI color helper for terminal output.

Color is enabled when:
- ``FORCE_COLOR=1`` is set in the environment, OR
- stdout is a TTY and ``NO_COLOR`` is not set.

``NO_COLOR`` follows https://no-color.org — presence of the variable (any
value) disables color.  ``FORCE_COLOR=1`` overrides both TTY detection and
``NO_COLOR`` so that tools like ``watch -c`` work without extra flags.
"""

from __future__ import annotations

import os
import sys

# ANSI escape sequences
_RESET = "\033[0m"
_CODES: dict[str, str] = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "red_bold": "\033[1;31m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "green": "\033[32m",
    "yellow": "\033[33m",
}

# Semantic style name constants for import convenience
BOLD = "bold"
DIM = "dim"
RED = "red"
RED_BOLD = "red_bold"
CYAN = "cyan"
MAGENTA = "magenta"
GREEN = "green"
YELLOW = "yellow"


def _color_enabled() -> bool:
    """Return True if ANSI color output should be used."""
    if os.environ.get("FORCE_COLOR") == "1":
        return True
    if "NO_COLOR" in os.environ:
        return False
    return sys.stdout.isatty()


def color(style: str, text: str) -> str:
    """Wrap *text* in ANSI escape codes for *style* if color is enabled.

    Returns *text* unchanged when color is disabled or *style* is unknown.
    """
    if not _color_enabled():
        return text
    code = _CODES.get(style, "")
    if not code:
        return text
    return f"{code}{text}{_RESET}"
