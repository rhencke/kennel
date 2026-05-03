"""Constructor-tag-predicate tests.

The extraction backend recognises that single-argument boolean functions
shaped like ``match c with | C => true | _ => false end`` are constructor
tag predicates: rather than emitting a top-level ``def color_is_red(...)``
that callers must invoke, it inlines an equivalent ``isinstance(c, C)``
expression at every call site (#1095).  ``color_is_red`` and
``color_is_red_tag`` are both classified that way, so neither shows up
as a top-level function in the generated module — but the inlining
behaviour is observable through ``color_tag_matches_filter``, which
calls ``color_is_red_tag`` and gets the inlined ``isinstance(c, Red)``
expansion.
"""

import inspect

import datatypes
from datatypes import Blue, Color, Green, Red, color_tag_matches_filter


def test_color_constructor_classes_are_present() -> None:
    """The Color base class and its three constructors must still be
    emitted — only the tag-predicate functions are suppressed."""
    assert isinstance(Red(), Color)
    assert isinstance(Green(), Color)
    assert isinstance(Blue(), Color)
    module_source = inspect.getsource(datatypes)
    assert "class Color:" in module_source
    assert "class Red(Color):" in module_source
    assert "class Green(Color):" in module_source
    assert "class Blue(Color):" in module_source


def test_constructor_tag_predicates_are_suppressed_at_top_level() -> None:
    """Both ``color_is_red`` and ``color_is_red_tag`` lower to inline
    ``isinstance`` checks at their call sites; neither is emitted as a
    top-level ``def`` in datatypes.py."""
    module_source = inspect.getsource(datatypes)
    assert "def color_is_red(" not in module_source
    assert "def color_is_red_tag(" not in module_source


def test_color_tag_matches_filter_inlines_constructor_check() -> None:
    """``color_tag_matches_filter`` is the public function that
    exercises the inlined predicate.  Verify both behaviour and the
    lowered form."""
    assert color_tag_matches_filter(True, Red()) is True
    assert color_tag_matches_filter(True, Green()) is False
    assert color_tag_matches_filter(False, Red()) is False
    assert color_tag_matches_filter(False, Green()) is True

    filter_source = inspect.getsource(color_tag_matches_filter)
    assert "color_is_red_tag(" not in filter_source
    assert "isinstance(c, Red)" in filter_source
    assert "not isinstance(c, Red)" in filter_source


def test_generated_dataclasses_are_final() -> None:
    module_source = inspect.getsource(datatypes)

    assert "    final,\n" in module_source
    assert "@final\n@dataclass(frozen=True)\nclass Red(Color):" in module_source
