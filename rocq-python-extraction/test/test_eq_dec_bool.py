"""Sumbool extraction tests.

``eq_dec_bool`` is a constructor-tag predicate over the
``{0 = 0} + {0 <> 0}`` Sumbool — after Prop-erasure the body is
``match d with Left _ -> true | Right _ -> false``, which the extraction
backend recognises and inlines as ``isinstance(d, Left)`` at call sites
(#1095).  The function itself is therefore not emitted as a top-level
def in ``eq_dec_bool.py``; only the ``Left`` and ``Right`` constructor
classes (the computational content of the Sumbool after erasure) appear.
"""

from eq_dec_bool import Left, Right


def test_sumbool_constructors_are_emitted(build_default) -> None:
    """``Left`` and ``Right`` are the computational survivors of
    Prop-erasure on the Sumbool — they must still be reachable from
    Python."""
    assert isinstance(Left(), Left)
    assert isinstance(Right(), Right)
    source = (build_default / "eq_dec_bool.py").read_text()
    # The dataclasses themselves are emitted, just not the predicate
    # function that would have wrapped them.
    assert "class Left" in source
    assert "class Right" in source


def test_eq_dec_bool_function_is_suppressed_as_tag_predicate(build_default) -> None:
    """The predicate body shapes as ``match d with Left _ -> true | Right
    _ -> false`` so the backend classifies it as a constructor tag
    predicate and inlines call sites with ``isinstance``.  Verify it
    isn't emitted as a top-level def, and verify the surviving file
    doesn't accidentally emit fallback ``return __...`` shims that the
    older non-erasing extraction sometimes produced."""
    source = (build_default / "eq_dec_bool.py").read_text()
    assert "def eq_dec_bool(" not in source
    assert "return _impossible()" not in source
    assert "return __" not in source
