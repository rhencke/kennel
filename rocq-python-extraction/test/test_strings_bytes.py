from pathlib import Path

from conftest import RenderedSourceAssert
from strings_bytes import (
    ascii_A,
    ascii_roundtrip,
    byte_label,
    byte_lf,
    first_ascii_or_A,
    github_key,
    payload_fragment,
    string_neq,
    tail_or_empty,
)


def test_strings_and_bytes_are_native_python_values() -> None:
    assert github_key == "pull_request"
    assert isinstance(github_key, str)
    assert payload_fragment == b"pull_request"
    assert isinstance(payload_fragment, bytes)


def test_ascii_is_str_byte_is_int() -> None:
    assert ascii_A == "A"
    assert isinstance(ascii_A, str)
    assert byte_lf == 10
    assert first_ascii_or_A("") == "A"
    assert first_ascii_or_A("Zed") == "Z"
    assert ascii_roundtrip("A") == "A"
    assert ascii_roundtrip("\x00") == "\x00"
    assert ascii_roundtrip("\xff") == "\xff"


def test_string_and_byte_patterns() -> None:
    assert tail_or_empty("") == ""
    assert tail_or_empty("Fido") == "ido"
    assert byte_label(10) == "lf"
    assert byte_label(65) == "other"


def test_string_neq_round_trip() -> None:
    assert string_neq("task", "task") is False
    assert string_neq("task", "thread") is True


def test_string_neq_lowers_to_direct_comparison(
    build_default: Path,
    assert_rendered_source: RenderedSourceAssert,
) -> None:
    source = (build_default / "strings_bytes.py").read_text()

    assert_rendered_source(
        source,
        "return left != right",
        ("return not left == right", "return not (left == right)"),
    )
