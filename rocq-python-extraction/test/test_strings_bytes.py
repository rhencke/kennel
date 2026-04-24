import pytest
from ascii_A import ascii_A
from ascii_roundtrip import ascii_roundtrip
from byte_label import byte_label
from byte_lf import byte_lf
from first_ascii_or_A import first_ascii_or_A
from github_key import github_key
from payload_fragment import payload_fragment
from tail_or_empty import _RocqUtf8BoundaryError, tail_or_empty


def test_strings_and_bytes_are_native_python_values() -> None:
    assert github_key == "pull_request"
    assert isinstance(github_key, str)
    assert payload_fragment == b"pull_request"
    assert isinstance(payload_fragment, bytes)


def test_ascii_and_byte_are_ints() -> None:
    assert ascii_A == 65
    assert byte_lf == 10
    assert first_ascii_or_A("") == 65
    assert first_ascii_or_A("Zed") == 90
    assert ascii_roundtrip(0) == 0
    assert ascii_roundtrip(255) == 255


def test_string_and_byte_patterns() -> None:
    assert tail_or_empty("") == ""
    assert tail_or_empty("Fido") == "ido"
    assert byte_label(10) == "lf"
    assert byte_label(65) == "other"


def test_string_pattern_rejects_invalid_utf8_split() -> None:
    with pytest.raises(_RocqUtf8BoundaryError):
        tail_or_empty("é")
