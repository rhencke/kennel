from ascii_A import ascii_A
from ascii_roundtrip import ascii_roundtrip
from byte_label import byte_label
from byte_lf import byte_lf
from first_ascii_or_A import first_ascii_or_A
from github_key import github_key
from payload_fragment import payload_fragment
from tail_or_empty import tail_or_empty


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
