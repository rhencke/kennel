from typing import assert_type

from ascii_A import ascii_A
from byte_lf import byte_lf
from github_key import github_key
from payload_fragment import payload_fragment
from tail_or_empty import tail_or_empty

assert_type(github_key, str)
assert_type(payload_fragment, bytes)
assert_type(ascii_A, str)
assert_type(byte_lf, int)
assert_type(tail_or_empty("abc"), str)
