# pyright: reportUnknownLambdaType=false
from typing import assert_type

from polymorphism import list_map

assert_type(list_map(lambda x: x + 1, [1, 2, 3]), list[int])
assert_type(list_map(str, [1, 2, 3]), list[str])
