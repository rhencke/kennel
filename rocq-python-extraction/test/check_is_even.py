# ruff: noqa: F821
exec(open("is_even.py").read())
assert is_even(0) is True, "is_even(0): got " + repr(is_even(0))
assert is_even(1) is False, "is_even(1): got " + repr(is_even(1))
assert is_even(2) is True, "is_even(2): got " + repr(is_even(2))
assert is_even(3) is False, "is_even(3): got " + repr(is_even(3))
assert is_even(4) is True, "is_even(4): got " + repr(is_even(4))
print("Phase 4 is_even round-trip: OK")
