# ruff: noqa: F821
exec(open("nat_double.py").read())
assert nat_double(0) == 0, "nat_double(0): got " + repr(nat_double(0))
assert nat_double(3) == 6, "nat_double(3): got " + repr(nat_double(3))
assert nat_double(5) == 10, "nat_double(5): got " + repr(nat_double(5))
print("Phase 3 nat round-trip: OK")
