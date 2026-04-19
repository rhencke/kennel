# ruff: noqa: F821
exec(open("bool_not.py").read())
assert bool_not(True) is False, "bool_not(True): got " + repr(bool_not(True))
assert bool_not(False) is True, "bool_not(False): got " + repr(bool_not(False))
print("Phase 3 bool round-trip: OK")
