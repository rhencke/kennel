# ruff: noqa: F821
exec(open("option_inc.py").read())
assert option_inc(None) is None, "option_inc(None): got " + repr(option_inc(None))
assert option_inc(0) == 1, "option_inc(0): got " + repr(option_inc(0))
assert option_inc(5) == 6, "option_inc(5): got " + repr(option_inc(5))
print("Phase 3 option round-trip: OK")
