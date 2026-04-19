# ruff: noqa: F821
exec(open("pair_swap.py").read())
assert pair_swap((0, 1)) == (1, 0), "pair_swap((0, 1)): got " + repr(pair_swap((0, 1)))
assert pair_swap((3, 7)) == (7, 3), "pair_swap((3, 7)): got " + repr(pair_swap((3, 7)))
assert pair_swap((5, 5)) == (5, 5), "pair_swap((5, 5)): got " + repr(pair_swap((5, 5)))
print("Phase 3 prod round-trip: OK")
