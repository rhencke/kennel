# ruff: noqa: F821
exec(open("even_depth.py").read())
assert even_depth(EvenO()) == 0, "even_depth(EvenO()): got " + repr(even_depth(EvenO()))
assert even_depth(EvenS(OddS(EvenO()))) == 2, "even_depth depth-2: got " + repr(
    even_depth(EvenS(OddS(EvenO())))
)
assert even_depth(EvenS(OddS(EvenS(OddS(EvenO()))))) == 4, (
    "even_depth depth-4: got " + repr(even_depth(EvenS(OddS(EvenS(OddS(EvenO()))))))
)
assert isinstance(EvenO(), Even), "EvenO() must be instance of Even"
assert isinstance(OddS(EvenO()), Odd), "OddS() must be instance of Odd"
print("Phase 4 Even/Odd round-trip: OK")
