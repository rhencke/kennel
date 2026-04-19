# ruff: noqa: F821
exec(open("myopt_flatten.py").read())
r1 = myopt_flatten(MyNone())
assert isinstance(r1, MyNone), "myopt_flatten(MyNone()): got " + repr(r1)
r2 = myopt_flatten(MySome(MyNone()))
assert isinstance(r2, MyNone), "myopt_flatten(MySome(MyNone())): got " + repr(r2)
r3 = myopt_flatten(MySome(MySome(42)))
assert isinstance(r3, MySome) and r3.arg0 == 42, (
    "myopt_flatten(MySome(MySome(42))): got " + repr(r3)
)
print("Phase 4 MyOpt round-trip: OK")
