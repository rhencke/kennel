# ruff: noqa: F821
exec(open("mylist_is_empty.py").read())
assert mylist_is_empty(MNil()) is True, "mylist_is_empty(MNil()): got " + repr(
    mylist_is_empty(MNil())
)
assert mylist_is_empty(MCons(1, MNil())) is False, (
    "mylist_is_empty(MCons(...)): got " + repr(mylist_is_empty(MCons(1, MNil())))
)
print("Phase 4 MyList round-trip: OK")
