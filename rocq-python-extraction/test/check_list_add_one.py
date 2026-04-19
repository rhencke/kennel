# ruff: noqa: F821
exec(open("list_add_one.py").read())
assert list_add_one([]) == [], "list_add_one([]): got " + repr(list_add_one([]))
assert list_add_one([0, 1, 2]) == [1, 2, 3], "list_add_one([0,1,2]): got " + repr(
    list_add_one([0, 1, 2])
)
assert list_add_one([5]) == [6], "list_add_one([5]): got " + repr(list_add_one([5]))
print("Phase 3 list round-trip: OK")
