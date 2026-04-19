# ruff: noqa: F821
exec(open("bintree_is_leaf.py").read())
assert bintree_is_leaf(BLeaf()) is True, "bintree_is_leaf(BLeaf()): got " + repr(
    bintree_is_leaf(BLeaf())
)
assert bintree_is_leaf(BNode(BLeaf(), 42, BLeaf())) is False, (
    "bintree_is_leaf(BNode(...)): got "
    + repr(bintree_is_leaf(BNode(BLeaf(), 42, BLeaf())))
)
print("Phase 4 BinTree round-trip: OK")
