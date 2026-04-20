from bintree_is_leaf import BLeaf, BNode, bintree_is_leaf


def test_bintree_is_leaf_round_trip() -> None:
    assert bintree_is_leaf(BLeaf()) is True, "bintree_is_leaf(BLeaf()): got " + repr(
        bintree_is_leaf(BLeaf())
    )
    assert bintree_is_leaf(BNode(BLeaf(), 42, BLeaf())) is False, (
        "bintree_is_leaf(BNode(...)): got "
        + repr(bintree_is_leaf(BNode(BLeaf(), 42, BLeaf())))
    )
