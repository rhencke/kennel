from ntree_is_leaf import NLeaf, NNode, NTree, ntree_is_leaf


def test_ntree_is_leaf_round_trip() -> None:
    assert ntree_is_leaf(NLeaf()) is True, "ntree_is_leaf(NLeaf()): got " + repr(
        ntree_is_leaf(NLeaf())
    )
    assert ntree_is_leaf(NNode([])) is False, "ntree_is_leaf(NNode([])): got " + repr(
        ntree_is_leaf(NNode([]))
    )
    assert ntree_is_leaf(NNode([NLeaf()])) is False, (
        "ntree_is_leaf(NNode([NLeaf()])): got " + repr(ntree_is_leaf(NNode([NLeaf()])))
    )
    assert isinstance(NLeaf(), NTree), "NLeaf() must be instance of NTree"
    assert isinstance(NNode([]), NTree), "NNode([]) must be instance of NTree"
