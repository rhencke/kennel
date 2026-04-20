# ruff: noqa: E402
from test_support import add_build_default_to_syspath, run_as_script

add_build_default_to_syspath()

from bintree_is_leaf import BLeaf, BNode, bintree_is_leaf


def test_bintree_is_leaf_round_trip() -> None:
    assert bintree_is_leaf(BLeaf()) is True, "bintree_is_leaf(BLeaf()): got " + repr(
        bintree_is_leaf(BLeaf())
    )
    assert bintree_is_leaf(BNode(BLeaf(), 42, BLeaf())) is False, (
        "bintree_is_leaf(BNode(...)): got "
        + repr(bintree_is_leaf(BNode(BLeaf(), 42, BLeaf())))
    )


if __name__ == "__main__":
    run_as_script(test_bintree_is_leaf_round_trip, "BinTree round-trip: OK")
