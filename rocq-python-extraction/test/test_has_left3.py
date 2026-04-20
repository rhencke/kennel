from has_left3 import Leaf, Node, Tree, has_left3


def test_has_left3_round_trip() -> None:
    assert has_left3(Leaf()) is False, "has_left3(Leaf()): got " + repr(
        has_left3(Leaf())
    )
    assert has_left3(Node(1, Leaf(), Leaf())) is False, (
        "has_left3(1-level): got " + repr(has_left3(Node(1, Leaf(), Leaf())))
    )
    two_deep = Node(1, Node(2, Leaf(), Leaf()), Leaf())
    assert has_left3(two_deep) is False, "has_left3(2-level): got " + repr(
        has_left3(two_deep)
    )
    three_deep = Node(1, Node(2, Node(3, Leaf(), Leaf()), Leaf()), Leaf())
    assert has_left3(three_deep) is True, "has_left3(3-level): got " + repr(
        has_left3(three_deep)
    )
    right_deep = Node(1, Leaf(), Node(2, Node(3, Leaf(), Leaf()), Leaf()))
    assert has_left3(right_deep) is False, "has_left3(right-deep): got " + repr(
        has_left3(right_deep)
    )
    assert isinstance(Leaf(), Tree), "Leaf() must be instance of Tree"
    assert isinstance(Node(0, Leaf(), Leaf()), Tree), (
        "Node(...) must be instance of Tree"
    )
