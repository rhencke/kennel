from itertools import islice

from repeat_tree import CNode, coforce, repeat_tree
from tree_root_of_repeat import tree_root_of_repeat
from zeros import zeros
from zeros_pair import coprefix_eq, coprefix_hash, zeros_pair


def test_coinductive_round_trip() -> None:
    assert list(islice(zeros, 6)) == [0, 0, 0, 0, 0, 0]

    left = zeros_pair.arg0
    right = zeros_pair.arg1
    assert coprefix_eq(8, left, right)
    assert coprefix_hash(8, left) == coprefix_hash(8, right)

    step = coforce(repeat_tree)
    assert isinstance(step, CNode)
    assert step.arg0 == 0

    assert tree_root_of_repeat == 0
