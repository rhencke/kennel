# ruff: noqa: E402
from itertools import islice

from test_support import add_build_default_to_syspath, run_as_script

add_build_default_to_syspath()

from repeat_tree import CNode, coforce, repeat_tree
from repeat_tree import O as TreeO
from tree_root_of_repeat import O as RootO
from tree_root_of_repeat import tree_root_of_repeat
from zeros import O as ZeroO
from zeros import zeros
from zeros_pair import coprefix_eq, coprefix_hash, zeros_pair


def test_coinductive_round_trip() -> None:
    assert list(islice(zeros, 6)) == [
        ZeroO(),
        ZeroO(),
        ZeroO(),
        ZeroO(),
        ZeroO(),
        ZeroO(),
    ]

    left = zeros_pair.arg0
    right = zeros_pair.arg1
    assert coprefix_eq(8, left, right)
    assert coprefix_hash(8, left) == coprefix_hash(8, right)

    step = coforce(repeat_tree)
    assert isinstance(step, CNode)
    assert step.arg0 == TreeO()

    assert tree_root_of_repeat == RootO()


if __name__ == "__main__":
    run_as_script(
        test_coinductive_round_trip,
        "Coinductive stream and forcing round-trip: OK",
    )
