from itertools import islice
from typing import cast

from repeat_tree import CNode, Cotree, coforce, repeat_tree
from tree_root_of_repeat import Nat as TreeNat
from tree_root_of_repeat import tree_root_of_repeat
from zeros import Nat as ZeroNat
from zeros import O, Stream, zeros
from zeros_pair import Pair, coprefix_eq, coprefix_hash, zeros_pair

zero_prefix = list(islice(zeros, 4))
assert all(isinstance(x, O) for x in zero_prefix)

pair = cast(Pair[Stream[ZeroNat], Stream[ZeroNat]], zeros_pair)
assert coprefix_eq(4, pair.arg0, pair.arg1)
assert isinstance(coprefix_hash(4, pair.arg0), int)
assert isinstance(coforce(repeat_tree), CNode)
assert isinstance(repeat_tree, Cotree)
assert isinstance(tree_root_of_repeat, TreeNat)
