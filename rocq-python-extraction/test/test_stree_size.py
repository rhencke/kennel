from stree_size import DDecl, DEnd, DTree, SLit, SSeq, STree, stree_size


def test_stree_size_round_trip() -> None:
    assert stree_size(SLit(42)) == 1, "stree_size(SLit(42)): got " + repr(
        stree_size(SLit(42))
    )
    assert stree_size(SSeq(DEnd(), SLit(0))) == 1, (
        "stree_size(SSeq(DEnd(), SLit(0))): got "
        + repr(stree_size(SSeq(DEnd(), SLit(0))))
    )
    assert stree_size(SSeq(DDecl(SLit(1), DDecl(SLit(2), DEnd())), SLit(3))) == 3, (
        "stree_size nested: got "
        + repr(stree_size(SSeq(DDecl(SLit(1), DDecl(SLit(2), DEnd())), SLit(3))))
    )
    assert isinstance(SLit(0), STree), "SLit(0) must be instance of STree"
    assert isinstance(DEnd(), DTree), "DEnd() must be instance of DTree"
