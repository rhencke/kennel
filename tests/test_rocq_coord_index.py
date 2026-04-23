from dataclasses import FrozenInstanceError

import pytest

from fido.rocq.coord_index import (
    CoordIndex,
    empty_coord_index,
)


def test_coord_index_record_lowers_to_type_named_class() -> None:
    assert isinstance(empty_coord_index, CoordIndex)

    with pytest.raises(FrozenInstanceError):
        empty_coord_index.coord_claims = frozenset({1})  # pyright: ignore[reportAttributeAccessIssue]


def test_coord_index_groups_repeated_coordination_state() -> None:
    index = CoordIndex(
        coord_claims=frozenset(),
        coord_issue_owners={},
        coord_repos=[("FidoCanCode/home", ("/workspace/home", "claude-code"))],
    )

    claimed = index.add_claim(7)
    assert claimed.has_claim(7)
    assert not index.has_claim(7)

    assigned = claimed.assign_issue(721, "fido")
    assert assigned.issue_owner(721) == "fido"

    unassigned = assigned.unassign_issue(721)
    assert unassigned.issue_owner(721) is None

    unclaimed = unassigned.remove_claim(7)
    assert not unclaimed.has_claim(7)
    assert unclaimed.repo_providers() == ["claude-code"]
    assert unclaimed.repo_count() == 1
