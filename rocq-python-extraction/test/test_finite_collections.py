import FiniteCollectionFixtures

fixtures = FiniteCollectionFixtures.FiniteCollectionFixtures


def test_positive_maps_are_native_persistent_dicts() -> None:
    assert fixtures.positive_task_map == {1: "plan", 3: "ci"}
    assert fixtures.positive_task_hit == "plan"
    assert fixtures.positive_task_missing is None
    assert fixtures.positive_task_removed == {3: "ci"}
    assert fixtures.positive_task_map == {1: "plan", 3: "ci"}
    assert fixtures.positive_task_has_3 is True
    assert fixtures.positive_task_count == 2
    assert fixtures.positive_task_elements == [(1, "plan"), (3, "ci")]


def test_positive_sets_are_native_persistent_frozensets() -> None:
    assert fixtures.positive_claim_set == frozenset({2, 5})
    assert fixtures.positive_claim_union == frozenset({2, 5, 7})
    assert fixtures.positive_claim_inter == frozenset({5})
    assert fixtures.positive_claim_diff == frozenset({7})
    assert fixtures.positive_claim_union_expr == frozenset({2, 5, 7})
    assert fixtures.positive_claim_inter_expr == frozenset({2, 5})
    assert fixtures.positive_claim_diff_expr == frozenset({7})
    assert fixtures.positive_claim_nested_expr == frozenset({2, 5})
    assert fixtures.positive_claim_union_inter_expr == frozenset({2, 5, 7})
    assert fixtures.positive_claim_diff_union_expr == frozenset({2})
    assert fixtures.positive_claim_inter_diff_expr == frozenset({2, 5})
    assert fixtures.positive_claim_removed == frozenset({5})
    assert fixtures.positive_claim_has_2 is True
    assert fixtures.positive_claim_count == 2
    assert fixtures.positive_claim_elements == [2, 5]


def test_string_maps_and_sets_have_sorted_views() -> None:
    assert fixtures.string_label_map == [("alpha", 1), ("beta", 2)]
    assert fixtures.string_label_hit == 1
    assert fixtures.string_label_elements == [("alpha", 1), ("beta", 2)]
    labels = ["alpha", "beta"]
    assert fixtures.string_label_set(labels) == labels
    assert fixtures.string_label_set_elements(labels) == labels


def test_set_infix_lowerings_preserve_precedence(build_default) -> None:
    source = (build_default / "FiniteCollectionFixtures.py").read_text()

    assert (
        "positive_claim_union_expr = (positive_claim_set | positive_claim_diff)"
        in source
    )
    assert (
        "positive_claim_inter_expr = (positive_claim_set & positive_claim_union)"
        in source
    )
    assert (
        "positive_claim_diff_expr = (positive_claim_union - positive_claim_set)"
        in source
    )
    assert (
        "positive_claim_nested_expr = "
        "((positive_claim_set | positive_claim_diff) & positive_claim_union)"
    ) in source
    assert (
        "positive_claim_union_inter_expr = "
        "(positive_claim_diff | positive_claim_set & positive_claim_union)"
    ) in source
    assert (
        "positive_claim_diff_union_expr = "
        "(positive_claim_union - (positive_claim_diff | positive_claim_inter))"
    ) in source
    assert (
        "positive_claim_inter_diff_expr = "
        "((positive_claim_union - positive_claim_diff) & positive_claim_set)"
    ) in source
