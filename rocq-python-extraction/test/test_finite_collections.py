import FiniteCollectionFixtures

fixtures = FiniteCollectionFixtures.FiniteCollectionFixtures


def test_positive_maps_are_native_persistent_dicts() -> None:
    assert fixtures.positive_task_map == {1: "plan", 3: "ci"}
    assert fixtures.positive_task_hit == "plan"
    assert fixtures.positive_task_missing is None
    assert fixtures.positive_task_removed == {3: "ci"}
    assert fixtures.positive_task_find_expr == "plan"
    assert fixtures.positive_task_mem_expr is True
    assert fixtures.positive_task_cardinal_expr == 2
    assert fixtures.positive_task_elements_expr == [(1, "plan"), (3, "ci")]
    assert fixtures.positive_task_fold_count == 2
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
    assert fixtures.positive_claim_mem_expr is True
    assert fixtures.positive_claim_cardinal_expr == 2
    assert fixtures.positive_claim_elements_expr == [2, 5]
    assert fixtures.positive_claim_fold_count == 2
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


def test_set_infix_lowerings_preserve_precedence(
    build_default,
    assert_rendered_source,
) -> None:
    source = (build_default / "FiniteCollectionFixtures.py").read_text()

    assert_rendered_source(
        source,
        "positive_claim_union_expr = positive_claim_set | positive_claim_diff",
    )
    assert_rendered_source(
        source,
        "positive_claim_inter_expr = positive_claim_set & positive_claim_union",
    )
    assert_rendered_source(
        source,
        "positive_claim_diff_expr = positive_claim_union - positive_claim_set",
    )
    assert_rendered_source(
        source,
        "positive_claim_nested_expr = "
        "(positive_claim_set | positive_claim_diff) & positive_claim_union",
    )
    assert_rendered_source(
        source,
        "positive_claim_union_inter_expr = "
        "positive_claim_diff | positive_claim_set & positive_claim_union",
    )
    assert_rendered_source(
        source,
        "positive_claim_diff_union_expr = "
        "positive_claim_union - (positive_claim_diff | positive_claim_inter)",
    )
    assert_rendered_source(
        source,
        "positive_claim_inter_diff_expr = "
        "(positive_claim_union - positive_claim_diff) & positive_claim_set",
    )


def test_finite_collection_rule_shapes_are_characterized(
    build_default,
    assert_rendered_source,
) -> None:
    source = (build_default / "FiniteCollectionFixtures.py").read_text()

    expected_snippets = (
        'positive_task_map = _rocq_map_add(3, b"ci", _rocq_map_add(1, b"plan", {}))',
        "positive_task_removed = _rocq_map_remove(1, positive_task_map)",
        "positive_task_find_expr = positive_task_map.get(1)",
        "positive_task_mem_expr = 3 in positive_task_map",
        "positive_task_cardinal_expr = len(positive_task_map)",
        "positive_task_elements_expr = _rocq_map_elements(positive_task_map)",
        "positive_claim_set = _rocq_set_add(5, _rocq_set_add(2, frozenset()))",
        "positive_claim_removed = _rocq_set_remove(2, positive_claim_set)",
        "positive_claim_mem_expr = 2 in positive_claim_set",
        "positive_claim_cardinal_expr = len(positive_claim_set)",
        "positive_claim_elements_expr = _rocq_set_elements(positive_claim_set)",
    )

    for snippet in expected_snippets:
        assert_rendered_source(source, snippet)

    assert "_rocq_map_fold(" in source
    assert "_rocq_set_fold(" in source
    assert "class PositiveMap" not in source
    assert "class PositiveSet" not in source
