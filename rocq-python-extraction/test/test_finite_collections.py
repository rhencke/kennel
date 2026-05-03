from pathlib import Path

import FiniteCollectionFixtures
from conftest import RenderedSourceAssert

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
    # ``positive_claim_nested_expr = inter (union set diff) union`` —
    # ({2,5} ∪ {7}) ∩ {2,5,7} = {2,5,7}.  The earlier expectation of
    # {2,5} was a stale fixture (likely from when ``positive_claim_union``
    # didn't include p7).
    assert fixtures.positive_claim_nested_expr == frozenset({2, 5, 7})
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
    build_default: Path,
    assert_rendered_source: RenderedSourceAssert,
) -> None:
    source = (build_default / "FiniteCollectionFixtures.py").read_text()

    # Each ``positive_claim_*`` binding is now emitted with a
    # ``: frozenset[int]`` type annotation between the name and the
    # ``=``, so loosen the substring match to accept either form.
    assert_rendered_source(
        source,
        "positive_claim_union_expr: frozenset[int] = "
        "positive_claim_set | positive_claim_diff",
    )
    assert_rendered_source(
        source,
        "positive_claim_inter_expr: frozenset[int] = "
        "positive_claim_set & positive_claim_union",
    )
    assert_rendered_source(
        source,
        "positive_claim_diff_expr: frozenset[int] = "
        "positive_claim_union - positive_claim_set",
    )
    assert_rendered_source(
        source,
        "positive_claim_nested_expr: frozenset[int] = "
        "(positive_claim_set | positive_claim_diff) & positive_claim_union",
    )
    assert_rendered_source(
        source,
        "positive_claim_union_inter_expr: frozenset[int] = "
        "positive_claim_diff | positive_claim_set & positive_claim_union",
    )
    assert_rendered_source(
        source,
        "positive_claim_diff_union_expr: frozenset[int] = "
        "positive_claim_union - (positive_claim_diff | positive_claim_inter)",
    )
    # Python's ``-`` binds tighter than ``&`` for frozensets, so the
    # precedence-aware extraction can omit the parens around
    # ``(union - diff)``.  Both forms compute the same value; accept
    # the parenthesized or unparenthesized rendering as long as the
    # forbidden ``union - (diff & set)`` regrouping doesn't appear.
    has_parens = (
        "positive_claim_inter_diff_expr: frozenset[int] = "
        "(positive_claim_union - positive_claim_diff) & positive_claim_set" in source
    )
    no_parens = (
        "positive_claim_inter_diff_expr: frozenset[int] = "
        "positive_claim_union - positive_claim_diff & positive_claim_set" in source
    )
    assert has_parens or no_parens
    assert (
        "positive_claim_inter_diff_expr: frozenset[int] = "
        "positive_claim_union - (positive_claim_diff & positive_claim_set)"
        not in source
    )


def test_finite_collection_rule_shapes_are_characterized(
    build_default: Path,
    assert_rendered_source: RenderedSourceAssert,
) -> None:
    source = (build_default / "FiniteCollectionFixtures.py").read_text()

    expected_snippets = (
        'positive_task_map: dict[int, str] = _rocq_map_add(_rocq_positive_key(p3), "ci", _rocq_map_add(_rocq_positive_key(p1), "plan", {}))',
        "positive_task_removed: dict[int, str] = _rocq_map_remove(_rocq_positive_key(p1), positive_task_map)",
        "positive_task_find_expr: str | None = positive_task_map.get(_rocq_positive_key(p1))",
        "positive_task_mem_expr: bool = _rocq_positive_key(p3) in positive_task_map",
        "positive_task_cardinal_expr: int = len(positive_task_map)",
        "positive_task_elements_expr: list[tuple[int, str]] = _rocq_map_elements(positive_task_map)",
        "positive_claim_set: frozenset[int] = _rocq_set_add(_rocq_positive_key(p5), _rocq_set_add(_rocq_positive_key(p2), frozenset()))",
        "positive_claim_removed: frozenset[int] = _rocq_set_remove(_rocq_positive_key(p2), positive_claim_set)",
        "positive_claim_mem_expr: bool = _rocq_positive_key(p2) in positive_claim_set",
        "positive_claim_cardinal_expr: int = len(positive_claim_set)",
        "positive_claim_elements_expr: list[int] = _rocq_set_elements(positive_claim_set)",
    )

    for snippet in expected_snippets:
        assert_rendered_source(source, snippet)

    assert "_rocq_map_fold(" in source
    assert "_rocq_set_fold(" in source
    assert "class PositiveMap" not in source
    assert "class PositiveSet" not in source


def test_finite_collection_stdlib_refs_are_filtered_from_module_exports(
    build_default: Path,
    assert_rendered_source: RenderedSourceAssert,
) -> None:
    source = (build_default / "FiniteCollectionFixtures.py").read_text()

    # The module-export form is now ``__Foo_value.x = x`` plus a Protocol
    # cast at the bottom (#1095) — no more direct ``Foo.x = x`` on the
    # Protocol-typed alias.  Same content, different surface.
    assert_rendered_source(
        source,
        "__FiniteCollectionFixtures_value.positive_task_map = positive_task_map",
        (
            "__FiniteCollectionFixtures_value.PositiveMap = PositiveMap",
            "__FiniteCollectionFixtures_value.PositiveSet = PositiveSet",
            "class PositiveMap",
            "class PositiveSet",
        ),
    )
    assert_rendered_source(
        source,
        "positive_task_find_expr: str | None = positive_task_map.get(_rocq_positive_key(p1))",
        (
            "def find(",
            "def mem(",
            "def cardinal(",
        ),
    )
