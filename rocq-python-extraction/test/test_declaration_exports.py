from pathlib import Path


def assert_module_exports(source: str, target: str, names: tuple[str, ...]) -> None:
    for name in names:
        assert f"{target}.{name} = {name}" in source


def assert_module_does_not_export(
    source: str,
    target: str,
    names: tuple[str, ...],
) -> None:
    for name in names:
        assert f"{target}.{name} = {name}" not in source


def test_finite_collection_fixture_exports_stay_stable(build_default: Path) -> None:
    source = (build_default / "FiniteCollectionFixtures.py").read_text()

    assert_module_exports(
        source,
        "FiniteCollectionFixtures",
        (
            "p1",
            "p2",
            "p3",
            "p5",
            "p7",
            "positive_task_map",
            "positive_task_hit",
            "positive_task_missing",
            "positive_task_removed",
            "positive_task_find_expr",
            "positive_task_mem_expr",
            "positive_task_cardinal_expr",
            "positive_task_elements_expr",
            "positive_task_fold_count",
            "positive_task_has_3",
            "positive_task_count",
            "positive_task_elements",
            "positive_claim_set",
            "positive_claim_union",
            "positive_claim_inter",
            "positive_claim_diff",
            "positive_claim_union_expr",
            "positive_claim_inter_expr",
            "positive_claim_diff_expr",
            "positive_claim_nested_expr",
            "positive_claim_union_inter_expr",
            "positive_claim_diff_union_expr",
            "positive_claim_inter_diff_expr",
            "positive_claim_removed",
            "positive_claim_mem_expr",
            "positive_claim_cardinal_expr",
            "positive_claim_elements_expr",
            "positive_claim_fold_count",
            "positive_claim_has_2",
            "positive_claim_count",
            "positive_claim_elements",
            "string_label_map",
            "string_label_hit",
            "string_label_elements",
            "string_label_set",
            "string_label_set_elements",
        ),
    )


def test_finite_collection_stdlib_declarations_stay_out_of_exports(
    build_default: Path,
) -> None:
    source = (build_default / "FiniteCollectionFixtures.py").read_text()

    assert_module_does_not_export(
        source,
        "FiniteCollectionFixtures",
        (
            "PositiveMap",
            "PositiveSet",
            "add",
            "remove",
            "find",
            "mem",
            "cardinal",
            "elements",
            "fold",
        ),
    )
    assert "class PositiveMap" not in source
    assert "class PositiveSet" not in source


def test_runtime_marker_declarations_stay_suppressed(build_default: Path) -> None:
    source = (build_default / "concurrency_primitives.py").read_text()

    for marker_name in (
        "new_mutex",
        "new_channel",
        "new_future",
        "mutex_acquire",
        "mutex_release",
        "channel_send",
        "channel_receive",
        "future_set",
        "future_result",
        "future_done",
    ):
        assert f"def {marker_name}(" not in source
        assert f"{marker_name} =" not in source

    assert "__PYCONC_" not in source
    assert "__PYMONAD_IO_" not in source


def test_custom_aliases_and_record_accessors_stay_classified(
    build_default: Path,
) -> None:
    records_source = (build_default / "records.py").read_text()
    proof_source = (build_default / "proof_pair_zero.py").read_text()

    assert "def pair_r_same(" in records_source
    assert "pair_r_eq =" not in records_source
    assert "__PY_NATIVE_EQ__" not in records_source

    assert "def pfst_r(" not in records_source
    assert "def psnd_r(" not in records_source
    assert "return p.pfst_r" in records_source
    assert "return p.psnd_r" in records_source

    assert "def proof_pair_zero(" in proof_source
    assert "eq_refl" not in proof_source
